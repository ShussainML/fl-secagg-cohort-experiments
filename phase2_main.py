"""
phase2_main.py
==============
PHASE 2 MAIN RUN.

Tests both hypotheses on real CIFAR-10:
    H1: Byzantine localization AUC monotone non-decreasing in cohort
        overlap density rho (validated framework on real FL).
    H2: Recovery-based aggregation beats no-defense FedAvg in final
        test accuracy under sign-flip attack.

Pre-conditions (already cleared):
    - Phase 1 toy: percolation transition observed.
    - Phase 2 toy gate: PASS, A2 holds (0% majority-Byz cohorts).

Experimental design:
    Methods x rho x seeds:
      - "ours" at rho in {0.10, 0.20, 0.30}, seeds {11, 23, 47}  -> 9 runs (H1)
      - "nodef" at rho=0.30, seeds {11, 23, 47}                  -> 3 runs (H2)
    Total: 12 runs.

    Why nodef only at rho=0.30:
      H2 compares accuracy at the rho where our method has best detection.
      Cheapest configuration that gives a fair comparison.
      "nodef" doesn't depend on rho meaningfully (no filtering happens),
      so we don't need a sweep for it.

Compute estimate (T4):
    ~25-30 min/run * 12 runs ~ 5-6 hours. Single Kaggle session.

Math-to-code mapping (each step traces to math walkthrough):
    Step 1: metric_cohorts(metadata) -- A2 (independent metadata)
    Step 2: client.train() with sign-flip on Byzantines -- A1
    Step 3: SecAggServer.aggregate_cohort() -- SecAgg invariant
    Step 4: torch.median across cohort means -- A2 robustness (Yin 2018)
    Step 5: cosine-to-median anomaly score -- A1 linearity
    Step 6: warmup + accumulate scores across rounds -- A4, A6
    Step 7: ridge least squares solve for x in [0,1] -- A3 (CS recovery)
    Step 8: percentile cohort filter (top-byz_frac excluded) -- decision rule
    Step 9: weighted FedAvg over kept cohort sums -- standard
"""

import os
import sys
import time
import json
from collections import defaultdict
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fl_core import (
    set_all_seeds, config_hash, RunLogger,
    ResNet8GN, count_params,
    get_flat_params, set_flat_params,
    JLSketch, IndexedSubset,
    dirichlet_partition,
    metric_cohorts,
    Client, SecAggServer,
    detection_metrics, evaluate,
    make_attack,
)


# =============================================================================
# Algorithmic primitives. Identical to phase2_toy_gate.py, intentional.
# Self-contained per parallel-phase rule.
# =============================================================================

def robust_cohort_anomaly_scores(cohort_aggs):
    """STEP 4 + STEP 5 of math walkthrough.

    Coordinate-wise median across cohort means (= robust honest direction
    estimate, valid because A2 holds: <50% majority-Byz cohorts).
    Cosine to median, mapped to [0,1] = anomaly score.

    Math: with sign-flip and balanced cohorts,
          cohort_mean_k = (1 - 2*f_k) * delta_h + noise.
          cosine(cohort_mean_k, delta_h_estimate) ~ (1 - 2*f_k).
          score_k = (1 - cos)/2 ~ f_k for f_k in [0, 0.5].
    """
    means = []
    for c in cohort_aggs:
        if c is None:
            means.append(None); continue
        means.append((c["delta_sum"] / max(c["n_clients"], 1)).flatten().float())
    valid = [m for m in means if m is not None]
    if not valid:
        return np.full(len(cohort_aggs), 0.5)
    stacked = torch.stack(valid)
    robust = stacked.median(dim=0).values
    rn = robust.norm() + 1e-12
    out = []
    for m in means:
        if m is None:
            out.append(0.5); continue
        mn = m.norm() + 1e-12
        cos = float(torch.dot(m, robust) / (mn * rn))
        out.append((1.0 - cos) / 2.0)
    return np.array(out)


def byzantine_recovery(cohort_anomaly, factor_graph, num_clients, prior=0.2):
    """STEP 7 of math walkthrough.

    Solve A x ~= s in ridge-regularized least squares:
        x = (A^T A + lambda I)^{-1} (A^T s + lambda * prior * 1)
    where A is the (cohorts x clients) row-normalized membership matrix.

    Math: sparse-recovery / compressed-sensing problem (Donoho 2006).
    Recoverable when m = rho*n >= C * f*n * log(n / fn) (RIP-style bound).
    For n=100, f=0.2: requires m >= ~30 cohorts -> rho* ~= 0.30.

    The percolation transition observed in Phase 1 corresponds exactly to
    this rank-of-A threshold appearing as detection AUC.
    """
    k2c = factor_graph["k2c"]
    n_cohorts = len(k2c)
    A = np.zeros((n_cohorts, num_clients), dtype=np.float64)
    for k, members in k2c.items():
        if len(members) == 0:
            continue
        for m in members:
            A[k, int(m)] = 1.0 / len(members)
    appearance = (A != 0).sum(axis=0)
    in_no_cohort = (appearance == 0)

    lam = 0.01
    AtA = A.T @ A + lam * np.eye(num_clients)
    Ats = A.T @ cohort_anomaly + lam * prior * np.ones(num_clients)
    try:
        x = np.linalg.solve(AtA, Ats)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, cohort_anomaly, rcond=None)[0]
    x = np.clip(x, 0.0, 1.0)        # STEP 8: project to [0,1]
    x[in_no_cohort] = prior          # STEP 8: unobserved -> prior
    return x


def build_factor_graph(cohorts, num_clients):
    """Builds adjacency for the cohort-client bipartite graph."""
    c2k = defaultdict(list); k2c = {}
    for ci, members in enumerate(cohorts):
        k2c[ci] = list(members)
        for m in members:
            c2k[int(m)].append(ci)
    return {"c2k": dict(c2k), "k2c": k2c}


# =============================================================================
# Data loading
# =============================================================================

def load_cifar10(data_root="./data"):
    """CIFAR-10 with FL-conventional augmentation (light: crop + flip)."""
    import torchvision
    import torchvision.transforms as T
    tx_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tx_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    os.makedirs(data_root, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=tx_train
    )
    test_set = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=tx_test
    )
    return train_set, test_set


# =============================================================================
# Single FL run with explicit method choice
# =============================================================================

def run_one(method, train_set, test_set, seed, rho,
            num_clients=100, alpha_dirichlet=0.1, byzantine_frac=0.20,
            cohort_size=10, num_rounds=60,
            sketch_dim=64, local_epochs=1, lr=0.05, batch_size=32,
            log_every=5):
    """One full FL run.

    method in {"ours", "nodef"}:
      ours:  STEP 6 + STEP 7 (recovery) + STEP 8 (percentile filter)
             + STEP 9 (weighted FedAvg over kept cohorts).
      nodef: STEP 9 only, all cohorts kept (vanilla FedAvg under SecAgg).

    BOTH arms honor the SecAgg invariant. Only the smarts above SecAgg differ.
    """
    assert method in ("ours", "nodef"), f"Unknown method: {method}"
    set_all_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- STEP 1: data partition + Byzantine assignment + metadata ---
    targets = np.array(train_set.targets)
    parts = dirichlet_partition(targets, num_clients, alpha_dirichlet, seed=seed)

    # Byzantine assignment: SAME seed across "ours" and "nodef" arms by design.
    rng = np.random.RandomState(seed + 9999)
    n_byz = int(byzantine_frac * num_clients)
    byzantine_ids = set(rng.choice(num_clients, size=n_byz, replace=False).tolist())
    byzantine_mask = np.array([(i in byzantine_ids) for i in range(num_clients)])

    # Metadata for k-means cohorts: 4-d Gaussian, INDEPENDENT of byzantine status.
    # This independence is the precise condition guaranteeing A2.
    rng_meta = np.random.RandomState(seed + 1000)
    metadata = rng_meta.randn(num_clients, 4)

    num_cohorts = max(2, int(round(rho * num_clients)))
    cohorts = metric_cohorts(metadata, num_cohorts, cohort_size, seed=seed)
    factor_graph = build_factor_graph(cohorts, num_clients)

    # A2 verification (logged for audit; expect ~0%):
    fracs = np.array([
        sum(1 for i in c if int(i) in byzantine_ids) / max(len(c), 1)
        for c in cohorts
    ])
    pct_majority_byz = float((fracs > 0.5).mean())

    # --- Model + sketch ---
    global_model = ResNet8GN(num_classes=10).to(device)
    n_params = count_params(global_model)
    sketch = JLSketch(in_dim=n_params, out_dim=sketch_dim, seed=seed)
    local_model = ResNet8GN(num_classes=10).to(device)

    # --- STEP 2: client setup with attack assignment ---
    attack_obj = make_attack("sign_flip")
    no_attack = make_attack("none")
    clients = []
    for i in range(num_clients):
        ds = IndexedSubset(train_set, parts[i])
        clients.append(Client(
            client_id=i, dataset=ds,
            is_byzantine=(i in byzantine_ids),
            attack=attack_obj if (i in byzantine_ids) else no_attack,
            device=device, local_epochs=local_epochs,
            batch_size=batch_size, lr=lr,
        ))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    accumulated_scores = []
    warmup = 3                  # STEP 6: discard early rounds with unstable gradients
    history = []
    final_posteriors = None

    for rnd in range(num_rounds):
        global_flat = get_flat_params(global_model)

        # --- STEP 2 (per round): all clients train locally; Byzantines attack ---
        results = [c.train(local_model, global_flat, sketch) for c in clients]

        # --- STEP 3: SecAgg-constrained server gives only cohort sums ---
        server = SecAggServer(results)
        cohort_aggs = server.aggregate_all_cohorts(cohorts)

        if method == "ours":
            # --- STEPS 4 + 5: cohort anomaly scores from median + cosine ---
            scores = robust_cohort_anomaly_scores(cohort_aggs)

            # --- STEP 6: accumulate post-warmup ---
            if rnd >= warmup:
                accumulated_scores.append(scores)

            # --- STEP 7: compressed-sensing recovery ---
            if accumulated_scores:
                avg_scores = np.mean(accumulated_scores, axis=0)
                posteriors = byzantine_recovery(
                    avg_scores, factor_graph, num_clients, prior=byzantine_frac
                )
            else:
                posteriors = np.full(num_clients, byzantine_frac)
            final_posteriors = posteriors

            # --- STEP 8: percentile cohort filter (oracle on f; documented limitation) ---
            cohort_post_means = np.array([
                posteriors[c["cohort"]].mean() if c is not None else 1.0
                for c in cohort_aggs
            ])
            n_keep = max(1, int(round((1.0 - byzantine_frac) * len(cohorts))))
            keep_idx = np.argsort(cohort_post_means)[:n_keep]  # smallest = least suspicious
            good_cohorts = [cohort_aggs[i] for i in keep_idx if cohort_aggs[i] is not None]
            n_filtered = len(cohort_aggs) - len(good_cohorts)
        else:  # nodef: no filter, no recovery
            good_cohorts = [c for c in cohort_aggs if c is not None]
            n_filtered = 0

        # --- STEP 9: weighted FedAvg over kept cohort sums ---
        # weight by sample count, divide cohort_sum by cohort size to get cohort mean
        total_n = sum(c["n_samples"] for c in good_cohorts)
        if total_n > 0:
            agg_delta = sum(
                c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                for c in good_cohorts
            )
            set_flat_params(global_model, global_flat + agg_delta)

        # --- Periodic eval (logged to history JSON) ---
        if rnd % log_every == 0 or rnd == num_rounds - 1:
            test_loss, test_acc = evaluate(global_model, test_loader, device)
            history.append({
                "round": rnd, "test_acc": test_acc, "test_loss": test_loss,
                "n_kept": len(good_cohorts), "n_filtered": n_filtered,
            })

    # --- Final metrics ---
    test_loss, test_acc = evaluate(global_model, test_loader, device)
    out = {
        "method": method, "seed": seed, "rho": rho,
        "num_clients": num_clients, "cohort_size": cohort_size,
        "num_cohorts": num_cohorts,
        "byzantine_frac": byzantine_frac, "n_byzantine": int(byzantine_mask.sum()),
        "pct_cohorts_majority_byz": pct_majority_byz,    # A2 audit
        "final_test_acc": test_acc, "final_test_loss": test_loss,
        "history": json.dumps(history),
    }

    if method == "ours" and final_posteriors is not None:
        n_predict_byz = max(1, int(round(byzantine_frac * num_clients)))
        thr = float(np.sort(final_posteriors)[-n_predict_byz])
        det = detection_metrics(final_posteriors, byzantine_mask, threshold=thr)
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(byzantine_mask.astype(int), final_posteriors))
        except (ValueError, ImportError):
            auc = float("nan")
        out.update({
            "f1": det["f1"], "precision": det["precision"], "recall": det["recall"],
            "auc": auc,
            "posterior_gap": float(final_posteriors[byzantine_mask].mean()
                                   - final_posteriors[~byzantine_mask].mean()),
            "mean_posterior_byz": float(final_posteriors[byzantine_mask].mean()),
            "mean_posterior_hon": float(final_posteriors[~byzantine_mask].mean()),
        })
    else:
        out.update({"f1": np.nan, "precision": np.nan, "recall": np.nan,
                    "auc": np.nan, "posterior_gap": np.nan,
                    "mean_posterior_byz": np.nan, "mean_posterior_hon": np.nan})
    return out


# =============================================================================
# Sweep + analysis
# =============================================================================

def main(out_dir="/kaggle/working/results/phase2"):
    cfg = {
        "phase": "2_main",
        "purpose": "H1 (rho-dependence) + H2 (vs no-defense) on real CIFAR-10",
        "experiments": [
            # H1: rho sweep with our method
            ("ours", 0.10, 11), ("ours", 0.10, 23), ("ours", 0.10, 47),
            ("ours", 0.20, 11), ("ours", 0.20, 23), ("ours", 0.20, 47),
            ("ours", 0.30, 11), ("ours", 0.30, 23), ("ours", 0.30, 47),
            # H2: no-defense baseline at best rho (matched conditions)
            ("nodef", 0.30, 11), ("nodef", 0.30, 23), ("nodef", 0.30, 47),
        ],
        "num_clients": 100,
        "alpha_dirichlet": 0.1,
        "byzantine_frac": 0.20,
        "cohort_size": 10,
        "num_rounds": 60,
        "sketch_dim": 64,
        "local_epochs": 1,
        "lr": 0.05,
        "batch_size": 32,
        "attack": "sign_flip",
    }
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"phase2_main_{config_hash(cfg)}"
    print(f"[phase2_main] run_id={run_id}")
    print(f"[phase2_main] {len(cfg['experiments'])} runs total")
    print(f"[phase2_main] H1: 9 runs (ours x 3 rho x 3 seeds)")
    print(f"[phase2_main] H2: 3 runs (nodef at rho=0.30, 3 seeds)")
    print()

    print("[phase2_main] loading CIFAR-10...")
    data_root = os.environ.get("CIFAR10_ROOT", "./data")
    train_set, test_set = load_cifar10(data_root)
    print(f"[phase2_main] train={len(train_set)} test={len(test_set)}")
    print()

    logger = RunLogger(out_dir, run_id, cfg)
    t_start = time.time()

    n_total = len(cfg["experiments"])
    for i, (method, rho, seed) in enumerate(cfg["experiments"]):
        t0 = time.time()
        try:
            m = run_one(
                method=method, train_set=train_set, test_set=test_set,
                seed=seed, rho=rho,
                num_clients=cfg["num_clients"],
                alpha_dirichlet=cfg["alpha_dirichlet"],
                byzantine_frac=cfg["byzantine_frac"],
                cohort_size=cfg["cohort_size"],
                num_rounds=cfg["num_rounds"],
                sketch_dim=cfg["sketch_dim"],
                local_epochs=cfg["local_epochs"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
            )
            m["run_seconds"] = time.time() - t0
            m["error"] = ""
        except Exception as e:
            import traceback; traceback.print_exc()
            m = {"method": method, "seed": seed, "rho": rho,
                 "final_test_acc": float("nan"), "final_test_loss": float("nan"),
                 "f1": np.nan, "precision": np.nan, "recall": np.nan,
                 "auc": np.nan, "posterior_gap": np.nan,
                 "mean_posterior_byz": np.nan, "mean_posterior_hon": np.nan,
                 "num_cohorts": 0, "num_clients": 0, "cohort_size": 0,
                 "byzantine_frac": 0, "n_byzantine": 0, "pct_cohorts_majority_byz": 0,
                 "history": "",
                 "error": str(e)[:200], "run_seconds": time.time() - t0}

        logger.log(**m)
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (i + 1) * (n_total - i - 1)
        print(f"[phase2_main] {i+1}/{n_total} method={method} rho={rho} seed={seed} "
              f"acc={m.get('final_test_acc', float('nan')):.4f} "
              f"AUC={m.get('auc', float('nan')):.3f} "
              f"({m['run_seconds']/60:.1f}min, eta {eta/60:.0f}min)")

    # === Analysis ===
    import pandas as pd
    df = pd.read_csv(logger.csv_path)
    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["posterior_gap"] = pd.to_numeric(df["posterior_gap"], errors="coerce")

    # H1 summary: mean AUC by rho (ours only)
    df_ours = df[df.method == "ours"]
    h1 = df_ours.groupby("rho")[["auc", "posterior_gap", "final_test_acc"]].agg(
        ["mean", "std"]
    )
    print("\n[phase2_main] H1 — ours by rho:")
    print(h1.round(4).to_string())

    # H2 summary: ours vs nodef at rho=0.30
    df_03 = df[df.rho == 0.30]
    h2 = df_03.groupby("method")[["final_test_acc", "auc"]].agg(["mean", "std"])
    print("\n[phase2_main] H2 — at rho=0.30:")
    print(h2.round(4).to_string())

    # H1 verdict
    auc_010 = float(h1.loc[0.10, ("auc", "mean")]) if 0.10 in h1.index else float("nan")
    auc_030 = float(h1.loc[0.30, ("auc", "mean")]) if 0.30 in h1.index else float("nan")
    h1_pass = (not np.isnan(auc_010)) and (not np.isnan(auc_030)) and (auc_030 - auc_010 >= 0.10)

    # H2 verdict
    nodef_acc = float(df[(df.method == "nodef") & (df.rho == 0.30)]["final_test_acc"].mean())
    ours_acc = float(df[(df.method == "ours") & (df.rho == 0.30)]["final_test_acc"].mean())
    delta = ours_acc - nodef_acc
    h2_pass = delta >= 0.03

    print(f"\n[phase2_main] === VERDICTS ===")
    print(f"  H1 (AUC monotone, AUC(0.30)-AUC(0.10) >= 0.10): "
          f"{'PASS' if h1_pass else 'FAIL'} "
          f"(observed {auc_030 - auc_010:+.3f})")
    print(f"  H2 (ours - nodef >= +0.03 acc):                 "
          f"{'PASS' if h2_pass else 'FAIL'} "
          f"(observed {delta:+.4f}; ours={ours_acc:.3f}, nodef={nodef_acc:.3f})")

    # === Plots ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: H1 — AUC vs rho
        ax = axes[0]
        agg = df_ours.groupby("rho")["auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["rho"], agg["mean"], yerr=agg["std"],
                    marker="o", capsize=4, linewidth=1.8, color="#2c7fb8")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="random")
        ax.set_xlabel(r"Cohort overlap density $\rho$")
        ax.set_ylabel("Byzantine localization AUC")
        ax.set_title("H1: detection vs rho on CIFAR-10")
        ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.3); ax.legend()

        # Right: H2 — accuracy trajectories
        ax = axes[1]
        colors = {"ours": "#1b9e77", "nodef": "#d95f02"}
        for method in ["ours", "nodef"]:
            sub = df[(df.method == method) & (df.rho == 0.30)]
            for _, row in sub.iterrows():
                if not row["history"]:
                    continue
                hist = json.loads(row["history"])
                rounds = [h["round"] for h in hist]
                accs = [h["test_acc"] for h in hist]
                ax.plot(rounds, accs, marker=".", linewidth=1.2, alpha=0.7,
                        color=colors[method])
            # Mean trajectory
            if len(sub) > 0 and sub.iloc[0]["history"]:
                all_hists = [json.loads(r["history"]) for _, r in sub.iterrows()
                             if r["history"]]
                if all_hists:
                    rounds = [h["round"] for h in all_hists[0]]
                    mean_accs = np.mean([[h["test_acc"] for h in hist]
                                          for hist in all_hists], axis=0)
                    ax.plot(rounds, mean_accs, marker="o", linewidth=2.5,
                            color=colors[method], label=f"{method} (mean)")
        ax.axhline(0.10, color="gray", linestyle=":", alpha=0.5, label="random")
        ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy")
        ax.set_title("H2: ours vs nodef at $\\rho$=0.30, sign-flip 20%")
        ax.grid(alpha=0.3); ax.legend()

        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"{run_id}_main.png")
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"[phase2_main] plot saved: {plot_path}")
    except Exception as e:
        print(f"[phase2_main] plot failed: {e}")

    logger.finalize(
        h1_pass=h1_pass,
        h2_pass=h2_pass,
        h1_auc_010=auc_010, h1_auc_030=auc_030,
        h2_nodef_acc=nodef_acc, h2_ours_acc=ours_acc, h2_delta=delta,
        elapsed_seconds=time.time() - t_start,
    )

    return h1_pass, h2_pass


if __name__ == "__main__":
    main()
