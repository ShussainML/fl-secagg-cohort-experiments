"""
phase4_baselines.py
===================
PHASE 4 — Head-to-head comparison against non-SecAgg robust baselines.

The natural reviewer question after Phase 2: "If you let yourself violate
SecAgg, you could use Krum or Trimmed Mean. Are you really competitive?"

This phase answers it by running standard non-SecAgg baselines under
matched conditions and reporting the comparison honestly.

Hypotheses:
  H4.1 competitive: ours within 5% accuracy of Krum / Trimmed Mean / Median /
       FLTrust-lite, while satisfying SecAgg (which they do not).
  H4.2 dominates random selection under SecAgg: ours > nodef (replicates
       Phase 2 H2 with fresh seeds).

Configuration (matched to Phase 2 main):
  CIFAR-10, 100 clients, Dirichlet alpha=0.1
  Byzantine fraction f=0.20, sign-flip attack
  rho=0.30, cohort_size=10, 60 rounds, ResNet-8-GN

Methods:
  - "nodef":      vanilla weighted FedAvg, all clients (SecAgg-respecting)
  - "ours":       compressed-sensing recovery + percentile filter (SecAgg)
  - "krum":       Krum aggregation (Blanchard 2017) -- VIOLATES SecAgg
  - "trimmed":    coordinate-wise trimmed mean (Yin 2018) -- VIOLATES SecAgg
  - "median":     coordinate-wise median (Yin 2018) -- VIOLATES SecAgg
  - "fltrust":    FLTrust-lite (no root dataset) -- VIOLATES SecAgg
  3 seeds per method.
  Total: 6 x 3 = 18 runs. ~10 hours on T4. Feasible in one session;
  resume support included if it gets cut off.

SecAgg invariant respected? Per method:
  ours, nodef:  YES — only cohort sums used.
  krum, trimmed, median, fltrust: NO — they need per-client deltas.
       We use SecAggServer.allow_individual_access=True with logging so
       the audit trail explicitly records the violation.

This is the right comparison. We do NOT cheat by giving these baselines
SecAgg-incompatible information secretly. We flag the violation in the
output table; the paper says "ours achieves X within Y of [non-private
baselines] while preserving SecAgg".
"""

import os
import sys
import time
import json
import traceback
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
    krum_aggregate, trimmed_mean_aggregate, median_aggregate,
)


# =============================================================================
# Algorithmic primitives for ours method (copy-paste from phase2_main)
# =============================================================================

def robust_cohort_anomaly_scores(cohort_aggs):
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
    x = np.clip(x, 0.0, 1.0)
    x[in_no_cohort] = prior
    return x


def build_factor_graph(cohorts, num_clients):
    c2k = defaultdict(list); k2c = {}
    for ci, members in enumerate(cohorts):
        k2c[ci] = list(members)
        for m in members:
            c2k[int(m)].append(ci)
    return {"c2k": dict(c2k), "k2c": k2c}


# =============================================================================
# FLTrust-lite (no-root version): cosine-similarity weighted aggregation
# =============================================================================

def fltrust_lite_aggregate(individual_deltas, individual_n_samples):
    """FLTrust-lite without a trusted root dataset.

    Original FLTrust (Cao 2021) uses a server-held root dataset to compute a
    reference gradient g_0; trust score t_i = ReLU(cos(g_i, g_0)); aggregates
    by trust-weighted average.

    No-root version: replace g_0 with the *mean* of all individual gradients.
    Each client's trust = ReLU(cos(g_i, g_mean)). This is a simpler heuristic
    that's been used in similar form in robust aggregation literature.

    NOT SecAgg compatible: requires per-client deltas.
    """
    deltas = torch.stack(individual_deltas)  # (n, d)
    g_mean = deltas.mean(dim=0)
    gm_norm = g_mean.norm() + 1e-12
    trust = []
    for d in deltas:
        cs = float(torch.dot(d, g_mean) / (d.norm() * gm_norm + 1e-12))
        trust.append(max(0.0, cs))
    trust = np.asarray(trust)
    if trust.sum() == 0:
        return deltas.mean(dim=0)  # fallback
    weights = trust / trust.sum()
    weights_t = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
    return (deltas * weights_t.unsqueeze(1)).sum(dim=0)


def load_cifar10(data_root="./data"):
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
    expected = os.path.join(data_root, "cifar-10-batches-py")
    do_download = not os.path.exists(expected)
    train_set = torchvision.datasets.CIFAR10(
        data_root, train=True, download=do_download, transform=tx_train
    )
    test_set = torchvision.datasets.CIFAR10(
        data_root, train=False, download=do_download, transform=tx_test
    )
    return train_set, test_set


# =============================================================================
# Single FL run with method-dispatched aggregation
# =============================================================================

def run_one(method, train_set, test_set, seed,
            num_clients=100, alpha_dirichlet=0.1, byzantine_frac=0.20,
            rho=0.30, cohort_size=10, num_rounds=60,
            sketch_dim=64, local_epochs=1, lr=0.05, batch_size=32,
            log_every=10):
    """One FL run.
    method in {"nodef", "ours", "krum", "trimmed", "median", "fltrust"}.
    SecAgg compatibility per method tracked in returned dict.
    """
    valid = ("nodef", "ours", "krum", "trimmed", "median", "fltrust")
    assert method in valid, f"Unknown method: {method}"
    set_all_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Common setup (same as Phase 2/3)
    targets = np.array(train_set.targets)
    parts = dirichlet_partition(targets, num_clients, alpha_dirichlet, seed=seed)

    rng = np.random.RandomState(seed + 9999)
    n_byz = int(byzantine_frac * num_clients)
    byzantine_ids = set(rng.choice(num_clients, size=n_byz, replace=False).tolist())
    byzantine_mask = np.array([(i in byzantine_ids) for i in range(num_clients)])

    rng_meta = np.random.RandomState(seed + 1000)
    metadata = rng_meta.randn(num_clients, 4)

    num_cohorts = max(2, int(round(rho * num_clients)))
    cohorts = metric_cohorts(metadata, num_cohorts, cohort_size, seed=seed)
    factor_graph = build_factor_graph(cohorts, num_clients)

    fracs = np.array([
        sum(1 for i in c if int(i) in byzantine_ids) / max(len(c), 1)
        for c in cohorts
    ])
    pct_majority_byz = float((fracs > 0.5).mean())

    global_model = ResNet8GN(num_classes=10).to(device)
    n_params = count_params(global_model)
    sketch = JLSketch(in_dim=n_params, out_dim=sketch_dim, seed=seed)
    local_model = ResNet8GN(num_classes=10).to(device)

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

    # Methods that need per-client access -> use escape hatch with audit logging
    needs_individual = method in ("krum", "trimmed", "median", "fltrust")
    secagg_compatible = method in ("ours", "nodef")

    accumulated_scores = []
    warmup = 3
    history = []
    final_posteriors = None

    for rnd in range(num_rounds):
        global_flat = get_flat_params(global_model)
        results = [c.train(local_model, global_flat, sketch) for c in clients]

        # Server with appropriate access mode
        server = SecAggServer(results, allow_individual_access=needs_individual)

        if method == "nodef":
            cohort_aggs = server.aggregate_all_cohorts(cohorts)
            good_cohorts = [c for c in cohort_aggs if c is not None]
            total_n = sum(c["n_samples"] for c in good_cohorts)
            if total_n > 0:
                agg_delta = sum(
                    c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                    for c in good_cohorts
                )
                set_flat_params(global_model, global_flat + agg_delta)

        elif method == "ours":
            cohort_aggs = server.aggregate_all_cohorts(cohorts)
            scores = robust_cohort_anomaly_scores(cohort_aggs)
            if rnd >= warmup:
                accumulated_scores.append(scores)
            if accumulated_scores:
                avg_scores = np.mean(accumulated_scores, axis=0)
                posteriors = byzantine_recovery(
                    avg_scores, factor_graph, num_clients, prior=byzantine_frac
                )
            else:
                posteriors = np.full(num_clients, byzantine_frac)
            final_posteriors = posteriors

            cohort_post_means = np.array([
                posteriors[c["cohort"]].mean() if c is not None else 1.0
                for c in cohort_aggs
            ])
            n_keep = max(1, int(round((1.0 - byzantine_frac) * len(cohorts))))
            keep_idx = np.argsort(cohort_post_means)[:n_keep]
            good_cohorts = [cohort_aggs[i] for i in keep_idx if cohort_aggs[i] is not None]
            total_n = sum(c["n_samples"] for c in good_cohorts)
            if total_n > 0:
                agg_delta = sum(
                    c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                    for c in good_cohorts
                )
                set_flat_params(global_model, global_flat + agg_delta)

        else:
            # Non-SecAgg baselines: read individuals (logged as audit trail)
            individuals = [server.get_individual(i) for i in range(num_clients)]
            indiv_deltas = [r.delta for r in individuals]
            indiv_n = [r.n_samples for r in individuals]

            if method == "krum":
                agg_delta = krum_aggregate(individuals, num_byzantine=n_byz)
            elif method == "trimmed":
                # Yin 2018: trim_ratio is the per-side fraction; total trimmed = 2*trim_ratio
                # Set trim_ratio = byzantine_frac to be conservative.
                agg_delta = trimmed_mean_aggregate(individuals, trim_ratio=byzantine_frac)
            elif method == "median":
                agg_delta = median_aggregate(individuals)
            elif method == "fltrust":
                agg_delta = fltrust_lite_aggregate(indiv_deltas, indiv_n)
            else:
                raise ValueError(method)

            set_flat_params(global_model, global_flat + agg_delta)

        if rnd % log_every == 0 or rnd == num_rounds - 1:
            test_loss, test_acc = evaluate(global_model, test_loader, device)
            history.append({
                "round": rnd, "test_acc": test_acc, "test_loss": test_loss,
            })

    test_loss, test_acc = evaluate(global_model, test_loader, device)

    out = {
        "method": method, "seed": seed,
        "secagg_compatible": int(secagg_compatible),
        "rho": rho, "byzantine_frac": byzantine_frac,
        "num_clients": num_clients, "cohort_size": cohort_size,
        "num_cohorts": num_cohorts, "n_byzantine": int(byzantine_mask.sum()),
        "pct_cohorts_majority_byz": pct_majority_byz,
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
        })
    else:
        out.update({"f1": np.nan, "precision": np.nan, "recall": np.nan,
                    "auc": np.nan, "posterior_gap": np.nan})
    return out


# =============================================================================
# Resume support
# =============================================================================

def already_completed_cells(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        import csv
        done = set()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("error", "").strip():
                    continue
                try:
                    acc = float(row["final_test_acc"])
                    if np.isnan(acc):
                        continue
                except (ValueError, KeyError):
                    continue
                done.add((row["method"], int(row["seed"])))
        return done
    except Exception as e:
        print(f"[phase4] could not read existing CSV ({e}); starting fresh")
        return set()


def main(out_dir="/kaggle/working/results/phase4", run_id_override=None,
         auto_pusher=None):
    cfg = {
        "phase": "4_baselines",
        "purpose": "head-to-head against non-SecAgg robust baselines",
        "methods": ["nodef", "ours", "krum", "trimmed", "median", "fltrust"],
        "seeds": [11, 23, 47],
        "rho": 0.30,
        "byzantine_frac": 0.20,
        "attack": "sign_flip",
        "num_clients": 100,
        "alpha_dirichlet": 0.1,
        "cohort_size": 10,
        "num_rounds": 60,
        "sketch_dim": 64,
        "local_epochs": 1,
        "lr": 0.05,
        "batch_size": 32,
    }
    os.makedirs(out_dir, exist_ok=True)
    run_id = run_id_override or f"phase4_{config_hash(cfg)}"
    print(f"[phase4] run_id={run_id}")

    cells = list(product(cfg["methods"], cfg["seeds"]))
    print(f"[phase4] total cells: {len(cells)}")

    csv_path_fixed = os.path.join(out_dir, f"{run_id}.csv")
    done = already_completed_cells(csv_path_fixed)
    if done:
        print(f"[phase4] resuming: {len(done)} cells already complete")

    print("[phase4] loading CIFAR-10...")
    data_root = os.environ.get("CIFAR10_ROOT", "./data")
    train_set, test_set = load_cifar10(data_root)
    print(f"[phase4] train={len(train_set)} test={len(test_set)}")

    if not done:
        logger = RunLogger(out_dir, run_id, cfg)
    else:
        logger = RunLogger.__new__(RunLogger)
        logger.out_dir = out_dir
        logger.run_id = run_id
        logger.csv_path = csv_path_fixed
        logger.json_path = os.path.join(out_dir, f"{run_id}.json")
        import csv as _csv
        with open(csv_path_fixed) as f:
            reader = _csv.reader(f)
            logger.fields = next(reader)
        with open(logger.json_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    t_start = time.time()
    completed = skipped = failed = 0

    for i, (method, seed) in enumerate(cells):
        if (method, seed) in done:
            skipped += 1
            continue
        t0 = time.time()
        try:
            m = run_one(
                method=method, train_set=train_set, test_set=test_set, seed=seed,
                num_clients=cfg["num_clients"],
                alpha_dirichlet=cfg["alpha_dirichlet"],
                byzantine_frac=cfg["byzantine_frac"],
                rho=cfg["rho"],
                cohort_size=cfg["cohort_size"],
                num_rounds=cfg["num_rounds"],
                sketch_dim=cfg["sketch_dim"],
                local_epochs=cfg["local_epochs"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
            )
            m["run_seconds"] = time.time() - t0
            m["error"] = ""
            completed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[phase4] ERROR cell=({method}, {seed}): {e}")
            print(tb[-1500:])
            m = {"method": method, "seed": seed,
                 "secagg_compatible": int(method in ("ours", "nodef")),
                 "rho": cfg["rho"], "byzantine_frac": cfg["byzantine_frac"],
                 "num_clients": 0, "cohort_size": 0, "num_cohorts": 0,
                 "n_byzantine": 0, "pct_cohorts_majority_byz": 0.0,
                 "final_test_acc": float("nan"), "final_test_loss": float("nan"),
                 "history": "",
                 "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                 "auc": float("nan"), "posterior_gap": float("nan"),
                 "run_seconds": time.time() - t0,
                 "error": str(e)[:200]}
            failed += 1

        logger.log(**m)

        # Auto-push partial results periodically (survives 12-hour kernel kill)
        if auto_pusher is not None:
            auto_pusher.cell_completed()
            auto_pusher.maybe_push()

        elapsed = time.time() - t_start
        remaining = len(cells) - i - 1 - skipped
        eta_sec = (elapsed / max(completed, 1)) * remaining if completed > 0 else 0
        print(f"[phase4] {i+1}/{len(cells)} m={method:8s} seed={seed:3d} "
              f"acc={m.get('final_test_acc', float('nan')):.4f} "
              f"({m['run_seconds']/60:.1f}min, eta {eta_sec/60:.0f}min, "
              f"done={completed} skip={skipped} fail={failed})")

    # === Analysis ===
    print(f"\n[phase4] sweep done. completed={completed}, skipped={skipped}, failed={failed}")

    import pandas as pd
    df = pd.read_csv(csv_path_fixed)
    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")
    df["secagg_compatible"] = pd.to_numeric(df["secagg_compatible"], errors="coerce")

    summary = df.groupby("method").agg(
        secagg=("secagg_compatible", "first"),
        acc_mean=("final_test_acc", "mean"),
        acc_std=("final_test_acc", "std"),
        n=("seed", "count"),
    ).round(4)
    # Sort: SecAgg-compatible first, then by accuracy desc
    summary = summary.reset_index()
    summary["sort_key"] = -summary["secagg"]
    summary = summary.sort_values(["sort_key", "acc_mean"], ascending=[True, False])
    summary = summary.drop(columns="sort_key").set_index("method")
    print("\n[phase4] Method comparison (sorted by SecAgg + accuracy):")
    print(summary.to_string())

    # H4.1 verdict: ours within 5% of best non-SecAgg
    if "ours" in summary.index:
        ours_acc = float(summary.loc["ours", "acc_mean"])
        non_secagg = summary[summary["secagg"] == 0]
        if len(non_secagg) > 0:
            best_nonsec_method = non_secagg["acc_mean"].idxmax()
            best_nonsec_acc = float(non_secagg["acc_mean"].max())
            gap = best_nonsec_acc - ours_acc
            h4_1 = gap <= 0.05
            print(f"\n[phase4] H4.1 (ours within 5% of best non-SecAgg):")
            print(f"  ours acc = {ours_acc:.4f}")
            print(f"  best non-SecAgg = {best_nonsec_method} acc = {best_nonsec_acc:.4f}")
            print(f"  gap = {gap:+.4f}")
            print(f"  {'PASS' if h4_1 else 'FAIL'}")

    if "nodef" in summary.index and "ours" in summary.index:
        ours_acc = float(summary.loc["ours", "acc_mean"])
        nodef_acc = float(summary.loc["nodef", "acc_mean"])
        h4_2 = (ours_acc - nodef_acc) >= 0.03
        print(f"\n[phase4] H4.2 (ours - nodef >= +0.03):")
        print(f"  delta = {ours_acc - nodef_acc:+.4f}")
        print(f"  {'PASS' if h4_2 else 'FAIL'}")

    # === Plot ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        # Bar chart with hatched pattern for non-SecAgg
        methods_order = list(summary.index)
        accs = [float(summary.loc[m, "acc_mean"]) for m in methods_order]
        stds = [float(summary.loc[m, "acc_std"]) for m in methods_order]
        secagg_status = [int(summary.loc[m, "secagg"]) for m in methods_order]
        colors = ["#1b9e77" if s == 1 else "#d95f02" for s in secagg_status]
        hatches = ["" if s == 1 else "//" for s in secagg_status]

        bars = ax.bar(range(len(methods_order)), accs, yerr=stds,
                      color=colors, capsize=5, edgecolor="black", linewidth=0.8)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.axhline(0.10, color="gray", linestyle=":", alpha=0.5, label="random")
        ax.set_xticks(range(len(methods_order)))
        ax.set_xticklabels(methods_order, rotation=15)
        ax.set_ylabel("Final test accuracy")
        ax.set_title("Phase 4: ours (SecAgg-compatible, green) vs non-SecAgg robust baselines (orange, hatched)")
        ax.grid(axis="y", alpha=0.3)
        for i, (m, a) in enumerate(zip(methods_order, accs)):
            ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"{run_id}_baselines.png")
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"[phase4] plot saved: {plot_path}")
    except Exception as e:
        print(f"[phase4] plot failed: {e}")

    logger.finalize(
        completed=completed, skipped=skipped, failed=failed,
        elapsed_seconds=time.time() - t_start,
    )

    # Final push of complete results + plot
    if auto_pusher is not None:
        auto_pusher.final_push()

    return completed, skipped, failed


if __name__ == "__main__":
    main()
