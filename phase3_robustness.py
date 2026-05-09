"""
phase3_robustness.py
====================
PHASE 3 — Robustness profile across Byzantine fraction x attack type.

Tests how our SecAgg-compatible recovery method degrades across operating
conditions. Single configuration variable: rho=0.30 (Phase 2 confirmed
strongest). Sweep f and attack.

Hypotheses:
  H3.1 graceful degradation: AUC and accuracy decrease monotonically as f
       increases; ours still beats nodef at f<=0.30.
  H3.2 attack-dependent: sign-flip and scaled give strong signal; label-flip
       weak; gaussian violates A1 (no linear bias) -> near-random AUC.

Grid:
  Byzantine fraction f in {0.10, 0.20, 0.30}
  Attack in {sign_flip, label_flip, scaled, gaussian}
  Method in {ours, nodef}     (nodef as paired baseline per cell)
  Seeds {11, 23}              (2 seeds per cell to manage compute)
  Total: 3 x 4 x 2 x 2 = 48 runs

Compute estimate (T4):
  ~2000 sec/run as observed in Phase 2 ~ 33 min.
  48 runs x 33 min ~ 26 hours -- TOO LONG for one Kaggle session.
  Strategy: split across 2 sessions or 2 accounts. We log defensively so
  partial results survive a session timeout. The script auto-detects an
  existing CSV and skips already-completed (method, f, attack, seed) cells.

Math-to-code mapping (every step traced):
  STEP 1: same metric cohorts as Phase 2 (A2-safe, independent metadata)
  STEP 2: per-client attack assignment (sign_flip / label_flip / scaled / gaussian)
  STEP 3: SecAgg-constrained server (cohort sums only)
  STEP 4: coord-wise median across cohort means (Yin 2018 robustness)
  STEP 5: cosine-to-median anomaly score
  STEP 6: warmup + accumulate scores across rounds
  STEP 7: ridge least-squares CS recovery: x = (A^T A + lam I)^-1 (A^T s + lam pi)
  STEP 8: percentile filter (top-f cohorts excluded)
  STEP 9: weighted FedAvg over kept cohort sums

A2 audit logged per run (pct_cohorts_majority_byz). At f=0.30, P(maj-Byz cohort)
~= 5% per Binomial(10, 0.30) > 5; we expect some cohorts to cross threshold
and report that honestly.
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
)


# =============================================================================
# Algorithmic primitives — copy-paste of phase2_main, intentional
# =============================================================================

def robust_cohort_anomaly_scores(cohort_aggs):
    """STEP 4 + 5: coord-wise median across cohort means; cosine to median maps to [0,1]."""
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
    """STEP 7: ridge LS solve A x = s, projected to [0,1]."""
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


def load_cifar10(data_root="./data"):
    """Use Kaggle's hosted dataset if KAGGLE_CIFAR_DIR is set; else download."""
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
    # Use download=False if data already exists at data_root/cifar-10-batches-py
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
# Single FL run — one (method, f, attack, seed) cell
# =============================================================================

def run_one(method, attack_name, byzantine_frac, seed,
            train_set, test_set,
            num_clients=100, alpha_dirichlet=0.1, rho=0.30,
            cohort_size=10, num_rounds=60,
            sketch_dim=64, local_epochs=1, lr=0.05, batch_size=32,
            log_every=10):
    """One full FL run. method in {'ours', 'nodef'}; attack_name in
    {'sign_flip', 'label_flip', 'scaled', 'gaussian'}.
    """
    assert method in ("ours", "nodef")
    assert attack_name in ("sign_flip", "label_flip", "scaled", "gaussian")
    set_all_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # STEP 1: data partition + Byzantine assignment
    targets = np.array(train_set.targets)
    parts = dirichlet_partition(targets, num_clients, alpha_dirichlet, seed=seed)

    rng = np.random.RandomState(seed + 9999)
    n_byz = int(byzantine_frac * num_clients)
    byzantine_ids = set(rng.choice(num_clients, size=n_byz, replace=False).tolist())
    byzantine_mask = np.array([(i in byzantine_ids) for i in range(num_clients)])

    # STEP 1: independent metadata for cohorts (A2-safe)
    rng_meta = np.random.RandomState(seed + 1000)
    metadata = rng_meta.randn(num_clients, 4)

    num_cohorts = max(2, int(round(rho * num_clients)))
    cohorts = metric_cohorts(metadata, num_cohorts, cohort_size, seed=seed)
    factor_graph = build_factor_graph(cohorts, num_clients)

    # A2 audit
    fracs = np.array([
        sum(1 for i in c if int(i) in byzantine_ids) / max(len(c), 1)
        for c in cohorts
    ])
    pct_majority_byz = float((fracs > 0.5).mean())
    max_cohort_byz_frac = float(fracs.max())

    # Model + sketch
    global_model = ResNet8GN(num_classes=10).to(device)
    n_params = count_params(global_model)
    sketch = JLSketch(in_dim=n_params, out_dim=sketch_dim, seed=seed)
    local_model = ResNet8GN(num_classes=10).to(device)

    # STEP 2: attack object based on attack_name
    # Math: each attack stresses different assumptions (see walkthrough table).
    if attack_name == "sign_flip":
        attack_obj = make_attack("sign_flip")
        label_flip_pairs = {}
    elif attack_name == "scaled":
        attack_obj = make_attack("scaled", factor=10.0)
        label_flip_pairs = {}
    elif attack_name == "gaussian":
        # std chosen to match honest update magnitude order; high enough to mask
        attack_obj = make_attack("gaussian", std=1.0)
        label_flip_pairs = {}
    elif attack_name == "label_flip":
        attack_obj = make_attack("label_flip")
        # Cyclic shift: c -> (c+1) mod 10. Affects all classes equally.
        label_flip_pairs = {c: (c + 1) % 10 for c in range(10)}
    else:
        raise ValueError(attack_name)

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
            label_flip_pairs=label_flip_pairs if (i in byzantine_ids) else {},
        ))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    accumulated_scores = []
    warmup = 3
    history = []
    final_posteriors = None

    for rnd in range(num_rounds):
        global_flat = get_flat_params(global_model)
        results = [c.train(local_model, global_flat, sketch) for c in clients]
        server = SecAggServer(results)
        cohort_aggs = server.aggregate_all_cohorts(cohorts)

        if method == "ours":
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

            # STEP 8: percentile filter (top-byzantine_frac cohorts excluded)
            cohort_post_means = np.array([
                posteriors[c["cohort"]].mean() if c is not None else 1.0
                for c in cohort_aggs
            ])
            n_keep = max(1, int(round((1.0 - byzantine_frac) * len(cohorts))))
            keep_idx = np.argsort(cohort_post_means)[:n_keep]
            good_cohorts = [cohort_aggs[i] for i in keep_idx if cohort_aggs[i] is not None]
            n_filtered = len(cohort_aggs) - len(good_cohorts)
        else:  # nodef
            good_cohorts = [c for c in cohort_aggs if c is not None]
            n_filtered = 0

        # STEP 9: weighted FedAvg
        total_n = sum(c["n_samples"] for c in good_cohorts)
        if total_n > 0:
            agg_delta = sum(
                c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                for c in good_cohorts
            )
            set_flat_params(global_model, global_flat + agg_delta)

        if rnd % log_every == 0 or rnd == num_rounds - 1:
            test_loss, test_acc = evaluate(global_model, test_loader, device)
            history.append({
                "round": rnd, "test_acc": test_acc, "test_loss": test_loss,
                "n_kept": len(good_cohorts), "n_filtered": n_filtered,
            })

    test_loss, test_acc = evaluate(global_model, test_loader, device)

    out = {
        "method": method, "attack": attack_name,
        "byzantine_frac": byzantine_frac, "seed": seed, "rho": rho,
        "num_clients": num_clients, "cohort_size": cohort_size,
        "num_cohorts": num_cohorts, "n_byzantine": int(byzantine_mask.sum()),
        "pct_cohorts_majority_byz": pct_majority_byz,    # A2 audit
        "max_cohort_byz_frac": max_cohort_byz_frac,
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
# Resume support: skip already-completed cells
# =============================================================================

def already_completed_cells(csv_path):
    """Read existing CSV (if any), return set of (method, attack, f, seed) cells already done."""
    if not os.path.exists(csv_path):
        return set()
    try:
        import csv
        done = set()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows that errored out
                if row.get("error", "").strip():
                    continue
                # Skip rows with NaN final_test_acc (incomplete)
                try:
                    acc = float(row["final_test_acc"])
                    if np.isnan(acc):
                        continue
                except (ValueError, KeyError):
                    continue
                key = (
                    row["method"],
                    row["attack"],
                    float(row["byzantine_frac"]),
                    int(row["seed"]),
                )
                done.add(key)
        return done
    except Exception as e:
        print(f"[phase3] could not read existing CSV ({e}); starting fresh")
        return set()


# =============================================================================
# Main sweep
# =============================================================================

def main(out_dir="/kaggle/working/results/phase3", run_id_override=None,
         auto_pusher=None):
    cfg = {
        "phase": "3_robustness",
        "purpose": "robustness profile across Byzantine fraction x attack type",
        "byzantine_fracs": [0.10, 0.20, 0.30],
        "attacks": ["sign_flip", "scaled", "label_flip", "gaussian"],
        "methods": ["ours", "nodef"],
        "seeds": [11, 23],
        "rho": 0.30,                     # validated by Phase 2
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
    run_id = run_id_override or f"phase3_{config_hash(cfg)}"
    print(f"[phase3] run_id={run_id}")

    # All cells
    cells = []
    for f, atk, m, seed in product(
        cfg["byzantine_fracs"], cfg["attacks"], cfg["methods"], cfg["seeds"]
    ):
        cells.append((m, atk, f, seed))
    print(f"[phase3] total cells: {len(cells)}")

    # Resume support
    csv_path_fixed = os.path.join(out_dir, f"{run_id}.csv")
    done = already_completed_cells(csv_path_fixed)
    if done:
        print(f"[phase3] resuming: {len(done)} cells already complete; "
              f"{len(cells) - len(done)} remaining")

    # Load data
    print("[phase3] loading CIFAR-10...")
    data_root = os.environ.get("CIFAR10_ROOT", "./data")
    train_set, test_set = load_cifar10(data_root)
    print(f"[phase3] train={len(train_set)} test={len(test_set)}")

    # Logger: append to existing CSV if resuming
    if not done:
        logger = RunLogger(out_dir, run_id, cfg)
    else:
        # Keep using the same csv path; do NOT overwrite. Construct a logger
        # that re-uses the existing file.
        logger = RunLogger.__new__(RunLogger)
        logger.out_dir = out_dir
        logger.run_id = run_id
        logger.csv_path = csv_path_fixed
        logger.json_path = os.path.join(out_dir, f"{run_id}.json")
        # Use existing file's header as fields list
        import csv as _csv
        with open(csv_path_fixed) as f:
            reader = _csv.reader(f)
            logger.fields = next(reader)
        # Refresh JSON config
        with open(logger.json_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    t_start = time.time()
    skipped = 0
    completed = 0
    failed = 0

    for i, (method, attack, f, seed) in enumerate(cells):
        cell_key = (method, attack, f, seed)
        if cell_key in done:
            skipped += 1
            continue

        t0 = time.time()
        try:
            m = run_one(
                method=method, attack_name=attack, byzantine_frac=f, seed=seed,
                train_set=train_set, test_set=test_set,
                num_clients=cfg["num_clients"],
                alpha_dirichlet=cfg["alpha_dirichlet"],
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
            print(f"[phase3] ERROR cell={cell_key}: {e}")
            print(tb[-1500:])
            m = {
                "method": method, "attack": attack, "byzantine_frac": f, "seed": seed,
                "rho": cfg["rho"], "num_clients": cfg["num_clients"],
                "cohort_size": cfg["cohort_size"], "num_cohorts": 0,
                "n_byzantine": 0, "pct_cohorts_majority_byz": 0.0,
                "max_cohort_byz_frac": 0.0,
                "final_test_acc": float("nan"), "final_test_loss": float("nan"),
                "history": "",
                "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "auc": float("nan"), "posterior_gap": float("nan"),
                "run_seconds": time.time() - t0,
                "error": str(e)[:200],
            }
            failed += 1

        logger.log(**m)

        # Auto-push partial results periodically (survives 12-hour kernel kill)
        if auto_pusher is not None:
            auto_pusher.cell_completed()
            auto_pusher.maybe_push()

        elapsed = time.time() - t_start
        remaining = len(cells) - i - 1 - skipped
        eta_sec = (elapsed / max(completed, 1)) * remaining if completed > 0 else 0
        print(f"[phase3] {i+1}/{len(cells)} m={method:6s} atk={attack:11s} "
              f"f={f:.2f} seed={seed:3d} acc={m.get('final_test_acc', float('nan')):.3f} "
              f"AUC={m.get('auc', float('nan')):.3f} "
              f"({m['run_seconds']/60:.1f}min, eta {eta_sec/60:.0f}min, "
              f"done={completed} skip={skipped} fail={failed})")

    # === Analysis ===
    print(f"\n[phase3] sweep done. completed={completed}, skipped={skipped}, failed={failed}")

    # Read the CSV with proper parser (logger now writes quoted fields)
    import pandas as pd
    df = pd.read_csv(csv_path_fixed)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")
    df["byzantine_frac"] = pd.to_numeric(df["byzantine_frac"], errors="coerce")
    df["posterior_gap"] = pd.to_numeric(df["posterior_gap"], errors="coerce")

    print("\n[phase3] H3.1: ours method, AUC by (attack, f):")
    ours = df[df.method == "ours"].copy()
    if len(ours):
        h31 = ours.groupby(["attack", "byzantine_frac"]).agg(
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
            acc_mean=("final_test_acc", "mean"), acc_std=("final_test_acc", "std"),
            n=("seed", "count"),
        ).round(4)
        print(h31.to_string())

    print("\n[phase3] H3 paired delta (ours - nodef) acc by (attack, f):")
    deltas = []
    for atk in cfg["attacks"]:
        for f in cfg["byzantine_fracs"]:
            o = df[(df.method == "ours") & (df.attack == atk) &
                   (np.abs(df.byzantine_frac - f) < 1e-6)]
            n = df[(df.method == "nodef") & (df.attack == atk) &
                   (np.abs(df.byzantine_frac - f) < 1e-6)]
            if len(o) > 0 and len(n) > 0:
                d = float(o["final_test_acc"].mean() - n["final_test_acc"].mean())
                deltas.append({"attack": atk, "f": f, "delta": d,
                              "ours": float(o["final_test_acc"].mean()),
                              "nodef": float(n["final_test_acc"].mean())})
                print(f"  {atk:11s} f={f:.2f}: ours={d+float(n['final_test_acc'].mean()):.3f} "
                      f"nodef={float(n['final_test_acc'].mean()):.3f} "
                      f"delta={d:+.4f}")

    # === Plots ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # AUC heatmap-style: lines per attack across f
        ax = axes[0]
        colors = {"sign_flip": "#1b9e77", "scaled": "#d95f02",
                  "label_flip": "#7570b3", "gaussian": "#e7298a"}
        for atk in cfg["attacks"]:
            sub = ours[ours.attack == atk]
            if len(sub) == 0: continue
            agg = sub.groupby("byzantine_frac")["auc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["byzantine_frac"], agg["mean"], yerr=agg["std"],
                        marker="o", capsize=3, linewidth=1.6,
                        color=colors[atk], label=atk)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="random")
        ax.set_xlabel("Byzantine fraction f")
        ax.set_ylabel("Detection AUC (ours)")
        ax.set_title("H3: detection by attack and Byzantine fraction")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc="best")

        # Accuracy delta heatmap-style
        ax = axes[1]
        for atk in cfg["attacks"]:
            atk_deltas = [d for d in deltas if d["attack"] == atk]
            if not atk_deltas: continue
            fs = [d["f"] for d in atk_deltas]
            ds = [d["delta"] for d in atk_deltas]
            ax.plot(fs, ds, marker="s", linewidth=1.6, color=colors[atk], label=atk)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Byzantine fraction f")
        ax.set_ylabel("Accuracy delta (ours - nodef)")
        ax.set_title("H3: accuracy benefit by attack and f")
        ax.grid(alpha=0.3); ax.legend(loc="best")

        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"{run_id}_robustness.png")
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"[phase3] plot saved: {plot_path}")
    except Exception as e:
        print(f"[phase3] plot failed: {e}")

    logger.finalize(
        completed=completed, skipped=skipped, failed=failed,
        elapsed_seconds=time.time() - t_start,
        deltas_summary=deltas,
    )

    # Final push of complete results + plot
    if auto_pusher is not None:
        auto_pusher.final_push()

    return completed, skipped, failed


if __name__ == "__main__":
    main()
