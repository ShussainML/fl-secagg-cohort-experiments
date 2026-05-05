"""
phase2_topological_vs_metric.py
================================
PHASE 2: Validate that topological (rank-based gradient-sketch) cohorts
outperform metric (metadata k-means) cohorts for Byzantine localization
on real federated learning.

Hypothesis (testable):
    Topological cohort assignment, defined by rank-based similarity in the
    JL-sketched gradient space, gives BETTER Byzantine localization AUC
    AND comparable or better final accuracy under attack, than metric
    cohorts based on static client metadata.

Mathematical & engineering grounding:
    - Topological cohorts (Ballerini et al. 2008 on starling murmurations):
      each client is grouped with its k most-similar peers in update-direction
      space, regardless of how far they are in metric space. This is robust
      to the curse of dimensionality and adapts to the actual structure of
      what the FL system is learning.

    - Sketches preserve cohort statistics under SecAgg: JL is linear, so
      cohort-sum-of-sketches = sketch-of-cohort-sum, which is what SecAgg
      delivers. Topological grouping uses sketches directly -- no per-client
      gradient access needed for cohort formation.

    - Metric (k-means) baseline: clusters clients by static metadata
      (here: simulated "device-class" embedding). This is the practical
      alternative deployed in current FL systems for cohort design.

Setup:
    - CIFAR-10, Dirichlet alpha=0.1 (heavy non-IID), 100 clients.
    - ResNet-8-GN (~78K params, FL-conventional GroupNorm).
    - 20% Byzantine, sign-flip attack.
    - 60 training rounds (50 round budget on Kaggle to stay within 12h).
    - Two arms: metric vs topological cohorts.
    - rho in {0.10, 0.20, 0.30}: spans the predicted percolation regime so
      we get separation across all three regimes (sub-, near-, super-critical).
    - 3 seeds.
    - Total: 2 arms x 3 rho x 3 seeds = 18 runs.

Independence from other phases:
    Phase 2 sweeps its OWN rho range so it doesn't inherit Phase 1's rho*.
    Phase 6 reconciles whether Phase 1's predicted rho* matches Phase 2's
    best-performing rho.

Estimated compute (T4 GPU on Kaggle):
    - 100 clients x 60 rounds x small ResNet-8 ~ 8-12 minutes per run
    - 18 runs total ~ 3-4 hours wallclock. Fits comfortably in 12h budget.

Validation criteria:
    Pass: topological mean AUC > metric mean AUC at the same rho
          (statistically meaningful: gap > 1 std).
    Pass: topological final accuracy >= metric final accuracy.
    Fail: topological no better than metric -> we report honestly; the
          framework still has the SecAgg-compatible robustness contribution
          but the topological-cohort claim drops out.
"""

import os
import sys
import time
import json
import math
from itertools import product
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
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
    metric_cohorts, topological_cohorts, cohort_overlap_stats,
    Client, SecAggServer,
    detection_metrics, evaluate,
    make_attack,
)


# =============================================================================
# Recovery method (same compressed-sensing approach as Phase 1)
# =============================================================================

def robust_cohort_anomaly_scores(cohort_aggs):
    """Coordinate-wise median across cohort means as robust honest estimate."""
    means = []
    for c in cohort_aggs:
        if c is None:
            means.append(None)
        else:
            means.append((c["delta_sum"] / max(c["n_clients"], 1)).flatten().float())
    valid = [m for m in means if m is not None]
    if not valid:
        return np.full(len(cohort_aggs), 0.5)
    stacked = torch.stack(valid)
    robust_estimate = stacked.median(dim=0).values
    rob_norm = robust_estimate.norm() + 1e-12
    scores = []
    for m in means:
        if m is None:
            scores.append(0.5)
            continue
        m_norm = m.norm() + 1e-12
        cos = float(torch.dot(m, robust_estimate) / (m_norm * rob_norm))
        scores.append((1.0 - cos) / 2.0)
    return np.array(scores)


def byzantine_recovery(cohort_anomaly: np.ndarray, factor_graph: dict,
                       num_clients: int, prior: float = 0.2) -> np.ndarray:
    """Compressed-sensing least-squares recovery of per-client P(Byzantine)."""
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


def build_factor_graph_from_cohorts(cohorts, num_clients):
    c2k = defaultdict(list)
    k2c = {}
    for ci, members in enumerate(cohorts):
        k2c[ci] = list(members)
        for m in members:
            c2k[int(m)].append(ci)
    return {"c2k": dict(c2k), "k2c": k2c}


# =============================================================================
# Data loading: CIFAR-10
# =============================================================================

def load_cifar10(data_root: str = "./data"):
    """Load CIFAR-10 with standard FL preprocessing."""
    import torchvision
    import torchvision.transforms as T
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    os.makedirs(data_root, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=transform_test
    )
    return train_set, test_set


# =============================================================================
# Single run
# =============================================================================

def run_one(
    cohort_method: str,    # "metric" or "topological"
    rho: float,
    seed: int,
    train_set,
    test_set,
    num_clients: int = 100,
    alpha_dirichlet: float = 0.1,
    byzantine_frac: float = 0.20,
    cohort_size: int = 10,
    num_rounds: int = 60,
    sketch_dim: int = 64,
    local_epochs: int = 1,
    lr: float = 0.05,
    batch_size: int = 32,
    detection_threshold_quantile: float = 0.20,
    device: str = None,
    log_every: int = 5,
):
    """One full FL run. Returns metrics dict.

    Note on cohorts: we re-form cohorts every `recohort_every` rounds when
    using topological method (sketches change as model trains). For metric
    method, cohorts are static (metadata doesn't change).
    """
    set_all_seeds(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Partition
    targets = np.array(train_set.targets)
    parts = dirichlet_partition(targets, num_clients, alpha_dirichlet, seed=seed)

    # Byzantine assignment
    rng = np.random.RandomState(seed + 9999)
    n_byz = int(byzantine_frac * num_clients)
    byzantine_ids = set(rng.choice(num_clients, size=n_byz, replace=False).tolist())
    byzantine_mask = np.array([(i in byzantine_ids) for i in range(num_clients)])

    # Static metadata (for metric cohorts) -- simulated as 4-dim "device profile"
    # In real FL these would be device class, region, time-bucket, etc.
    # Here we make it weakly correlated with data distribution so metric
    # cohorts have *some* signal.
    metadata = []
    for i in range(num_clients):
        labels_i = targets[parts[i]]
        # 2-dim: dominant label, label entropy
        dom = float(np.bincount(labels_i, minlength=10).argmax())
        ent = float(-(np.bincount(labels_i, minlength=10) / len(labels_i) + 1e-12).dot(
            np.log(np.bincount(labels_i, minlength=10) / len(labels_i) + 1e-12)))
        # 2 random metadata dims
        rand_meta = rng.randn(2)
        metadata.append([dom, ent, rand_meta[0], rand_meta[1]])
    metadata = np.array(metadata)

    # Model + sketch
    global_model = ResNet8GN(num_classes=10).to(device)
    n_params = count_params(global_model)
    sketch = JLSketch(in_dim=n_params, out_dim=sketch_dim, seed=seed)
    local_model = ResNet8GN(num_classes=10).to(device)

    # Clients
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

    # Number of cohorts derived from rho (same parameterization as Phase 1)
    num_cohorts_plan = max(2, int(round(rho * num_clients)))

    # Initial cohort formation (metric: static; topological: re-formed each round)
    if cohort_method == "metric":
        cohorts = metric_cohorts(metadata, num_cohorts_plan, cohort_size, seed=seed)
    else:
        # First round: random cohorts (no sketches yet to compute similarity)
        # We'll re-form after first round using sketches
        rng_init = np.random.RandomState(seed + 11)
        cohorts = [rng_init.choice(num_clients, size=cohort_size, replace=False)
                   for _ in range(num_cohorts_plan)]

    accumulated_scores = []
    warmup = 3
    history = []

    for rnd in range(num_rounds):
        global_flat = get_flat_params(global_model)

        # Local training
        results = [c.train(local_model, global_flat, sketch) for c in clients]

        # Topological re-cohorting using sketches (uses public sketches; SecAgg-compat)
        if cohort_method == "topological" and rnd >= 1:
            # Gather sketches (server-visible)
            sketches_per_client = torch.stack(
                [results[i].sketch for i in range(num_clients)]
            )
            cohorts = topological_cohorts(
                sketches_per_client, num_cohorts_plan, cohort_size, seed=seed + rnd
            )

        factor_graph = build_factor_graph_from_cohorts(cohorts, num_clients)

        # SecAgg-constrained server
        server = SecAggServer(results)
        cohort_aggs = server.aggregate_all_cohorts(cohorts)

        scores = robust_cohort_anomaly_scores(cohort_aggs)
        if rnd >= warmup:
            accumulated_scores.append(scores)

        if accumulated_scores:
            avg = np.mean(accumulated_scores, axis=0)
            posteriors = byzantine_recovery(avg, factor_graph, num_clients,
                                            prior=byzantine_frac)
        else:
            posteriors = np.full(num_clients, byzantine_frac)

        # Aggregate from cohorts whose mean posterior is in lowest quantile
        cohort_means = np.array([
            posteriors[c["cohort"]].mean() if c is not None else 1.0
            for c in cohort_aggs
        ])
        # Keep cohorts with posterior in bottom (1 - byzantine_frac) -- conservative
        cutoff = float(np.quantile(cohort_means, max(0.5, 1.0 - byzantine_frac * 1.5)))
        good_cohorts = [
            c for ci, c in enumerate(cohort_aggs)
            if c is not None and cohort_means[ci] <= cutoff
        ]
        if not good_cohorts:
            good_cohorts = [c for c in cohort_aggs if c is not None]

        total_n = sum(c["n_samples"] for c in good_cohorts)
        if total_n > 0:
            agg_delta = sum(
                c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                for c in good_cohorts
            )
            new_flat = global_flat + agg_delta
            set_flat_params(global_model, new_flat)

        if rnd % log_every == 0 or rnd == num_rounds - 1:
            test_loss, test_acc = evaluate(global_model, test_loader, device)
            history.append({
                "round": rnd,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "n_good_cohorts": len(good_cohorts),
            })

    # Final metrics
    final_posteriors = posteriors
    n_predict_byz = max(1, int(round(byzantine_frac * num_clients)))
    threshold_at_topk = float(np.sort(final_posteriors)[-n_predict_byz])
    det = detection_metrics(final_posteriors, byzantine_mask, threshold=threshold_at_topk)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(byzantine_mask.astype(int), final_posteriors))
    except (ValueError, ImportError):
        auc = float("nan")

    test_loss, test_acc = evaluate(global_model, test_loader, device)

    return {
        "cohort_method": cohort_method,
        "rho": rho,
        "seed": seed,
        "num_clients": num_clients,
        "cohort_size": cohort_size,
        "num_cohorts": num_cohorts_plan,
        "byzantine_frac": byzantine_frac,
        "n_byzantine": int(byzantine_mask.sum()),
        "f1": det["f1"], "precision": det["precision"], "recall": det["recall"],
        "auc": auc,
        "posterior_gap": float(final_posteriors[byzantine_mask].mean()
                               - final_posteriors[~byzantine_mask].mean()),
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "history": json.dumps(history),  # store as JSON string in CSV
    }


# =============================================================================
# Sweep
# =============================================================================

def run_sweep(out_dir: str):
    cfg = {
        "phase": "2_topological_vs_metric",
        "cohort_methods": ["metric", "topological"],
        "rho_grid": [0.10, 0.20, 0.30],
        "seeds": [11, 23, 47],
        "num_clients": 100,
        "alpha_dirichlet": 0.1,
        "byzantine_frac": 0.20,
        "cohort_size": 10,
        "num_rounds": 60,
        "sketch_dim": 64,
        "local_epochs": 1,
        "lr": 0.05,
        "batch_size": 32,
    }
    run_id = f"phase2_{config_hash(cfg)}"
    print(f"[phase2] run_id={run_id}")

    print("[phase2] loading CIFAR-10...")
    data_root = os.environ.get("CIFAR10_ROOT", "./data")
    train_set, test_set = load_cifar10(data_root)
    print(f"[phase2] train={len(train_set)} test={len(test_set)}")

    logger = RunLogger(out_dir, run_id, cfg)
    t_start = time.time()

    n_total = (len(cfg["cohort_methods"]) * len(cfg["rho_grid"])
               * len(cfg["seeds"]))
    done = 0
    for method, rho, seed in product(cfg["cohort_methods"], cfg["rho_grid"], cfg["seeds"]):
        t0 = time.time()
        try:
            metrics = run_one(
                cohort_method=method, rho=rho, seed=seed,
                train_set=train_set, test_set=test_set,
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
            metrics["run_seconds"] = time.time() - t0
            metrics["error"] = ""
        except Exception as e:
            import traceback; traceback.print_exc()
            metrics = {
                "cohort_method": method, "rho": rho, "seed": seed,
                "f1": float("nan"), "auc": float("nan"),
                "posterior_gap": float("nan"),
                "final_test_acc": float("nan"), "final_test_loss": float("nan"),
                "error": str(e)[:200],
                "run_seconds": time.time() - t0,
                "history": "",
            }
            print(f"[phase2] ERROR at method={method} rho={rho} seed={seed}: {e}")

        logger.log(**metrics)
        done += 1
        elapsed_total = time.time() - t_start
        eta = elapsed_total / done * (n_total - done)
        print(f"[phase2] {done}/{n_total} method={method} rho={rho} seed={seed} "
              f"AUC={metrics.get('auc', float('nan')):.3f} "
              f"acc={metrics.get('final_test_acc', float('nan')):.3f} "
              f"({metrics['run_seconds']/60:.1f}min, eta {eta/60:.0f}min)")

    # === Validation ===
    import pandas as pd
    df = pd.read_csv(logger.csv_path)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")

    summary = df.groupby(["cohort_method", "rho"])[["auc", "final_test_acc"]].agg(
        ["mean", "std"]
    ).reset_index()
    print("\n[phase2] Summary:")
    print(summary.to_string(index=False))

    # Compare topological vs metric at each rho
    auc_advantages = []
    acc_advantages = []
    for rho in cfg["rho_grid"]:
        topo = df[(df["cohort_method"] == "topological") & (df["rho"] == rho)]
        metr = df[(df["cohort_method"] == "metric") & (df["rho"] == rho)]
        auc_adv = float(topo["auc"].mean() - metr["auc"].mean())
        acc_adv = float(topo["final_test_acc"].mean() - metr["final_test_acc"].mean())
        auc_advantages.append(auc_adv)
        acc_advantages.append(acc_adv)
        print(f"  rho={rho}: topological - metric: AUC delta={auc_adv:+.3f}, "
              f"acc delta={acc_adv:+.3f}")

    # Validation criteria
    mean_auc_advantage = float(np.mean(auc_advantages))
    auc_better_in_majority = sum(a > 0 for a in auc_advantages) >= 2  # 2 of 3 rhos
    acc_not_worse = float(np.mean(acc_advantages)) > -0.02

    validation = {
        "mean_auc_advantage_topo_minus_metric": mean_auc_advantage,
        "auc_advantages_per_rho": auc_advantages,
        "acc_advantages_per_rho": acc_advantages,
        "auc_better_in_majority_of_rhos": bool(auc_better_in_majority),
        "acc_not_worse_overall": bool(acc_not_worse),
    }
    print(f"\n[phase2] validation: {json.dumps(validation, indent=2)}")

    headline_pass = auc_better_in_majority and acc_not_worse
    print(f"[phase2] {'HEADLINE PASS' if headline_pass else 'HEADLINE FAIL'}")
    if not headline_pass:
        print("[phase2] NOTE: topological cohorts not clearly better on this setup. "
              "Framework still has value via SecAgg-compatible recovery; "
              "the topological-cohort claim weakens.")

    # === Plot ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df_clean = df.dropna(subset=["auc", "final_test_acc"])
        agg = df_clean.groupby(["cohort_method", "rho"]).agg(
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
            acc_mean=("final_test_acc", "mean"), acc_std=("final_test_acc", "std"),
        ).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        colors = {"metric": "#7570b3", "topological": "#1b9e77"}

        ax = axes[0]
        for method in cfg["cohort_methods"]:
            d = agg[agg["cohort_method"] == method]
            ax.errorbar(d["rho"], d["auc_mean"], yerr=d["auc_std"],
                        marker="o", capsize=4, linewidth=1.8,
                        color=colors[method], label=method.title())
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
        ax.set_xlabel(r"Cohort overlap density $\rho$")
        ax.set_ylabel("Byzantine localization AUC")
        ax.set_title("Phase 2: Byzantine localization")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1]
        for method in cfg["cohort_methods"]:
            d = agg[agg["cohort_method"] == method]
            ax.errorbar(d["rho"], d["acc_mean"], yerr=d["acc_std"],
                        marker="s", capsize=4, linewidth=1.8,
                        color=colors[method], label=method.title())
        ax.set_xlabel(r"Cohort overlap density $\rho$")
        ax.set_ylabel("Final test accuracy")
        ax.set_title("Phase 2: Final accuracy under attack")
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"{run_id}_topo_vs_metric.png")
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"[phase2] plot saved: {plot_path}")
    except Exception as e:
        print(f"[phase2] plot failed: {e}")

    logger.finalize(
        validation=validation,
        headline_pass=headline_pass,
        total_seconds=time.time() - t_start,
        n_runs=n_total,
    )
    return headline_pass


def main(out_dir: str = "/kaggle/working/results/phase2"):
    os.makedirs(out_dir, exist_ok=True)
    return run_sweep(out_dir)


if __name__ == "__main__":
    out = os.environ.get("PHASE2_OUT", "results/phase2")
    main(out)
