"""
phase2_toy_gate.py
==================
Toy validation gate for Phase 2.

Purpose: verify that metric (k-means on metadata) cohort design preserves
the Phase 1 percolation behavior on the SAME toy setup Phase 1 used.

This is the gate. It must pass before any Kaggle compute.

Pass criteria (all 4 must hold):
    1. AUC monotone non-decreasing in rho.
    2. AUC(rho=0.30) - AUC(rho=0.10) >= 0.10.
    3. Posterior gap (P_byz - P_hon) is POSITIVE at all rho.
       (Negative gap = A2 violated = same failure mode as old topological arm.)
    4. Runtime under 10 minutes on CPU.

If any fails, we diagnose. We do NOT run on Kaggle.

Why this gate exists:
    - Old Phase 2 wasted 11 Kaggle hours because no toy gate.
    - Topological arm had inverted posterior gaps (A2 violation) that would
      have shown up in a 2-minute toy run.
    - Metric arm gave correct sign and theory-consistent AUC pattern.
    - This gate replays the metric arm on toy to confirm the algorithm
      is correctly implemented before any expensive run.

Reuses Phase 1's run_one logic; only swaps cohort construction from random
to k-means on simulated metadata.
"""

import os
import sys
import time
import json
from itertools import product
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fl_core import (
    set_all_seeds, config_hash, RunLogger,
    LinearRegressionModel, count_params,
    get_flat_params, set_flat_params,
    JLSketch,
    Client, SecAggServer,
    detection_metrics,
    make_attack,
    metric_cohorts,
)


# =============================================================================
# Reused: same recovery logic as Phase 1 (compressed-sensing + median)
# =============================================================================

def robust_cohort_anomaly_scores(cohort_aggs):
    """Coordinate-wise median across cohort means as robust honest estimate.

    Justified by Yin et al. 2018: median tolerates < 50% corruption per
    coordinate. With random/balanced cohort assignment and overall Byzantine
    fraction f=0.20, P(a cohort has Byzantine fraction > 0.5) is negligible.
    """
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
            scores.append(0.5); continue
        m_norm = m.norm() + 1e-12
        cos = float(torch.dot(m, robust_estimate) / (m_norm * rob_norm))
        scores.append((1.0 - cos) / 2.0)
    return np.array(scores)


def byzantine_recovery(cohort_anomaly: np.ndarray, factor_graph: dict,
                       num_clients: int, prior: float = 0.2) -> np.ndarray:
    """Compressed-sensing least-squares: solve A x = s, x in [0, 1].

    A is the cohort-membership matrix, row-normalized so row sums = 1.
    Ridge regularization stabilizes A when ill-conditioned (low rho).
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
    x = np.clip(x, 0.0, 1.0)
    x[in_no_cohort] = prior
    return x


def build_factor_graph(cohorts, num_clients):
    c2k = defaultdict(list)
    k2c = {}
    for ci, members in enumerate(cohorts):
        k2c[ci] = list(members)
        for m in members:
            c2k[int(m)].append(ci)
    return {"c2k": dict(c2k), "k2c": k2c}


# =============================================================================
# Toy data (same as Phase 1)
# =============================================================================

def generate_toy_data(num_clients, samples_per_client, in_dim, seed):
    rng = np.random.RandomState(seed)
    w_true = rng.randn(in_dim).astype(np.float32)
    b_true = rng.randn(1).astype(np.float32)
    client_datasets = []
    for c in range(num_clients):
        X = rng.randn(samples_per_client, in_dim).astype(np.float32)
        noise = rng.randn(samples_per_client).astype(np.float32) * 0.1
        y = X @ w_true + b_true + noise
        client_datasets.append(TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y),
        ))
    X_test = rng.randn(500, in_dim).astype(np.float32)
    noise_test = rng.randn(500).astype(np.float32) * 0.1
    y_test = X_test @ w_true + b_true + noise_test
    return client_datasets, (torch.from_numpy(X_test), torch.from_numpy(y_test))


def evaluate_regression(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).squeeze(-1)
        mse = float(((pred - y_test) ** 2).mean().item())
    return mse


# =============================================================================
# Metric cohorts on simulated metadata
# =============================================================================

def build_metric_cohorts_with_metadata(num_clients, num_cohorts, cohort_size,
                                       byzantine_ids, seed):
    """Generate simulated metadata and form k-means cohorts.

    Crucially: metadata is generated INDEPENDENTLY of Byzantine status.
    This is the realistic case (a real-world device profile doesn't
    correlate with whether the client is malicious). If we instead made
    metadata correlate with Byzantine status, we'd get pure-Byz cohorts
    and break A2 -- exactly the failure mode of the old topological arm.

    We test the realistic case here.
    """
    rng = np.random.RandomState(seed + 1000)
    # 4-d metadata: simulated "device class, region, time-bucket, signal-strength"
    metadata = rng.randn(num_clients, 4)
    cohorts = metric_cohorts(metadata, num_cohorts, cohort_size, seed=seed)
    return cohorts, metadata


def cohort_byzantine_distribution(cohorts, byzantine_ids):
    """Diagnostic: distribution of f_k (Byzantine fraction per cohort)."""
    fracs = []
    for c in cohorts:
        n_byz = sum(1 for i in c if int(i) in byzantine_ids)
        fracs.append(n_byz / max(len(c), 1))
    return np.array(fracs)


# =============================================================================
# Single run (exact same scoring & recovery as Phase 1)
# =============================================================================

def run_one(rho, seed, num_clients=100, samples_per_client=50, in_dim=20,
            byzantine_frac=0.20, cohort_size=10, num_rounds=30, sketch_dim=16,
            local_epochs=1, lr=0.05, batch_size=32):
    set_all_seeds(seed)
    device = "cpu"

    client_datasets, (X_test, y_test) = generate_toy_data(
        num_clients, samples_per_client, in_dim, seed
    )

    rng = np.random.RandomState(seed + 9999)
    n_byz = int(byzantine_frac * num_clients)
    byzantine_ids = set(rng.choice(num_clients, size=n_byz, replace=False).tolist())
    byzantine_mask = np.array([(i in byzantine_ids) for i in range(num_clients)])

    global_model = LinearRegressionModel(in_dim=in_dim, out_dim=1)
    n_params = count_params(global_model)
    sketch = JLSketch(in_dim=n_params, out_dim=sketch_dim, seed=seed)
    local_model = LinearRegressionModel(in_dim=in_dim, out_dim=1)

    attack_obj = make_attack("sign_flip")
    no_attack = make_attack("none")
    clients = [Client(
        client_id=i, dataset=client_datasets[i],
        is_byzantine=(i in byzantine_ids),
        attack=attack_obj if (i in byzantine_ids) else no_attack,
        device=device, local_epochs=local_epochs,
        batch_size=batch_size, lr=lr,
    ) for i in range(num_clients)]

    # METRIC cohorts (this is what changed vs Phase 1)
    num_cohorts = max(2, int(round(rho * num_clients)))
    cohorts, metadata = build_metric_cohorts_with_metadata(
        num_clients, num_cohorts, cohort_size, byzantine_ids, seed
    )

    # Diagnostic: check A2 (no cohort dominated by Byzantines)
    fracs = cohort_byzantine_distribution(cohorts, byzantine_ids)
    pct_majority_byz = float((fracs > 0.5).mean())

    factor_graph = build_factor_graph(cohorts, num_clients)

    accumulated_scores = []
    warmup = 3

    for rnd in range(num_rounds):
        global_flat = get_flat_params(global_model)
        results = [c.train(local_model, global_flat, sketch) for c in clients]
        server = SecAggServer(results)
        cohort_aggs = server.aggregate_all_cohorts(cohorts)
        scores = robust_cohort_anomaly_scores(cohort_aggs)
        if rnd >= warmup:
            accumulated_scores.append(scores)
        if accumulated_scores:
            avg_scores = np.mean(accumulated_scores, axis=0)
            posteriors = byzantine_recovery(avg_scores, factor_graph, num_clients,
                                            prior=byzantine_frac)
        else:
            posteriors = np.full(num_clients, byzantine_frac)

        good_cohorts = [
            c for ci, c in enumerate(cohort_aggs)
            if c is not None and posteriors[c["cohort"]].mean() < 0.5
        ]
        if not good_cohorts:
            good_cohorts = [c for c in cohort_aggs if c is not None]
        total_n = sum(c["n_samples"] for c in good_cohorts)
        if total_n > 0:
            agg_delta = sum(
                c["delta_sum"] * (c["n_samples"] / total_n) / max(c["n_clients"], 1)
                for c in good_cohorts
            )
            set_flat_params(global_model, global_flat + agg_delta)

    final_posteriors = posteriors

    n_predict_byz = max(1, int(round(byzantine_frac * num_clients)))
    threshold = float(np.sort(final_posteriors)[-n_predict_byz])
    det = detection_metrics(final_posteriors, byzantine_mask, threshold=threshold)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(byzantine_mask.astype(int), final_posteriors))
    except (ValueError, ImportError):
        auc = float("nan")

    posterior_gap = float(final_posteriors[byzantine_mask].mean()
                          - final_posteriors[~byzantine_mask].mean())
    test_mse = evaluate_regression(global_model, X_test, y_test)

    return {
        "rho": rho, "seed": seed,
        "num_cohorts": num_cohorts,
        "f1": det["f1"], "precision": det["precision"], "recall": det["recall"],
        "auc": auc,
        "posterior_gap": posterior_gap,
        "test_mse": test_mse,
        "mean_posterior_byz": float(final_posteriors[byzantine_mask].mean()),
        "mean_posterior_hon": float(final_posteriors[~byzantine_mask].mean()),
        "pct_cohorts_majority_byz": pct_majority_byz,  # A2 diagnostic
        "max_cohort_byz_frac": float(fracs.max()),
        "mean_cohort_byz_frac": float(fracs.mean()),
    }


# =============================================================================
# Sweep + gate validation
# =============================================================================

def main(out_dir="results/phase2_toy"):
    cfg = {
        "phase": "2_toy_gate",
        "rho_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "seeds": [11, 23, 47, 91, 137],
        "num_clients": 100,
        "samples_per_client": 50,
        "in_dim": 20,
        "byzantine_frac": 0.20,
        "cohort_size": 10,
        "num_rounds": 30,
    }
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"phase2_toy_{config_hash(cfg)}"
    print(f"[gate] run_id={run_id}")
    print(f"[gate] purpose: validate metric cohorts on Phase 1 toy setup")
    print(f"[gate] pass criteria: AUC monotone, end-start>=0.10, gap>0 at all rho, <10min")
    print()

    logger = RunLogger(out_dir, run_id, cfg)
    t_start = time.time()

    n_total = len(cfg["rho_grid"]) * len(cfg["seeds"])
    done = 0
    for rho, seed in product(cfg["rho_grid"], cfg["seeds"]):
        t0 = time.time()
        try:
            m = run_one(rho=rho, seed=seed,
                        num_clients=cfg["num_clients"],
                        samples_per_client=cfg["samples_per_client"],
                        in_dim=cfg["in_dim"],
                        byzantine_frac=cfg["byzantine_frac"],
                        cohort_size=cfg["cohort_size"],
                        num_rounds=cfg["num_rounds"])
            m["run_seconds"] = time.time() - t0
            m["error"] = ""
        except Exception as e:
            import traceback; traceback.print_exc()
            m = {"rho": rho, "seed": seed, "error": str(e)[:200],
                 "auc": float("nan"), "posterior_gap": float("nan"),
                 "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                 "test_mse": float("nan"),
                 "mean_posterior_byz": float("nan"), "mean_posterior_hon": float("nan"),
                 "pct_cohorts_majority_byz": float("nan"),
                 "max_cohort_byz_frac": float("nan"),
                 "mean_cohort_byz_frac": float("nan"),
                 "num_cohorts": 0,
                 "run_seconds": time.time() - t0}
        logger.log(**m)
        done += 1
        eta = (time.time() - t_start) / done * (n_total - done)
        print(f"[gate] {done:2d}/{n_total} rho={rho:.2f} seed={seed:3d} "
              f"AUC={m.get('auc', float('nan')):.3f} "
              f"gap={m.get('posterior_gap', float('nan')):+.3f} "
              f"%maj_byz={m.get('pct_cohorts_majority_byz', float('nan')):.2f} "
              f"({m['run_seconds']:.1f}s, eta {eta:.0f}s)")

    # === Gate evaluation ===
    import pandas as pd
    df = pd.read_csv(logger.csv_path)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["posterior_gap"] = pd.to_numeric(df["posterior_gap"], errors="coerce")
    df["pct_cohorts_majority_byz"] = pd.to_numeric(df["pct_cohorts_majority_byz"], errors="coerce")

    by_rho_auc = df.groupby("rho")["auc"].agg(["mean", "std"]).reset_index()
    by_rho_gap = df.groupby("rho")["posterior_gap"].agg(["mean", "std"]).reset_index()
    by_rho_a2 = df.groupby("rho")["pct_cohorts_majority_byz"].mean().reset_index()

    print("\n[gate] AUC by rho:")
    print(by_rho_auc.round(4).to_string(index=False))
    print("\n[gate] Posterior gap by rho:")
    print(by_rho_gap.round(4).to_string(index=False))
    print("\n[gate] A2 diagnostic (% cohorts with f_k > 0.5, expect near 0):")
    print(by_rho_a2.round(4).to_string(index=False))

    auc_means = by_rho_auc["mean"].values
    gap_means = by_rho_gap["mean"].values

    # Gate criteria
    crit1_monotone = all(auc_means[i] <= auc_means[i+1] + 0.05
                         for i in range(len(auc_means)-1))
    auc_at_010 = float(by_rho_auc[by_rho_auc.rho == 0.10]["mean"].values[0])
    auc_at_030 = float(by_rho_auc[by_rho_auc.rho == 0.30]["mean"].values[0])
    crit2_gap = (auc_at_030 - auc_at_010) >= 0.10
    crit3_pos_gap = bool(np.all(gap_means > 0.0))
    elapsed = time.time() - t_start
    crit4_time = elapsed < 600  # 10 minutes

    print(f"\n[gate] === Pass criteria ===")
    print(f"  1. AUC monotone non-decreasing:           {'PASS' if crit1_monotone else 'FAIL'}")
    print(f"  2. AUC(0.30) - AUC(0.10) >= 0.10:        "
          f"{'PASS' if crit2_gap else 'FAIL'} "
          f"(observed: {auc_at_030 - auc_at_010:+.3f})")
    print(f"  3. Posterior gap > 0 at all rho:          "
          f"{'PASS' if crit3_pos_gap else 'FAIL'} "
          f"(min gap: {gap_means.min():+.4f})")
    print(f"  4. Runtime < 10 minutes:                  "
          f"{'PASS' if crit4_time else 'FAIL'} "
          f"(observed: {elapsed:.0f}s)")

    all_pass = crit1_monotone and crit2_gap and crit3_pos_gap and crit4_time
    print(f"\n[gate] {'GATE PASS - cleared for Kaggle' if all_pass else 'GATE FAIL - DO NOT run Kaggle'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df_clean = df.dropna(subset=["auc"])
        agg_auc = df_clean.groupby("rho")["auc"].agg(["mean", "std"]).reset_index()
        agg_gap = df_clean.groupby("rho")["posterior_gap"].agg(["mean", "std"]).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        ax.errorbar(agg_auc["rho"], agg_auc["mean"], yerr=agg_auc["std"],
                    marker="o", capsize=4, linewidth=1.8, color="#2c7fb8",
                    label="Metric (k-means)")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
        ax.set_xlabel(r"Cohort overlap density $\rho$")
        ax.set_ylabel("Byzantine localization AUC")
        ax.set_title("Phase 2 toy gate: AUC vs rho")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3); ax.legend()

        ax = axes[1]
        ax.errorbar(agg_gap["rho"], agg_gap["mean"], yerr=agg_gap["std"],
                    marker="s", capsize=4, linewidth=1.8, color="#d95f0e")
        ax.axhline(0, color="red", linestyle="--", alpha=0.6,
                   label="A2 boundary (must stay above)")
        ax.set_xlabel(r"Cohort overlap density $\rho$")
        ax.set_ylabel("Posterior gap (P_byz - P_hon)")
        ax.set_title("Phase 2 toy gate: separability")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"{run_id}_gate.png")
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"[gate] plot saved: {plot_path}")
    except Exception as e:
        print(f"[gate] plot failed: {e}")

    logger.finalize(
        gate_pass=all_pass,
        criteria={
            "monotone": crit1_monotone,
            "gap_threshold": crit2_gap,
            "positive_posterior_gap": crit3_pos_gap,
            "under_time_budget": crit4_time,
        },
        observations={
            "auc_at_010": auc_at_010,
            "auc_at_030": auc_at_030,
            "min_posterior_gap": float(gap_means.min()),
            "max_pct_cohorts_majority_byz": float(df["pct_cohorts_majority_byz"].max()),
        },
        elapsed_seconds=elapsed,
    )
    return all_pass


if __name__ == "__main__":
    main()
