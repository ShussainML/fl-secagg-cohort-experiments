"""
phase0_smoke_test.py
====================
End-to-end smoke test of the full FL pipeline.

Goal: prove the simulator works end-to-end before any real phase runs.
We run vanilla FedAvg on CIFAR-10 with 10 clients, 5 rounds, 1 local epoch.
Expected: training loss decreases monotonically (or near-monotonically),
test accuracy goes up from random (10%) to >15%.

If this notebook fails, do NOT trust phase 1-5 results.

Designed to run on Kaggle: clone repo, run, push results.
On a T4 this should complete in <5 minutes.
"""

import os
import sys
import time

# When running on Kaggle, repo is cloned to /kaggle/working/fl-secagg-cohort-experiments
# When running locally, fl_core.py is in same dir
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from fl_core import (
    set_all_seeds, ResNet8GN, count_params, get_flat_params, set_flat_params,
    dirichlet_partition, IndexedSubset, JLSketch,
    metric_cohorts, topological_cohorts, cohort_overlap_stats, build_factor_graph,
    Client, SecAggServer, fedavg_secagg, evaluate, RunLogger,
    config_hash, make_attack,
)


def main(out_dir: str = "results/phase0"):
    SEED = 42
    set_all_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    cfg = {
        "phase": "0_smoke",
        "seed": SEED,
        "num_clients": 10,
        "num_rounds": 5,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.05,
        "alpha_dirichlet": 0.5,  # mild non-IID for smoke
        "byzantine_frac": 0.0,
        "attack": "none",
        "model": "ResNet8GN",
        "dataset": "CIFAR10",
        "num_cohorts": 5,
        "cohort_size": 4,
        "sketch_dim": 32,
    }
    run_id = f"smoke_{config_hash(cfg)}"
    print(f"[smoke] run_id={run_id}")

    # --- Data ---
    use_synthetic = os.environ.get("FL_USE_SYNTHETIC", "0") == "1"
    if use_synthetic:
        print("[smoke] using SYNTHETIC CIFAR-shaped data (offline mode)")
        from torch.utils.data import TensorDataset
        rng_d = np.random.RandomState(SEED)
        n_train, n_test = 2000, 500
        # Synthetic class-conditional Gaussians
        train_x = torch.randn(n_train, 3, 32, 32)
        train_y = torch.from_numpy(rng_d.randint(0, 10, n_train)).long()
        # Bias the input by class so model can learn something
        for c in range(10):
            mask = train_y == c
            train_x[mask] += 0.5 * (c - 4.5)
        test_x = torch.randn(n_test, 3, 32, 32)
        test_y = torch.from_numpy(rng_d.randint(0, 10, n_test)).long()
        for c in range(10):
            mask = test_y == c
            test_x[mask] += 0.5 * (c - 4.5)
        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)
        train_set.targets = train_y.numpy()  # for partition
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        data_root = os.environ.get("CIFAR10_ROOT", "./data")
        os.makedirs(data_root, exist_ok=True)
        train_set = torchvision.datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            data_root, train=False, download=True, transform=transform
        )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    # --- Partition ---
    targets = np.array(train_set.targets) if hasattr(train_set, "targets") else \
              np.array([int(y) for _, y in train_set])
    parts = dirichlet_partition(
        targets, cfg["num_clients"], cfg["alpha_dirichlet"], seed=SEED
    )
    print(f"[smoke] partition sizes: {[len(p) for p in parts]}")

    # --- Model ---
    global_model = ResNet8GN(num_classes=10).to(device)
    n_params = count_params(global_model)
    print(f"[smoke] model params: {n_params:,}")

    # --- Sketch ---
    sketch = JLSketch(in_dim=n_params, out_dim=cfg["sketch_dim"], seed=SEED)

    # --- Clients ---
    clients = []
    for i in range(cfg["num_clients"]):
        ds = IndexedSubset(train_set, parts[i])
        clients.append(Client(
            client_id=i,
            dataset=ds,
            is_byzantine=False,
            attack=make_attack("none"),
            device=device,
            local_epochs=cfg["local_epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
        ))

    # --- Cohorts: random metadata for smoke test ---
    metadata = np.random.RandomState(SEED).randn(cfg["num_clients"], 2)
    cohorts = metric_cohorts(metadata, cfg["num_cohorts"], cfg["cohort_size"], seed=SEED)
    stats = cohort_overlap_stats(cohorts, cfg["num_clients"])
    print(f"[smoke] cohort stats: {stats}")

    # --- Logger ---
    logger = RunLogger(out_dir, run_id, cfg)

    # --- Training loop ---
    total_samples = sum(len(c.dataset) for c in clients)
    local_model = ResNet8GN(num_classes=10).to(device)

    for rnd in range(cfg["num_rounds"]):
        t0 = time.time()
        global_flat = get_flat_params(global_model)

        # Local training
        results = []
        for client in clients:
            r = client.train(local_model, global_flat, sketch)
            results.append(r)

        # SecAgg-constrained server
        server = SecAggServer(results)
        delta = fedavg_secagg(server, cohorts, total_samples)
        new_flat = global_flat + delta
        set_flat_params(global_model, new_flat)

        # Eval
        test_loss, test_acc = evaluate(global_model, test_loader, device)
        train_loss = float(np.mean([r.train_loss for r in results]))
        delta_norm = float(delta.norm().item())
        elapsed = time.time() - t0

        logger.log(
            round=rnd,
            train_loss=train_loss,
            test_loss=test_loss,
            test_acc=test_acc,
            delta_norm=delta_norm,
            sec_per_round=elapsed,
        )
        print(f"[smoke] round {rnd}: train_loss={train_loss:.3f} "
              f"test_acc={test_acc:.3f} |delta|={delta_norm:.3f} "
              f"({elapsed:.1f}s)")

    # --- Validation ---
    import csv
    with open(logger.csv_path) as f:
        rows = list(csv.DictReader(f))
    final_acc = float(rows[-1]["test_acc"])
    initial_loss = float(rows[0]["train_loss"])
    final_loss = float(rows[-1]["train_loss"])

    checks = {
        "loss_decreased": final_loss < initial_loss,
        "acc_above_random": final_acc > 0.15,
        "no_nan": all(not np.isnan(float(r["train_loss"])) for r in rows),
    }
    passed = all(checks.values())
    print(f"[smoke] validation: {checks}")
    print(f"[smoke] {'PASS' if passed else 'FAIL'}")

    logger.finalize(
        final_test_acc=final_acc,
        initial_train_loss=initial_loss,
        final_train_loss=final_loss,
        validation_checks=checks,
        validation_passed=passed,
        n_params=n_params,
        cohort_stats=stats,
    )

    if not passed:
        raise RuntimeError(f"Smoke test failed: {checks}")
    return passed


if __name__ == "__main__":
    main()
