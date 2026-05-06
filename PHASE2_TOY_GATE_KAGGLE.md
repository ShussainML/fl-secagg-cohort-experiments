# Phase 2 Toy Gate — Kaggle Notebook

This is the gate that must pass before any expensive Phase 2 run.

**Use any Kaggle account.** CPU is fine — no GPU needed. ~4 minutes total.

## Pre-step

Push `notebooks/phase2_toy_gate.py` to your repo first.

## Cell 1: Setup (paste verbatim, identical to other phases)

```python
import os, subprocess, sys

REPO_OWNER = "ShussainML"
REPO_NAME  = "fl-secagg-cohort-experiments"
WORK_DIR   = f"/kaggle/working/{REPO_NAME}"

from kaggle_secrets import UserSecretsClient
PAT = UserSecretsClient().get_secret("GITHUB_PAT")

askpass_path = "/kaggle/working/.git_askpass.sh"
with open(askpass_path, "w") as f:
    f.write(f'#!/bin/bash\necho "{PAT}"\n')
os.chmod(askpass_path, 0o700)
os.environ["GIT_ASKPASS"] = askpass_path
os.environ["GIT_TERMINAL_PROMPT"] = "0"

clean_url = f"https://{REPO_OWNER}@github.com/{REPO_OWNER}/{REPO_NAME}.git"

if not os.path.exists(WORK_DIR):
    subprocess.run(["git", "clone", clean_url, WORK_DIR], check=True)
else:
    subprocess.run(["git", "-C", WORK_DIR, "remote", "set-url", "origin", clean_url], check=True)
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--rebase"], check=True)

sys.path.insert(0, WORK_DIR)
sys.path.insert(0, f"{WORK_DIR}/notebooks")

subprocess.run(["git", "-C", WORK_DIR, "config", "user.email", "kaggle@local"], check=True)
subprocess.run(["git", "-C", WORK_DIR, "config", "user.name", "kaggle-runner"], check=True)

import fl_core
fl_core.run_all_tests()
print("Setup complete.")
```

## Cell 2: Run the gate

```python
import os
os.makedirs("/kaggle/working/results/phase2_toy", exist_ok=True)

from phase2_toy_gate import main as gate_main
passed = gate_main(out_dir="/kaggle/working/results/phase2_toy")

if not passed:
    raise SystemExit("Gate FAILED. Do NOT proceed to expensive runs. Diagnose first.")
print("Gate cleared.")
```

Expected output:
- 40 lines of progress (8 rho × 5 seeds)
- Summary tables (AUC by rho, posterior gap by rho, A2 diagnostic)
- 4 pass criteria, all PASS
- "GATE PASS - cleared for Kaggle"
- Plot file saved

Expected runtime: **~4 minutes on CPU** (5–6 sec per run × 40 runs).

## Cell 3: Push results

```python
import subprocess, os, datetime
from kaggle_secrets import UserSecretsClient

PHASE_NAME = "phase2_toy"
RESULTS_DIR = f"{WORK_DIR}/results/{PHASE_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

SRC = f"/kaggle/working/results/{PHASE_NAME}"
if os.path.isdir(SRC) and os.listdir(SRC):
    subprocess.run(["cp", "-rT", SRC, RESULTS_DIR], check=True)
    print(f"Files: {os.listdir(RESULTS_DIR)}")
else:
    raise RuntimeError(f"No results at {SRC}.")

gi = f"{WORK_DIR}/.gitignore"
needed = ["__pycache__/", "*.pyc", ".ipynb_checkpoints/", "data/"]
existing = open(gi).read() if os.path.exists(gi) else ""
to_add = [x for x in needed if x not in existing]
if to_add:
    with open(gi, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write("\n".join(to_add) + "\n")
    subprocess.run(["git", "-C", WORK_DIR, "add", ".gitignore"], check=True)

subprocess.run(["git", "-C", WORK_DIR, "add", f"results/{PHASE_NAME}/"], check=True)

staged = subprocess.run(
    ["git", "-C", WORK_DIR, "diff", "--cached", "--name-only"],
    capture_output=True, text=True
).stdout.strip()

if not staged:
    print("Nothing to push.")
else:
    print("Staged:")
    print(staged)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    subprocess.run(
        ["git", "-C", WORK_DIR, "commit", "-m", f"{PHASE_NAME} results {stamp}"],
        check=False,
    )
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--rebase", "origin", "main"], check=True)
    push = subprocess.run(["git", "-C", WORK_DIR, "push", "origin", "main"],
                         capture_output=True, text=True)
    if push.returncode != 0:
        PAT_REDACT = UserSecretsClient().get_secret("GITHUB_PAT")
        err = push.stderr.replace(PAT_REDACT, "[REDACTED]")
        raise RuntimeError(f"Push failed: {err[:300]}")
    print("Pushed.")
```

## What to do with the result

Paste the printed output (the AUC/gap tables and the 4-criteria block) into chat.
I will compare against my local run to confirm reproducibility before writing
the main Phase 2 Kaggle code.
