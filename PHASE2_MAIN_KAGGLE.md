# Phase 2 Main — Kaggle Notebook

**Use any account with available GPU quota. Enable T4 x1 (or x2) GPU.**
Estimated runtime: **~5-6 hours** for 12 runs.

## Pre-conditions

Before running this:
1. Phase 2 toy gate must have passed on Kaggle (you ran it; it passed).
2. Push `notebooks/phase2_main.py` to your repo.

## Cell 1: Setup (paste verbatim)

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

import fl_core, torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name()}")
fl_core.run_all_tests()
print("Setup complete.")
```

**Stop here if `torch.cuda.is_available()` is False.** GPU is required; CPU
would take 30+ hours. Switch to a GPU runtime in Kaggle settings.

## Cell 2: Run Phase 2 main (long-running; do NOT close browser tab)

```python
import os
os.makedirs("/kaggle/working/results/phase2", exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"

from phase2_main import main as phase2_main
phase2_main(out_dir="/kaggle/working/results/phase2")
```

**What you'll see:**
- "loading CIFAR-10..." then training set size
- 12 progress lines, one per run, with `acc` and `AUC` and `eta`
- After all 12: H1 summary table, H2 summary table, verdicts
- Plot saved at end

**If a single run errors out** (e.g., out of memory), the exception is caught,
the run is logged with NaN metrics, and the next run starts. You won't lose
partial progress — every completed run is in the CSV before the next starts.

## Cell 3: Push results to repo

```python
import subprocess, os, datetime
from kaggle_secrets import UserSecretsClient

PHASE_NAME = "phase2"
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

## What to share back when done

Paste:
1. The 12 progress lines from Cell 2.
2. The H1 table and H2 table.
3. The two VERDICT lines.
4. The plot file (drag-and-drop into chat).

I'll cross-check against expectations from the math walkthrough (predictions
for AUC ranges and accuracy delta) and we'll discuss whether to proceed
to Phase 3 or revisit anything.

## Important: don't close the browser tab during Cell 2

Kaggle keeps notebook kernels alive on the server even if your browser tab
closes — but only for so long. To be safe: leave the tab open, or use
"Save & Run All (Commit)" mode which queues the run server-side independent
of your browser.
