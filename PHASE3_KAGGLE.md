# Phase 3 — Kaggle Notebook

**Use Account A (or any with GPU quota).** Enable T4 x1 GPU.
Estimated runtime: **~26 hours total**, but the script is *resumable*, so
running it across two 12-hour Kaggle sessions on the same account is fine.

If you have two accounts, see "Parallelization" below to split into halves.

## Pre-conditions

1. The fixed `fl_core.py` and `phase3_robustness.py` are in the repo.
2. The CIFAR-10 dataset is attached (the Python format, as you did for Phase 2).

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
fl_core.run_all_tests()
print("Setup complete.")
```

## Cell 2: Stage CIFAR-10 (same as Phase 2)

```python
import os, shutil
KAGGLE_CIFAR_ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "cifar-10-batches-py" in dirs:
        KAGGLE_CIFAR_ROOT = root; break

assert KAGGLE_CIFAR_ROOT is not None, "Attach 'CIFAR-10 Python' dataset to the notebook"
src = os.path.join(KAGGLE_CIFAR_ROOT, "cifar-10-batches-py")
dst = "/kaggle/working/data/cifar-10-batches-py"
os.makedirs("/kaggle/working/data", exist_ok=True)
if not os.path.exists(dst):
    shutil.copytree(src, dst)
    print(f"Copied to {dst}")
```

## Cell 3: Run Phase 3 (long-running, but resumable)

```python
import os
os.makedirs("/kaggle/working/results/phase3", exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"

from phase3_robustness import main as phase3_main
phase3_main(out_dir="/kaggle/working/results/phase3")
```

If your Kaggle session times out before completion, re-run Cells 1+2+3 in
a new session. The script reads existing CSV and skips completed cells.

## Cell 4: Push results

```python
import subprocess, os, datetime
from kaggle_secrets import UserSecretsClient

PHASE_NAME = "phase3"
RESULTS_DIR = f"{WORK_DIR}/results/{PHASE_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

SRC = f"/kaggle/working/results/{PHASE_NAME}"
if os.path.isdir(SRC) and os.listdir(SRC):
    subprocess.run(["cp", "-rT", SRC, RESULTS_DIR], check=True)
    print(f"Files: {os.listdir(RESULTS_DIR)}")

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

## Parallelization (optional — if you have 2 accounts)

Phase 3 has 48 cells. To run in two halves on two accounts:

**Account A:** edit `phase3_robustness.py`'s `cfg["byzantine_fracs"]` to `[0.10, 0.20]` (32 cells, ~17 hrs, fits in 2 sessions).

**Account B:** edit to `[0.30]` (16 cells, ~9 hrs, single session).

After both finish, push to repo. Phase 6 will combine both CSVs.

You can also skip parallelization and just run sequentially over 3 sessions
on one account — the resume-from-CSV is the safety net. Most people will
prefer this; it's simpler.
