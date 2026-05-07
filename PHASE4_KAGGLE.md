# Phase 4 — Kaggle Notebook

**Use Account B** (or any account different from Phase 3 if you're parallelizing).
Enable T4 x1 GPU.
Estimated runtime: **~10 hours** for 18 runs. Fits in one Kaggle session;
if it gets cut off, the script resumes from the CSV.

## Pre-conditions

1. Fixed `fl_core.py` and `phase4_baselines.py` are in the repo.
2. CIFAR-10 dataset attached.

## Cell 1: Setup (paste verbatim — same as Phase 3)

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

## Cell 2: Stage CIFAR-10

```python
import os, shutil
KAGGLE_CIFAR_ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "cifar-10-batches-py" in dirs:
        KAGGLE_CIFAR_ROOT = root; break

assert KAGGLE_CIFAR_ROOT is not None, "Attach 'CIFAR-10 Python' dataset"
src = os.path.join(KAGGLE_CIFAR_ROOT, "cifar-10-batches-py")
dst = "/kaggle/working/data/cifar-10-batches-py"
os.makedirs("/kaggle/working/data", exist_ok=True)
if not os.path.exists(dst):
    shutil.copytree(src, dst)
    print(f"Copied to {dst}")
```

## Cell 3: Run Phase 4

```python
import os
os.makedirs("/kaggle/working/results/phase4", exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"

from phase4_baselines import main as phase4_main
phase4_main(out_dir="/kaggle/working/results/phase4")
```

You'll see 18 progress lines (6 methods × 3 seeds). At the end, a summary
table shows methods sorted by SecAgg-compatibility (compatible first), with
H4.1 and H4.2 verdicts.

If session ends before completion, re-run the same cell in a new session
— it picks up where it left off.

## Cell 4: Push results

```python
import subprocess, os, datetime
from kaggle_secrets import UserSecretsClient

PHASE_NAME = "phase4"
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

## Phase 3 + Phase 4 in parallel

Phase 4 (Account B) and Phase 3 (Account A) write to **different result
subdirectories** (`results/phase3/` and `results/phase4/`). The git commits
won't conflict on file content, and `git pull --rebase` before each push
handles same-time pushes from both accounts.

Both can run simultaneously without coordination.
