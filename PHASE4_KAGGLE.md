# Phase 4 — Kaggle Notebook (with auto-push)

**Use Account A (or B if parallelizing).** Enable T4 x1 GPU.
~9 hours of compute, fits in one session. But auto-push is on regardless,
so if it crashes you can resume without losing more than ~5 cells of work.

## Pre-conditions

1. Repo contains the latest `fl_core.py`, `phase4_baselines.py`, and `auto_push.py`.
2. CIFAR-10 Python dataset attached.

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
os.environ["GIT_AUTH_PAT_FOR_REDACT"] = PAT

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
else:
    print(f"Already present at {dst}")
```

## Cell 3: Run Phase 4 with auto-push

```python
import os
PHASE_NAME = "phase4"
RESULTS_LOCAL = f"/kaggle/working/results/{PHASE_NAME}"
RESULTS_REPO  = f"{WORK_DIR}/results/{PHASE_NAME}"
os.makedirs(RESULTS_LOCAL, exist_ok=True)
os.makedirs(RESULTS_REPO, exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"

from auto_push import AutoPusher
pusher = AutoPusher(
    repo_dir=WORK_DIR,
    results_local_dir=RESULTS_LOCAL,
    results_repo_dir=RESULTS_REPO,
    phase_name=PHASE_NAME,
    push_every=3,            # push every 3 cells (~1.5 hours)
)

restored = pusher.restore_from_repo()
print(f"Restored {restored} files from previous session.")

from phase4_baselines import main as phase4_main
phase4_main(out_dir=RESULTS_LOCAL, auto_pusher=pusher)

print("\n=== DONE ===")
print(f"Total auto-pushes during run: {pusher.total_pushes}")
print(f"Last push status: {pusher.last_push_status}")
```

## Cross-session behavior

Same as Phase 3: if killed mid-run, just rerun the same 3 cells in a new
session. AutoPusher restores from repo, training resumes from where it
stopped. No manual steps.
