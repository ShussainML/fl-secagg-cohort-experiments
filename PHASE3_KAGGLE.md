# Phase 3 — Kaggle Notebook (with auto-push, survives 12-hour wall)

**Use Account A.** Enable T4 x1 GPU.

Phase 3 has 48 cells × ~30 min each = ~24 hours of compute. Kaggle kills
sessions at 12 hours, so you'll need 2-3 sessions. **The script auto-pushes
to GitHub every 5 cells.** If a kernel dies, just open a new notebook and
re-run — it picks up from the last push.

## Pre-conditions

1. Repo contains the latest `fl_core.py`, `phase3_robustness.py`, and `auto_push.py`.
2. CIFAR-10 Python dataset attached to the notebook.

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
os.environ["GIT_AUTH_PAT_FOR_REDACT"] = PAT  # auto_push uses this for log redaction

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

## Cell 3: Run Phase 3 with auto-push

```python
import os
PHASE_NAME = "phase3"
RESULTS_LOCAL = f"/kaggle/working/results/{PHASE_NAME}"
RESULTS_REPO  = f"{WORK_DIR}/results/{PHASE_NAME}"
os.makedirs(RESULTS_LOCAL, exist_ok=True)
os.makedirs(RESULTS_REPO, exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"

# Set up auto-pusher
from auto_push import AutoPusher
pusher = AutoPusher(
    repo_dir=WORK_DIR,
    results_local_dir=RESULTS_LOCAL,
    results_repo_dir=RESULTS_REPO,
    phase_name=PHASE_NAME,
    push_every=5,           # push after every 5 completed cells (~2.5 hours)
)

# Restore any partial CSV from previous session
restored = pusher.restore_from_repo()
print(f"Restored {restored} files from previous session.")

# Run training - auto-pusher handles periodic pushing automatically.
# If kernel dies mid-training, the most recent push (within last ~5 cells) is in repo.
from phase3_robustness import main as phase3_main
phase3_main(out_dir=RESULTS_LOCAL, auto_pusher=pusher)

print("\n=== DONE ===")
print(f"Total auto-pushes during run: {pusher.total_pushes}")
print(f"Last push status: {pusher.last_push_status}")
```

That's it. This is the only training cell you need.

## What happens across sessions

**Session 1** (start fresh):
- Cells 1, 2, 3 run. Auto-push happens every 5 cells. Around the 12-hour mark, kernel dies. Most recent push contains results 1-22 (approximately).

**Session 2** (re-open Kaggle):
- Open a NEW notebook (same code, same dataset attached, GPU enabled).
- Run Cells 1, 2, 3 again. AutoPusher restores partial CSV from repo. phase3_main reads CSV, skips done cells, runs the rest.

**Session 3** if needed: same pattern. Eventually all 48 cells complete.

## Expected output

```
[phase3] 1/48 m=ours   atk=sign_flip   f=0.10 seed= 11 acc=0.245 AUC=0.612 (29.4min, eta 1380min, done=1 skip=0 fail=0)
...
[auto_push] auto push #1 ok (3.4s)        # after 5 cells complete
...
[KERNEL KILLED at 12-hour wall]

# Session 2:
[auto_push] restored 4 files from previous session
[phase3] resuming: 22 cells already complete; 26 remaining
[phase3] 1/48 ... [SKIPPED — already done]
...
[phase3] 23/48 ... [actually runs]
```

## Important: what to do at start of each new session

Just run all 3 cells in order. That's it. No manual restore — Cell 3 does it automatically.

## Compute budget warning

Phase 3 is ~24 hours. Phase 4 is ~9 hours. Total ~33 hours. Kaggle weekly quota
is 30 hours, so you may run out before both finish on one account. Options:

- Run Phase 4 first (cheaper), then Phase 3 with whatever quota is left.
- Run Phase 3 in week 1 (quota resets weekly), Phase 4 in week 2.
