# Phase 2 Kaggle Notebook — Cell by Cell

Use **Account B**. **Enable GPU** (T4 x2 or T4 x1) for ~2-3 hour runtime.

## Cell 1: Setup (paste verbatim)

```python
# === Setup: pull repo + import fl_core ===
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

## Cell 2: Run Phase 2

```python
import os
os.makedirs("/kaggle/working/results/phase2", exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"

from phase2_topological_vs_metric import main as phase2_main
phase2_main(out_dir="/kaggle/working/results/phase2")
```

Expected runtime: **~2-3 hours** on T4 GPU.
- 18 runs (2 methods × 3 rho × 3 seeds)
- ~7-10 minutes per run

You'll see progress lines per run, then a final validation summary and plot.

**If a run fails**, the script catches the exception and continues to the
next run, logging the error in the CSV. Always re-check the CSV at the end
to see which (method, rho, seed) cells succeeded.

## Cell 3: Push results

```python
# === Push results to repo ===
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
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--rebase", "origin", "main"],
                   check=True)
    push = subprocess.run(["git", "-C", WORK_DIR, "push", "origin", "main"],
                         capture_output=True, text=True)
    if push.returncode != 0:
        PAT_REDACT = UserSecretsClient().get_secret("GITHUB_PAT")
        err = push.stderr.replace(PAT_REDACT, "[REDACTED]")
        raise RuntimeError(f"Push failed: {err[:300]}")
    print("Pushed.")
```

## Concurrency tip

Phase 1 (Account A) and Phase 2 (Account B) write to **different** subdirectories
(`results/phase1/` and `results/phase2/`), so they never conflict on file content.
The `git pull --rebase` before push handles the case where both finish at the
same time.
