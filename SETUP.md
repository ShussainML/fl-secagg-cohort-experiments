# Kaggle Setup Template for FL-SecAgg Experiments

This is the **template** you copy into the first cell of every Kaggle notebook (Phase 1, 2, 3, 4, 5). It handles:

1. Cloning your private GitHub repo using a PAT stored in Kaggle Secrets.
2. Setting up Python path to import `fl_core`.
3. Pulling latest code at start (in case you push fixes between phase runs).
4. Pushing results back at end.

## One-time setup steps (do these BEFORE running anything)

### Step 1: Create a private GitHub repo

1. Go to https://github.com/new
2. Name it (e.g.) `fl-secagg-cohort-experiments`.
3. **Private**.
4. Initialize with a README.
5. Push the contents of `/home/claude/fl-secagg-cohort/` to it (`fl_core.py`, `notebooks/`, `results/`).

```bash
cd ~/your-local-clone
git init
git remote add origin https://github.com/YOURUSER/fl-secagg-cohort-experiments.git
git add fl_core.py notebooks/ results/.gitkeep
git commit -m "Initial commit: fl_core + smoke test"
git branch -M main
git push -u origin main
```

### Step 2: Create a fine-grained Personal Access Token

1. https://github.com/settings/personal-access-tokens/new
2. Resource owner: yourself.
3. Repository access: only the new repo.
4. Permissions: **Contents = Read and write**.
5. Set expiry to 90 days (longer than your project).
6. Copy the token (starts with `github_pat_...`).

### Step 3: Add token to all 4 Kaggle accounts

On each Kaggle account separately:
1. Open any notebook.
2. Add-ons menu → Secrets → Add a secret.
3. Label: `GITHUB_PAT`. Value: paste the token.
4. Attach to the notebook.

You can use the same PAT across all 4 accounts (it's tied to your GitHub user, which has the repo). If you want stricter isolation, create 4 separate PATs.

---

## Cell 1 of every Kaggle phase notebook: paste this verbatim

```python
# === Kaggle setup: pull repo + import fl_core ===
import os, subprocess, sys

REPO_OWNER = "YOURUSER"   # <-- change this
REPO_NAME  = "fl-secagg-cohort-experiments"   # <-- change this
WORK_DIR   = f"/kaggle/working/{REPO_NAME}"

# Get PAT from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
PAT = UserSecretsClient().get_secret("GITHUB_PAT")

# Clone or pull
if not os.path.exists(WORK_DIR):
    url = f"https://{PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
    subprocess.run(["git", "clone", url, WORK_DIR], check=True)
else:
    subprocess.run(["git", "-C", WORK_DIR, "pull"], check=True)

sys.path.insert(0, WORK_DIR)

# Configure git for later push
subprocess.run(["git", "-C", WORK_DIR, "config", "user.email", "kaggle@local"], check=True)
subprocess.run(["git", "-C", WORK_DIR, "config", "user.name", "kaggle-runner"], check=True)

# Verify import
import fl_core
print("fl_core imported OK. Self-tests:")
fl_core.run_all_tests()
```

## Final cell of every Kaggle phase notebook: paste this verbatim

```python
# === Push results back to repo ===
import subprocess, os, datetime

PHASE_NAME = "phaseX"  # <-- change per notebook (phase1, phase2, ...)
RESULTS_DIR = f"{WORK_DIR}/results/{PHASE_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Move any results we wrote to /kaggle/working/results/<phase>/ into repo
SRC = f"/kaggle/working/results/{PHASE_NAME}"
if os.path.isdir(SRC):
    subprocess.run(["cp", "-rT", SRC, RESULTS_DIR], check=True)

stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
subprocess.run(["git", "-C", WORK_DIR, "add", f"results/{PHASE_NAME}/"], check=True)

# Only commit if there are changes
status = subprocess.run(
    ["git", "-C", WORK_DIR, "status", "--porcelain"],
    capture_output=True, text=True
).stdout
if status.strip():
    subprocess.run(
        ["git", "-C", WORK_DIR, "commit", "-m", f"{PHASE_NAME} results {stamp}"],
        check=True,
    )
    subprocess.run(["git", "-C", WORK_DIR, "push"], check=True)
    print(f"Pushed {PHASE_NAME} results.")
else:
    print("No changes to push.")
```

---

## Concurrency note

If two notebooks push at the same time, git will reject the second push.
Solution: each phase notebook writes to its **own subdirectory** (`results/phase1/`,
`results/phase2/`, ...) so there's never a real merge conflict. The git-level
push race is solved by the final cell doing `git pull --rebase` before push:

```python
# Robust push (handles concurrent updates from other accounts)
subprocess.run(["git", "-C", WORK_DIR, "pull", "--rebase"], check=True)
subprocess.run(["git", "-C", WORK_DIR, "push"], check=True)
```

This is already handled in the final cell above if you replace the last two
push lines with:

```python
    subprocess.run(["git", "-C", WORK_DIR, "pull", "--rebase"], check=True)
    subprocess.run(["git", "-C", WORK_DIR, "push"], check=True)
```

---

## Smoke test: run this on ONE account first

Before launching all 4 phases in parallel, on Account A:
1. Open a new Kaggle notebook.
2. Paste Cell 1 (setup).
3. Paste this body:

```python
import sys
sys.path.insert(0, f"{WORK_DIR}/notebooks")
from phase0_smoke_test import main
os.makedirs(f"/kaggle/working/results/phase0", exist_ok=True)
os.environ["CIFAR10_ROOT"] = "/kaggle/working/data"
main(out_dir="/kaggle/working/results/phase0")
```

4. Paste the final cell (push), with `PHASE_NAME = "phase0"`.
5. Run all. Should complete in ~5 minutes on T4.
6. Verify `results/phase0/smoke_*.csv` shows up in the GitHub repo.

If smoke passes on the repo, you're cleared to launch Phases 1–5 in parallel.
