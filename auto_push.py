"""
auto_push.py
============
Periodic git push of partial results during long-running training.

Why this exists:
  Kaggle kills kernels at the 12-hour wall without warning. To survive,
  we push partial CSVs to GitHub every N completed cells. If the kernel
  dies, the most recent push is preserved in the repo. Next session
  restores from repo, resumes from the CSV.

How to use (inside a training loop):

    from auto_push import AutoPusher
    pusher = AutoPusher(
        repo_dir="/kaggle/working/fl-secagg-cohort-experiments",
        results_local_dir="/kaggle/working/results/phase3",
        results_repo_dir="/kaggle/working/.../results/phase3",
        phase_name="phase3",
        push_every=5,
    )
    for i, cell in enumerate(cells):
        ... run cell ...
        logger.log(...)
        pusher.maybe_push()        # pushes if N completed since last push

    pusher.final_push()             # push any remaining (always at end)

Failure mode: if a push fails (network, auth, conflict), AutoPusher logs
the error and returns. It does NOT raise. Training continues. The push
will retry next time. The user's training run is more important than
any individual git operation.

Setup expectation:
  Caller has already configured git in WORK_DIR with a credential helper
  (the askpass.sh setup) so that `git push` works without interactive auth.
  See SETUP.md / Cell 1 in the Kaggle notebooks.
"""

import os
import shutil
import subprocess
import datetime
import time


class AutoPusher:
    def __init__(self, repo_dir, results_local_dir, results_repo_dir,
                 phase_name, push_every=5):
        self.repo_dir = repo_dir
        self.results_local_dir = results_local_dir
        self.results_repo_dir = results_repo_dir
        self.phase_name = phase_name
        self.push_every = push_every
        self.completed_since_last_push = 0
        self.total_pushes = 0
        self.last_push_status = "init"

        os.makedirs(self.results_local_dir, exist_ok=True)
        os.makedirs(self.results_repo_dir, exist_ok=True)

    def cell_completed(self):
        """Caller signals one cell has been completed."""
        self.completed_since_last_push += 1

    def maybe_push(self):
        """Push if we've accumulated push_every completed cells."""
        if self.completed_since_last_push >= self.push_every:
            self._do_push(label="auto")
            self.completed_since_last_push = 0

    def final_push(self):
        """Always-runs final push at end of training."""
        self._do_push(label="final")

    def _do_push(self, label="auto"):
        try:
            t0 = time.time()
            # Copy local CSV/PNG into repo dir
            if os.path.isdir(self.results_local_dir) and os.listdir(self.results_local_dir):
                subprocess.run(
                    ["cp", "-rT", self.results_local_dir, self.results_repo_dir],
                    check=True
                )

            # Stage
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "add",
                 f"results/{self.phase_name}/"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                self.last_push_status = f"add-failed: {r.stderr[:200]}"
                print(f"[auto_push] git add failed: {r.stderr[:200]}")
                return

            # Anything staged?
            staged = subprocess.run(
                ["git", "-C", self.repo_dir, "diff", "--cached", "--name-only"],
                capture_output=True, text=True
            ).stdout.strip()
            if not staged:
                self.last_push_status = "nothing-staged"
                return

            # Commit
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "commit",
                 "-m", f"{self.phase_name} {label} push {stamp}"],
                capture_output=True, text=True
            )
            # commit may fail with "nothing to commit"; treat as ok
            if r.returncode != 0 and "nothing to commit" not in r.stdout.lower() + r.stderr.lower():
                self.last_push_status = f"commit-failed: {r.stderr[:200]}"
                print(f"[auto_push] git commit failed: {r.stderr[:200]}")
                return

            # Pull rebase (concurrency safety)
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "pull", "--rebase", "origin", "main"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                self.last_push_status = f"pull-failed: {r.stderr[:200]}"
                print(f"[auto_push] git pull failed: {r.stderr[:200]}")
                return

            # Push
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "push", "origin", "main"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                # Redact PAT from output if present
                err = r.stderr
                pat = os.environ.get("GIT_AUTH_PAT_FOR_REDACT", "")
                if pat:
                    err = err.replace(pat, "[REDACTED]")
                self.last_push_status = f"push-failed: {err[:200]}"
                print(f"[auto_push] git push failed: {err[:200]}")
                return

            self.total_pushes += 1
            self.last_push_status = "ok"
            elapsed = time.time() - t0
            print(f"[auto_push] {label} push #{self.total_pushes} ok ({elapsed:.1f}s)")
        except Exception as e:
            # Never crash training because of a push failure
            self.last_push_status = f"exception: {str(e)[:200]}"
            print(f"[auto_push] exception during push: {e}")

    def restore_from_repo(self):
        """Pull existing partial CSV from repo into local results dir.
        Call this BEFORE training starts."""
        if not os.path.isdir(self.results_repo_dir):
            print("[auto_push] no prior results in repo")
            return 0
        files = [f for f in os.listdir(self.results_repo_dir)
                 if os.path.isfile(os.path.join(self.results_repo_dir, f))]
        if not files:
            print("[auto_push] repo results dir empty")
            return 0
        for f in files:
            src = os.path.join(self.results_repo_dir, f)
            dst = os.path.join(self.results_local_dir, f)
            shutil.copy(src, dst)
        print(f"[auto_push] restored {len(files)} files: {files}")
        return len(files)
