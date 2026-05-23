import subprocess
import os
from pathlib import Path

# Run from the repository root (parent of frontend)
repo_root = Path(__file__).resolve().parent.parent

def run_git(args):
    print(f"Running: git {' '.join(args)}")
    res = subprocess.run(["git"] + args, cwd=repo_root, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)
    return res.returncode

print("=== Git Status ===")
run_git(["status"])

print("\n=== Adding all updated files ===")
# Force add cloudflared.exe in backend if it's ignored or untracked
run_git(["add", "-A"])
# Explicitly add cloudflared.exe to make sure it's pushed as requested
run_git(["add", "-f", "backend/cloudflared.exe"])

print("\n=== Committing ===")
run_git(["commit", "-m", "Fix VLM remote URL loading, API URL fallback on Vercel, and auto-escape MONGO_URI credentials"])

print("\n=== Pushing to GitHub ===")
# Get current branch
res = subprocess.run(["git", "branch", "--show-current"], cwd=repo_root, capture_output=True, text=True)
branch = res.stdout.strip() or "main"
print(f"Active branch: {branch}")
run_git(["push", "origin", branch])
