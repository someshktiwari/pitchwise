"""
scripts/deploy_hf_space.py

Deploys Pitchwise to a Hugging Face Docker Space (D-020): creates the Space
if needed, sets the model API keys as Space secrets from your local .env,
and uploads the git-tracked files with the Space's own README card.

Hugging Face then builds the Dockerfile, which bakes the embedding model
and the vector index into the image (D-002, D-015), so the first build
takes several minutes.

Needs a Hugging Face access token with write permission
(huggingface.co/settings/tokens), in HF_TOKEN or from `hf auth login`.

Run from the project root:
  uv run python scripts/deploy_hf_space.py                   # <your-username>/pitchwise
  uv run python scripts/deploy_hf_space.py --space you/name  # another Space name
  uv run python scripts/deploy_hf_space.py --no-secrets      # upload code only

Only files tracked by git are uploaded, so .env and the local index are
never sent; notebooks, screenshots and evaluation results are left out
because the running app does not use them.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_README = ROOT / "deploy" / "huggingface" / "README.md"

SECRETS = ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
           "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
VARIABLES = {
    "PITCHWISE_RATE_PER_MINUTE": "6",   # questions per visitor per minute (ratelimit.py)
    "PITCHWISE_DAILY_LIMIT": "500",     # questions per day across all visitors
    "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
}
EXCLUDE_PREFIXES = ("notebooks/", "screenshots/", "deploy/", ".github/")
EXCLUDE_SUFFIXES = (".jsonl", ".ipynb", ".png", ".jpg", ".gif")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    for path in out.splitlines():
        if path.startswith(EXCLUDE_PREFIXES) or path.endswith(EXCLUDE_SUFFIXES):
            continue
        yield path


def stage(dest: Path):
    for path in tracked_files():
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    shutil.copy2(SPACE_README, dest / "README.md")  # the Space card replaces the GitHub README


def main():
    parser = argparse.ArgumentParser(description="Deploy Pitchwise to a Hugging Face Docker Space.")
    parser.add_argument("--space", help="Space id, e.g. username/pitchwise (default: <you>/pitchwise)")
    parser.add_argument("--no-secrets", action="store_true", help="do not set or update Space secrets")
    args = parser.parse_args()

    api = HfApi(token=os.getenv("HF_TOKEN"))
    user = api.whoami()["name"]
    space = args.space or f"{user}/pitchwise"

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space: https://huggingface.co/spaces/{space}")

    if not args.no_secrets:
        env = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k in SECRETS}}
        found = [k for k in SECRETS if env.get(k)]
        if not any(k in found for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")):
            sys.exit("No model API key in .env or the environment; the Space could not answer. "
                     "Add one, or pass --no-secrets if the Space already has them.")
        for key in found:
            api.add_space_secret(space, key, env[key])
        for key, value in VARIABLES.items():
            api.add_space_variable(space, key, value)
        print(f"Secrets set: {', '.join(found)} (values are not printed)")

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp,
                          commit_message="Deploy Pitchwise from GitHub",
                          delete_patterns=["*"])  # mirror: remove files no longer in the repo
    print("Uploaded. The Space is building; watch the logs on the Space page.")


if __name__ == "__main__":
    main()
