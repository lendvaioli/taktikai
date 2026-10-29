#!/usr/bin/env bash
# Build the static Space, check it against the publication rules, then upload it (only with
# --push). Usage:
#   scripts/publish_space.sh <hf-username>/taktikai           # build + checks + print command
#   scripts/publish_space.sh <hf-username>/taktikai --push    # ... and upload with the hf CLI
# Needs: uv (project environment with the ml extra) and, for --push, the hf CLI logged in
# (pip install hf; hf auth login). The Space must exist (huggingface.co/new-space, SDK: Static).
set -euo pipefail

space_id="${1:?usage: scripts/publish_space.sh <hf-username>/<space-name> [--push]}"
push="${2:-}"
cd "$(dirname "$0")/.."

MPLBACKEND=Agg uv run python scripts/build_space.py
uv run pytest -q tests/test_publication.py tests/space   # the guard checks space/ on disk

echo "Files to upload:"
find space -type f | sort
cmd=(hf upload "$space_id" space . --repo-type space --commit-message "Update TaktikAI demo")
if [ "$push" = "--push" ]; then
  "${cmd[@]}"
  echo "Uploaded: https://huggingface.co/spaces/$space_id"
else
  echo "Dry run. To upload, run again with --push, or: ${cmd[*]}"
fi
