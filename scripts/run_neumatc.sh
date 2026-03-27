#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TASK="${1:-inversion}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python "$REPO_ROOT/scripts/reproduce_neumatc.py" --task "$TASK" "$@"
