#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-inversion}"
if [[ $# -gt 0 ]]; then
  shift
fi

python scripts/reproduce_neumatc.py --task "$TASK" "$@"
