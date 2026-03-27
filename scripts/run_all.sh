#!/usr/bin/env bash
set -euo pipefail

python scripts/reproduce_neumatc.py --task inversion "$@"
python scripts/reproduce_neumatc.py --task svd "$@"
