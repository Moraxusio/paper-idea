#!/usr/bin/env bash
# Run any project script with conda env mpfadet.
set -euo pipefail
PY=/home/finnwe/miniconda3/envs/mpfadet/bin/python
if [[ ! -x "$PY" ]]; then
  echo "missing conda env mpfadet: $PY" >&2
  exit 1
fi
exec "$PY" "$@"
