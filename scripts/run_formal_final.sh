#!/usr/bin/env bash
# Formal v3 final 3-seed evaluation (strictly serial, test exactly once per seed).
set -euo pipefail
DS=${1:?dataset, e.g. Cora}
NC=${2:?num_clients, e.g. 5}
PY=${PY:-$(command -v python || command -v python3)}
cd "$(dirname "$0")/.."
exec "$PY" experiments/accuracy_benchmark/formal_final_eval.py --cell "$DS:$NC"
