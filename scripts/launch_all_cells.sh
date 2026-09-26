#!/usr/bin/env bash
# Formal v3: all 15 cells strictly serial on a single GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
for ds in Cora CiteSeer PubMed CS Physics; do
  for tier in 5 10 20; do
    echo "[FORMAL] starting ${ds} c${tier} (serial)"
    bash scripts/run_formal_cell.sh "$ds" "$tier" 12 15 1
  done
done
echo "FORMAL_ALL_CELLS_DONE"
