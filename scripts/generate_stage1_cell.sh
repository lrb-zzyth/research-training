#!/usr/bin/env bash
# Generate one formal-v3 Stage1 conditional-DDPM checkpoint, strictly serial.
# Usage: bash scripts/generate_stage1_cell.sh Cora 5
set -euo pipefail
DS=${1:?dataset, e.g. Cora}
NC=${2:?num_clients, e.g. 5}
PY=${PY:-$(command -v python || command -v python3)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
OUT="runs/formal_campaign_v3/$DS/clients$NC"
mkdir -p "$OUT/stage1"

echo "[Stage1 v3] $DS c$NC"
"$PY" train_fedtad.py --root ./dataset --dataset "$DS" --num_clients "$NC" \
  --partition Louvain --num_rounds 1 --num_epochs 3 --hid_dim 64 --dropout 0.5 \
  --lr 1e-2 --weight_decay 5e-4 --task_mode multiclass \
  --selection_metric accuracy --f1_threshold=-1e6 --auc_threshold=-1e6 \
  --seed 2024 --reliability_holdout_ratio 0 \
  --diffusion_skip_mode timestep_scalar --diffusion_pretrain_rounds 20 \
  --pretrain_diagnostic_only \
  --checkpoint_dir "$OUT/stage1" \
  > "$OUT/stage1/stage1_training.log" 2>&1

CK="$OUT/stage1/pretrained_generator.pt"
[ -f "$CK" ] || { echo "ERROR: missing $CK" >&2; exit 3; }
cp "$CK" "$OUT/stage1_generator.pt"

"$PY" - "$OUT" "$DS" "$NC" <<'PY'
import json, os, re, sys, torch
out, ds, nc = sys.argv[1], sys.argv[2], int(sys.argv[3])
log_path=os.path.join(out,'stage1','stage1_training.log')
text=open(log_path,errors='replace').read()
nonfinite=bool(re.search(r'(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)', text, re.I))
ck=os.path.join(out,'stage1_generator.pt')
state=torch.load(ck,map_location='cpu',weights_only=False)
all_finite=all(torch.isfinite(v).all().item() for v in state.values() if torch.is_tensor(v))
health={'cell':f'{ds}_c{nc}','checkpoint':ck,
        'checkpoint_tensors_finite':bool(all_finite),
        'log_nonfinite_token_detected':bool(nonfinite),
        'verdict':'PASS' if all_finite and not nonfinite else 'REVIEW'}
json.dump(health,open(os.path.join(out,'stage1','stage1_health.json'),'w'),indent=2)
print(json.dumps(health,indent=2))
if health['verdict']!='PASS':
    raise SystemExit(4)
PY

echo "[Stage1 v3] PASS: $OUT/stage1_generator.pt"
