#!/usr/bin/env bash
# 批量生成 15 个 cell 的 Stage1 checkpoint (联邦扩散预训练 R20 + 健康检查落盘)。
# 用法: bash scripts/generate_stage1_all.sh [并行数, 正式协议仅允许1]
# 每个 cell: runs/formal_campaign_v3/<ds>/clients<tier>/stage1/ 下产出
#   pretrained_generator.pt / stage1_training.log / stage1_health.json
#   + 复制 stage1_generator.pt 到 cell 根目录。
set -u
P=${1:-1}
if [ "$P" != "1" ]; then
  echo "ERROR: single-GPU formal protocol requires serial Stage1 generation (P=1)." >&2
  exit 2
fi
PY=${PY:-$(command -v python || command -v python3)}
cd "$(dirname "$0")/.."

cells="Cora:5 Cora:10 Cora:20 CiteSeer:5 CiteSeer:10 CiteSeer:20 PubMed:5 PubMed:10 PubMed:20 CS:5 CS:10 CS:20 Physics:5 Physics:10 Physics:20"

run_one() {
  IFS=: read -r ds tier <<< "$1"
  d="runs/formal_campaign_v3/$ds/clients$tier"
  mkdir -p "$d/stage1"
  echo "[Stage1] $ds c$tier 开始"
  $PY train_fedtad.py --root ./dataset --dataset "$ds" --num_clients "$tier" \
    --partition Louvain --num_rounds 1 --num_epochs 3 --hid_dim 64 --dropout 0.5 \
    --lr 1e-2 --weight_decay 5e-4 --task_mode multiclass \
    --selection_metric accuracy --f1_threshold=-1e6 --auc_threshold=-1e6 \
    --seed 2024 --reliability_holdout_ratio 0 \
    --diffusion_skip_mode timestep_scalar --diffusion_pretrain_rounds 20 \
    --pretrain_diagnostic_only \
    --checkpoint_dir "$d/stage1" \
    > "$d/stage1/stage1_training.log" 2>&1
  if [ -f "$d/stage1/pretrained_generator.pt" ]; then
    cp "$d/stage1/pretrained_generator.pt" "$d/stage1_generator.pt"
    $PY - "$d" "$ds" "$tier" << 'PYEOF'
import json, re, sys, os
d, ds, tier = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(os.path.join(d, 'stage1', 'stage1_training.log'),
           errors='replace').read()
def g(pat, grp=1):
    m = re.search(pat, log)
    return m.group(grp) if m else None
raw_r = g(r'\[Pretrain-Only 诊断\].*r=([0-9.]+)')
real_r = g(r'\[real stats\].*median_r=([0-9.]+)')
mse = g(r'round 19: 样本加权平均去噪损失=([0-9.]+)')
ratio = (float(raw_r) / float(real_r)) if raw_r and real_r and float(real_r) > 0 else None
fin = not re.search(r'=inf|NAN-DIAG|nan\b.*loss', log)
verdict = 'PASS' if (fin and ratio is not None and ratio < 20
                     and mse is not None and float(mse) < 0.5) else 'REVIEW'
json.dump({'cell': f'{ds}_c{tier}', 'raw_median_r': raw_r,
           'real_median_r': real_r, 'r_over_real': ratio,
           'final_denoise_mse': mse, 'nonfinite_detected': not fin,
           'verdict': verdict}, open(os.path.join(d, 'stage1', 'stage1_health.json'), 'w'), indent=1)
print(f'[Stage1] {ds} c{tier} 完成: verdict={verdict} r/real={ratio}')
PYEOF
  else
    echo "[Stage1] $ds c$tier 失败: 无 checkpoint, 见 $d/stage1/stage1_training.log"
  fi
}
export -f run_one
export PY
echo "$cells" | tr ' ' '\n' | xargs -P "$P" -I{} bash -c 'run_one "$@"' _ {}
echo "STAGE1_ALL_DONE"
