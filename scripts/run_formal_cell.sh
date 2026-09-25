#!/usr/bin/env bash
# 正式 v2 战役单 cell 启动脚本 (模板, 本轮不执行)。
# 用法: bash scripts/run_formal_cell.sh <dataset> <num_clients> <target_healthy>
# 例如: bash scripts/run_formal_cell.sh Cora 5 12
# 前置: 1) server_smoke_test.sh 全 PASS; 2) 该 cell 的 Stage1 checkpoint 已生成
#       于 runs/formal_campaign_v2/<ds>/clients<tier>/stage1_generator.pt;
#       3) 建议在 tmux 中运行以断线保持 (见 SERVER_MIGRATION_BEGINNER.md)。
set -e
DS=${1:?请提供 dataset, 如 Cora}
NC=${2:?请提供 num_clients, 如 5}
TARGET=${3:-12}
MAX_ATTEMPTS=${4:-15}
N_JOBS=${5:-2}   # 服务器大显存可 2-3; 本地 7GB 机器必须 1
# 服务器可移植: 优先用已激活 conda 环境的 python, 否则用 PATH 里的
PY=${PY:-$(command -v python || command -v python3)}
REPO=$(cd "$(dirname "$0")/.." && pwd)

echo "[Formal v2] 启动 $DS c$NC (target_healthy=$TARGET, max_attempts=$MAX_ATTEMPTS, n_jobs=$N_JOBS)"
cd "$REPO"
exec $PY experiments/accuracy_benchmark/formal_campaign.py \
    --target_healthy "$TARGET" --max_attempts "$MAX_ATTEMPTS" \
    --n_jobs "$N_JOBS" --gpu_mem_gate_mb 6000 \
    --cells "$DS:$NC"
