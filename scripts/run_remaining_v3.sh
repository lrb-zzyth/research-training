#!/usr/bin/env bash
# ============================================================
# FedTAD V3 正式战役: 剩余数据集链式 launcher
# CiteSeer -> PubMed -> CS -> Physics (数据集之间严格串行)
#
# 用法 (必须在 tmux 中):
#   MAX_CELL_JOBS=3 bash scripts/run_remaining_v3.sh
#
# 逻辑:
#   for dataset in CiteSeer PubMed CS Physics:
#     MAX_CELL_JOBS bash scripts/run_dataset_v3.sh <ds> 12
#     -> 校验 3 个 cell 全部 final_3seed_summary.json 存在 (COMPLETE)
#        -> COMPLETE: 自动进入下一 dataset
#        -> INCOMPLETE: 记录 cell 状态, BLOCKING FAILURE, campaign STOP
#   Physics 完成后生成 15-cell 总表报告
#
# 断点恢复: 本脚本被杀/服务器重启后重新执行即安全续跑
# (run_dataset_v3.sh 依据 cell 产物 + study.db 续跑, 不重做已完成 trial)。
#
# 日志:
#   logs/formal_v3_<dataset>.log  每 dataset master log
#   logs/remaining_campaign.log   链式主日志 (含所有异常与原因)
# ============================================================
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=${PY:-$(command -v python || command -v python3)}
[ -x "$PY" ] || PY=/root/miniconda3/envs/fedtad5060/bin/python
export PY

DATASETS="CiteSeer PubMed CS Physics"
OUT="$REPO/runs/formal_campaign_v3"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR" "$OUT"
MAIN_LOG="$LOG_DIR/remaining_campaign.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

log() { echo "[remaining $(date '+%F %T')] $*"; }

dataset_complete() {  # $1=ds -> 0 如果 3 cells 全有 final summary
  local t
  for t in 5 10 20; do
    [ -f "$OUT/$1/clients$t/final/final_3seed_summary.json" ] || return 1
  done
  return 0
}

dump_cell_state() {  # $1=ds: 输出每 cell 失败原因 (blocking failure 审计)
  local t d
  for t in 5 10 20; do
    d="$OUT/$1/clients$t"
    echo "  [$1-c$t] stage1=$(grep -o '"verdict": "[A-Z]*"' "$d/stage1/stage1_health.json" 2>/dev/null || echo n/a)" \
         "stability=$(cat "$d/.stability_failure" 2>/dev/null || echo none)" \
         "stage1_exit=$(cat "$d/.stage1_exit" 2>/dev/null || echo -)" \
         "tuning_exit=$(cat "$d/.tuning_exit" 2>/dev/null || echo -)" \
         "finals_exit=$(cat "$d/.finals_exit" 2>/dev/null || echo -)"
  done
}

log "=================================================================="
log "V3 remaining campaign start: ${DATASETS}"
log "MAX_CELL_JOBS=${MAX_CELL_JOBS:-3} (OOM 时 dataset launcher 自动降并发)"
log "protocol: study n_jobs=1 | tuning test=0 | final test exactly once/seed"
log "=================================================================="

ALL_OK=1
for ds in $DATASETS; do
  master="$LOG_DIR/formal_v3_$(echo "$ds" | tr 'A-Z' 'a-z').log"
  log "---------- dataset $ds START (master log: $master) ----------"

  complete=0
  for attempt in 1 2 3; do
    MAX_CELL_JOBS=${MAX_CELL_JOBS:-3} bash scripts/run_dataset_v3.sh "$ds" 12 >>"$master" 2>&1
    rc=$?
    log "dataset $ds launcher exit rc=$rc (attempt $attempt/3)"
    if dataset_complete "$ds"; then
      complete=1
      break
    fi
    if [ "$rc" = 0 ]; then
      log "dataset $ds: launcher 正常退出但 cells 未全完成 (cell-level failure, 不再重试)"
      break
    fi
    log "dataset $ds: launcher 异常退出 (crash/resource), 10s 后重试续跑"
    sleep 10
  done

  if [ "$complete" = 1 ]; then
    log "dataset $ds COMPLETE (3/3 cells finals done) -> next dataset"
  else
    log "dataset $ds INCOMPLETE -> BLOCKING FAILURE, campaign STOP"
    log "cell states:"
    dump_cell_state "$ds"
    ALL_OK=0
    break
  fi
done

log "---------- generating all-datasets report ----------"
"$PY" scripts/summarize_all_v3.py \
  || log "WARNING: summarize_all_v3.py failed; 请人工检查"
log "REMAINING_CAMPAIGN_DONE (all_complete=$ALL_OK)"
