#!/usr/bin/env bash
# ============================================================
# FedTAD V3 campaign 状态面板 (只读, 不触碰任何训练状态)
# 用法: bash scripts/campaign_stats.sh [dataset]
# 输出: 每 cell 的 phase/PID/trial/round/best_val/health + GPU/RAM
# ============================================================
set -uo pipefail
DS=${1:-Cora}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="$REPO/runs/formal_campaign_v3/$DS"
PY=${PY:-$(command -v python || command -v python3)}
NOW=$(date +%s)

fmt_elapsed() {  # start_epoch -> HH:MM:SS
  local s=${1:-$NOW} e=$((NOW-s))
  [ "$e" -lt 0 ] && e=0
  printf '%02d:%02d:%02d' $((e/3600)) $(((e%3600)/60)) $((e%60))
}

echo "================ FedTAD V3 campaign: $DS  ($(date '+%F %T')) ================"
if [ -f "$OUT/.max_cell_jobs" ]; then
  echo "MAX_CELL_JOBS (state): $(cat "$OUT/.max_cell_jobs")"
fi

for tier in 5 10 20; do
  d="$OUT/clients$tier"
  [ -d "$d" ] || { echo "  $DS-c$tier: (未启动)"; continue; }
  line="  $DS-c$tier:"

  # stage1
  if [ -f "$d/.stage1_running" ] && kill -0 "$(cat "$d/.stage1_running")" 2>/dev/null; then
    st=$(grep -m1 '^stage1 start' "$d/.stage1_times" 2>/dev/null | awk '{print $3}')
    rnd=$(grep -oE 'round [0-9]+' "$d/stage1_phase.log" 2>/dev/null | tail -1)
    line="$line STAGE1(pid=$(cat "$d/.stage1_running"), ${rnd:-?}, elapsed=$(fmt_elapsed "$st"))"
  elif [ -f "$d/stage1/stage1_health.json" ]; then
    v=$(grep -o '"verdict": "[A-Z]*"' "$d/stage1/stage1_health.json" | head -1)
    line="$line stage1=${v:-?}"
  else
    line="$line stage1=待调度"
  fi

  # tuning
  if [ -f "$d/.tuning_running" ] && kill -0 "$(cat "$d/.tuning_running")" 2>/dev/null; then
    st=$(grep -m1 '^tuning start' "$d/.tuning_times" 2>/dev/null | awk '{print $3}')
    tlog="$d/tuning_phase.log"
    last_trial=$(grep -oE 'trial [0-9]+' "$tlog" 2>/dev/null | tail -1)
    rnd=$(grep -oE '\[Round [0-9]+\]' "$d/trial_logs"/trial_*/stdout.log 2>/dev/null | tail -1)
    bv=$(find "$d/trial_logs" -name final_metrics.json 2>/dev/null \
      | xargs -r grep -h '"best_val_primary"' 2>/dev/null \
      | grep -oE '[0-9.]+' | sort -g | tail -1)
    line="$line TUNING($last_trial, ${rnd:-?}, best_val=${bv:-?}, elapsed=$(fmt_elapsed "$st"))"
  elif [ -f "$d/best_params.json" ]; then
    bv=$(find "$d/trial_logs" -name final_metrics.json 2>/dev/null \
      | xargs -r grep -h '"best_val_primary"' 2>/dev/null \
      | grep -oE '[0-9.]+' | sort -g | tail -1)
    line="$line tuning=完成 best_val=${bv:-?}"
    [ -f "$d/.stability_failure" ] && line="$line [稳定性失败: $(cat "$d/.stability_failure")]"
  fi

  # finals
  if [ -f "$d/.finals_running" ] && kill -0 "$(cat "$d/.finals_running")" 2>/dev/null; then
    st=$(grep -m1 '^finals start' "$d/.finals_times" 2>/dev/null | awk '{print $3}')
    flog="$d/finals_phase.log"
    cur_seed=$(grep -oE 'seed=[0-9]+' "$flog" 2>/dev/null | tail -1)
    rnd=$(grep -oE '\[Round [0-9]+\]' "$d/final"/seed_*/stdout.log 2>/dev/null | tail -1)
    line="$line FINALS(${cur_seed:-?}, ${rnd:-?}, elapsed=$(fmt_elapsed "$st"))"
  elif [ -f "$d/final/final_3seed_summary.json" ]; then
    line="$line finals=完成"
  fi

  echo "$line"
done

echo "--------------------------------------------------------------------------------"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | sed 's/^/GPU: used\/total MB + util%: /'
fi
free -g | awk 'NR==2 {print "RAM (GB): used " $3 " / total " $2}'
echo "ps (training-related):"
ps -eo pid,etime,args 2>/dev/null | grep -E "train_fedtad|formal_campaign|formal_final_eval" \
  | grep -v grep | sed 's/^/  /' || true
