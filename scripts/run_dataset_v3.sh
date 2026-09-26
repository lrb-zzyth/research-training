#!/usr/bin/env bash
# ============================================================
# FedTAD V3 正式战役: dataset-level launcher (跨 cell 并行, study 内串行)
#
# 用法:
#   MAX_CELL_JOBS=2 bash scripts/run_dataset_v3.sh Cora 12
#   $1 = dataset         (如 Cora)
#   $2 = target_healthy  (每 cell 健康 trial 数, 如 12)
#
# 环境变量:
#   MAX_CELL_JOBS   跨 cell 最大并行进程数 (默认 2, 启动时写入状态文件)
#   MAX_ATTEMPTS    每 cell Optuna attempt 上限 (V3 协议固定 20)
#
# 每 cell 流程 (严格顺序, 每阶段独立 OS 进程):
#   Stage1 联邦 DDPM 预训练  (scripts/generate_stage1_cell.sh)
#     -> Stage1 health gate  (verdict=PASS 才继续; FAIL 则该 cell STOP)
#     -> Optuna tuning       (scripts/run_formal_cell.sh, n_jobs=1)
#     -> 稳定性格口          (healthy >= target 才允许 finals, 否则
#                              FORMAL NUMERICAL STABILITY FAILURE, cell STOP)
#     -> 3-seed final        (scripts/run_formal_final.sh, --final_test_only,
#                              同 cell 三 seed 顺序执行)
#
# 断点恢复: 依据 cell 目录产物 + .phase_running 锁判断阶段;
# SSH 断开 (需在 tmux 中) / launcher 被杀 / 服务器重启后,
# 重新执行本脚本即安全续跑 (study resume 由 formal_campaign.py 校验)。
#
# 动态调整并发 (运行中):
#   echo 3 > runs/formal_campaign_v3/<ds>/.max_cell_jobs
#
# OOM 处理 (协议第 44 条): phase 日志出现 CUDA/内存 OOM 或退出码 137 时,
# launcher 自动把并发数减 1 (最低 1) 后按同配置重新运行该 phase,
# 绝不通过改 trial 配置 (fake_nodes/hid_dim/batch) 绕过。
# ============================================================
set -uo pipefail

DS=${1:?usage: run_dataset_v3.sh <dataset> <target_healthy>}
TARGET=${2:?usage: run_dataset_v3.sh <dataset> <target_healthy>}
TIERS="5 10 20"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
PHASE_MAX_LAUNCHES=4          # 每 phase 初始 1 次 + 最多 3 次重试
PHASE_OOM_LIMIT=2             # 每 cell 每 phase 最多因 OOM 降并发次数
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="$REPO/runs/formal_campaign_v3/$DS"
PY=${PY:-$(command -v python || command -v python3)}
[ -x "$PY" ] || PY=/root/miniconda3/envs/fedtad5060/bin/python
export PY
cd "$REPO"

mkdir -p "$OUT"
STATE="$OUT/.max_cell_jobs"
if [ ! -f "$STATE" ]; then
  echo "${MAX_CELL_JOBS:-2}" > "$STATE"
fi
exec > >(tee -a "$OUT/campaign.log") 2>&1

log()  { echo "[campaign $(date '+%F %T')] $*"; }
cell_dir() { echo "$OUT/clients$1"; }
maxj() { cat "$STATE" 2>/dev/null || echo "${MAX_CELL_JOBS:-2}"; }

# ---------------- 槽位与锁 ----------------

live_slots() {   # 输出当前存活 phase 进程数; 顺带清理陈旧锁
  local n=0 tier phase pid lock
  for tier in $TIERS; do
    for phase in stage1 tuning finals; do
      lock="$(cell_dir "$tier")/.${phase}_running"
      [ -f "$lock" ] || continue
      pid=$(awk '{print $1}' "$lock")
      if kill -0 "$pid" 2>/dev/null; then
        n=$((n+1))
      else
        rm -f "$lock"
      fi
    done
  done
  echo "$n"
}

launch_phase() {  # tier phase cmd...
  local tier=$1 phase=$2; shift 2
  local d; d=$(cell_dir "$tier")
  local lock="$d/.${phase}_running" cnt="$d/.${phase}_launches"
  mkdir -p "$d"
  local n=0
  [ -f "$cnt" ] && n=$(cat "$cnt")
  if [ "$n" -ge "$PHASE_MAX_LAUNCHES" ]; then return 1; fi
  echo $((n+1)) > "$cnt"
  echo "$phase start $(date +%s)" >> "$d/.${phase}_times"
  setsid bash -c '
    d=$1; phase=$2; shift 2
    "$@" >"$d/${phase}_phase.log" 2>&1
    rc=$?
    echo "$rc" > "$d/.${phase}_exit"
    echo "$phase end $(date +%s) rc=$rc" >> "$d/.${phase}_times"
    rm -f "$d/.${phase}_running"
  ' _ "$d" "$phase" "$@" </dev/null >/dev/null 2>&1 &
  local pid=$!
  echo "$pid" > "$lock"
  log "launch $DS c$tier $phase pid=$pid (launch #$((n+1)))"
  return 0
}

phase_oom() {   # tier phase: 0 = OOM 迹象 (日志或退出码 137)
  local d; d=$(cell_dir "$1")
  grep -qiE "out of memory|CUDA OOM|cannot allocate memory|Killed" \
    "$d/$2_phase.log" 2>/dev/null && return 0
  local rc; rc=$(cat "$d/.$2_exit" 2>/dev/null || echo 0)
  [ "$rc" = "137" ] && return 0
  return 1
}

maybe_retry() {  # tier phase: 0 = 该 phase 退出非零且未达最大启动次数, 可重试
  local tier=$1 phase=$2 d; d=$(cell_dir "$tier")
  [ -f "$d/.${phase}_exit" ] || return 1
  local rc; rc=$(cat "$d/.${phase}_exit" 2>/dev/null || echo 0)
  [ "$rc" = "0" ] && return 1
  local n=0
  [ -f "$d/.${phase}_launches" ] && n=$(cat "$d/.${phase}_launches")
  [ "$n" -lt "$PHASE_MAX_LAUNCHES" ] && return 0
  return 1
}

# ---------------- 阶段状态 (只读判断, 不改状态) ----------------
# 返回: running | done | failed | todo

stage1_status() {
  local d; d=$(cell_dir "$1")
  if [ -f "$d/.stage1_running" ] && kill -0 "$(cat "$d/.stage1_running")" 2>/dev/null; then
    echo running; return
  fi
  if [ -f "$d/stage1/stage1_health.json" ]; then
    if grep -q '"verdict": "PASS"' "$d/stage1/stage1_health.json"; then
      echo done; return
    fi
    echo failed; return
  fi
  if [ -f "$d/.stage1_exit" ] && ! maybe_retry "$1" stage1; then echo failed; return; fi
  echo todo
}

tuning_status() {
  local d; d=$(cell_dir "$1")
  if [ -f "$d/.tuning_running" ] && kill -0 "$(cat "$d/.tuning_running")" 2>/dev/null; then
    echo running; return
  fi
  if [ -f "$d/best_params.json" ]; then
    [ -f "$d/.stability_failure" ] && { echo failed; return; }
    echo done; return
  fi
  if [ -f "$d/.tuning_exit" ] && ! maybe_retry "$1" tuning; then echo failed; return; fi
  echo todo
}

finals_status() {
  local d; d=$(cell_dir "$1")
  if [ -f "$d/.finals_running" ] && kill -0 "$(cat "$d/.finals_running")" 2>/dev/null; then
    echo running; return
  fi
  if [ -f "$d/final/final_3seed_summary.json" ]; then echo done; return; fi
  if [ -f "$d/.finals_exit" ] && ! maybe_retry "$1" finals; then echo failed; return; fi
  echo todo
}

tuning_healthy_count() {  # tier -> "healthy attempts" (study.db 只读)
  local d; d=$(cell_dir "$1")
  $PY - "$d/study.db" "$DS" "$1" <<'PY'
import optuna, sys
storage = 'sqlite:///' + sys.argv[1]
ds, tier = sys.argv[2], sys.argv[3]
try:
    st = optuna.load_study(study_name='formal_v3_%s_c%s' % (ds, tier),
                           storage=storage)
    h = sum(1 for t in st.trials
            if t.state.name == 'COMPLETE' and t.user_attrs.get('health') == 'PASS')
    print(h, len(st.trials))
except Exception as e:
    print('ERR %s' % e)
PY
}

# ---------------- 主循环 ----------------

log "=================================================================="
log "V3 dataset campaign start: $DS  target_healthy=$TARGET  max_attempts=$MAX_ATTEMPTS"
log "MAX_CELL_JOBS=$(maxj)  (修改: echo N > $STATE)"
log "KL_DIRECTION = global||local   (各 cell protocol.json kl_direction 字段已记录)"
log "contrastive = subgraph_cross_view | RWR = component-safe V3 | projection = overflow-safe V3"
log "study n_jobs = 1 | test isolation = tuning_mode + final_test_only"
log "=================================================================="

while :; do
  MAXJ=$(maxj)
  slots=$(live_slots)

  # ---- 全局终止判断 ----
  all_terminal=1
  for tier in $TIERS; do
    case $(stage1_status "$tier") in
      running|todo) all_terminal=0; continue;;
      failed) continue;;
    esac
    case $(tuning_status "$tier") in
      running|todo) all_terminal=0; continue;;
      failed) continue;;
    esac
    case $(finals_status "$tier") in
      done) continue;;
      *) all_terminal=0;;
    esac
  done
  if [ "$all_terminal" = 1 ] && [ "$slots" = 0 ]; then
    log "all cells terminal; campaign loop exiting"
    break
  fi

  # ---- OOM 降并发 + 允许按同配置重试 (协议第 44 条) ----
  for tier in $TIERS; do
    d=$(cell_dir "$tier")
    for phase in stage1 tuning finals; do
      [ -f "$d/.${phase}_exit" ] || continue
      [ -f "$d/.${phase}_running" ] && continue
      if phase_oom "$tier" "$phase"; then
        oc=0
        [ -f "$d/.${phase}_oom_count" ] && oc=$(cat "$d/.${phase}_oom_count")
        if [ "$oc" -lt "$PHASE_OOM_LIMIT" ]; then
          echo $((oc+1)) > "$d/.${phase}_oom_count"
          cur=$(maxj)
          if [ "$cur" -gt 1 ]; then echo $((cur-1)) > "$STATE"; fi
          rm -f "$d/.${phase}_exit"    # 重置 -> 状态回到 todo, 同配置重跑
          log "OOM detected in $DS c$tier $phase (count=$((oc+1))): MAX_CELL_JOBS -> $(maxj); 按同配置重新运行"
        else
          log "OOM repeated in $DS c$tier $phase; 保持 cell STOP, 等待人工检查"
        fi
      fi
    done
  done

  if [ "$slots" -ge "$MAXJ" ]; then
    sleep 30
    continue
  fi

  # ---- 调度 (每个 cell 严格按 stage1 -> tuning -> finals) ----
  for tier in $TIERS; do
    slots=$(live_slots); MAXJ=$(maxj)
    [ "$slots" -ge "$MAXJ" ] && break
    d=$(cell_dir "$tier")

    case $(stage1_status "$tier") in
      running) continue;;
      failed)  log "$DS c$tier: Stage1 FAIL -> cell STOP"; continue;;
      todo)    launch_phase "$tier" stage1 bash scripts/generate_stage1_cell.sh "$DS" "$tier"; continue;;
    esac

    # Stage1 通过后补写 meta (SHA256 + code/data hash + config + health)
    if [ ! -f "$d/stage1/stage1_meta.json" ]; then
      "$PY" - "$d" "$DS" "$tier" <<'PY'
import hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'accuracy_benchmark'))
import formal_campaign as fc
d, ds, tier = sys.argv[1], sys.argv[2], sys.argv[3]
ck = os.path.join(d, 'stage1_generator.pt')
sha = hashlib.sha256(open(ck, 'rb').read()).hexdigest()
health = json.load(open(os.path.join(d, 'stage1', 'stage1_health.json')))
meta = {'stage1_checkpoint': ck, 'stage1_sha256': sha,
        'code_hash': fc.code_hash(),
        'data_identity_hash': fc.compute_data_identity(ds, int(tier)),
        'config': ('conditional DDPM, T=20, beta_end=0.5, posterior_variance=ON, '
                   'timestep_scalar residual=ON, federated diffusion pretrain=ON'),
        'health_report': health}
json.dump(meta, open(os.path.join(d, 'stage1', 'stage1_meta.json'), 'w'), indent=1)
print('stage1 meta written, sha256=', sha[:16])
PY
      log "$DS c$tier: stage1 meta written"
    fi

    case $(tuning_status "$tier") in
      running) continue;;
      failed)  log "$DS c$tier: tuning FAIL -> cell STOP"; continue;;
      todo)    launch_phase "$tier" tuning bash scripts/run_formal_cell.sh "$DS" "$tier" "$TARGET" "$MAX_ATTEMPTS" 1; continue;;
    esac

    # tuning done -> 稳定性格口: healthy >= target 才允许 finals
    if [ ! -f "$d/.stability_checked" ]; then
      read -r healthy attempts < <(tuning_healthy_count "$tier")
      if [ "$healthy" = "ERR" ]; then
        log "$DS c$tier: stability check deferred ($attempts)"
        continue
      fi
      if [ "$healthy" -ge "$TARGET" ]; then
        echo "healthy=$healthy attempts=$attempts" > "$d/.stability_checked"
        log "$DS c$tier: stability OK (healthy=$healthy >= $TARGET, attempts=$attempts)"
      else
        echo "healthy=$healthy attempts=$attempts target=$TARGET" > "$d/.stability_failure"
        log "STOP $DS c$tier: FORMAL NUMERICAL STABILITY FAILURE (healthy=$healthy < $TARGET, attempts=$attempts)"
        continue
      fi
    fi
    if [ -f "$d/.stability_failure" ]; then continue; fi

    case $(finals_status "$tier") in
      running) continue;;
      failed)  log "$DS c$tier: finals FAIL -> cell STOP"; continue;;
      todo)    launch_phase "$tier" finals bash scripts/run_formal_final.sh "$DS" "$tier"; continue;;
    esac
  done

  sleep 30
done

log "campaign loop exited; generating dataset summary"
"$PY" scripts/summarize_dataset_v3.py "$DS" \
  || log "WARNING: summary generation failed; 请人工运行 summarize_dataset_v3.py"
log "DATASET_CAMPAIGN_FINISHED $DS"
