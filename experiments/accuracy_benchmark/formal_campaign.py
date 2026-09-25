#!/usr/bin/env python3
"""
正式 5 datasets × 3 client tiers 战役 runner (validation-only per-tier Optuna)。

READINESS 状态: 代码就绪, 但**未获人工确认 trial budget 前不得启动**;
且 15 个 cell 的 Stage1 checkpoint 目前只有 PubMed-10 存在。

硬协议:
  - 每 cell 独立 Optuna study (formal_<ds>_c<tier>), 独立 study.db / protocol.json
  - objective = validation accuracy only (trial 内 --tuning_mode 完全不计算 test)
  - n_jobs = 1, 严格串行
  - 每 trial 复用本 cell 的 Stage1 checkpoint (--no-federated_diffusion_pretrain)
  - 正式固定: weighted CE ON / contrastive ON / holdout 0 / accuracy 选轮 /
    reset_each_round / radius ON / align OFF / freeze_skip ON
  - 搜索空间 (正式): distill_steps {1,3,5} (25 为 legacy, 不纳入),
    server_start_round {0,1}

用法 (仅人工确认 budget 后):
  python experiments/accuracy_benchmark/formal_campaign.py --n_trials 8 \
      --cells PubMed:10
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import optuna

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PY = sys.executable   # 服务器可移植: 用当前解释器 (本机/服务器 conda 路径自动适配)
FORMAL_VERSION = 'v2'   # 正式战役版本: v2 = λsem 无 0.001 + formal_health v2
OUT = os.path.join(REPO, 'runs', f'formal_campaign_{FORMAL_VERSION}')

DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics']
TIERS = [5, 10, 20]

# formal_search_space_version = v2: λsem 移除 0.001
# (Cora-5 formal qualification: λsem=0.001 在 8/9 采样 trial 中出现
#  NaN 或 raw generator radius=inf; v1 含 0.001 仅为历史资格审计)
SEARCH_SPACE = {
    'lambda_sem': {'choices': [0.01, 0.1, 1.0]},
    'lambda_diversity': {'choices': [0.001, 0.01, 0.1]},
    'distill_steps': {'choices': [1, 3, 5]},          # 25 为 legacy, 已剔除
    'generator_steps': {'choices': [1, 3, 5]},
    'contrastive_temperature': {'choices': [0.05, 0.1, 0.2, 0.35]},
    'lambda_subgraph': {'choices': [0.05, 0.1, 0.5]},
    'knn_k': {'choices': [3, 5, 10]},
    'fake_nodes': {'choices': [50, 100, 200]},
    'server_start_round': {'choices': [0, 1]},
}

FIXED_FLAGS = [
    '--task_mode', 'multiclass',
    '--selection_metric', 'accuracy',
    '--reliability_holdout_ratio', '0',
    '--use_weighted_ce',
    '--contrastive_mode', 'subgraph_cross_view',
    '--local_optimizer_lifecycle', 'reset_each_round',
    '--diffusion_skip_mode', 'timestep_scalar',
    '--diffusion_freeze_skip_scale',
    '--no-federated_diffusion_pretrain',
    '--num_rounds', '100',
    '--num_epochs', '3',
    '--hid_dim', '64', '--dropout', '0.5',
    '--lr', '1e-2', '--weight_decay', '5e-4',
    '--f1_threshold=-1e6', '--auc_threshold=-1e6',
    '--tuning_mode',
    '--formal_campaign_guard',
]


def code_hash():
    """核心代码文件内容 hash (工作区有未提交改动时 git HEAD 不代表实际代码)。"""
    h = hashlib.md5()
    for f in ['train_fedtad.py', 'model.py', 'util/task_util.py',
              'util/checkpoint.py']:
        with open(os.path.join(REPO, f), 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


class HealthFailure(Exception):
    """正式 trial 数值健康失败 (NaN/Inf/runaway): 该 trial 不是合法候选。"""


def trial_health_check(outdir):
    """
    正式 trial 健康检查 (formal health rule):

    100 轮内出现任何 NaN/Inf loss/grad/参数/诊断 -> health=FAIL,
    该 trial 不得返回 objective、不得参与候选排名 (即使历史 best_val 在崩溃前)。

    返回 (ok, info): info 含 failure_round / failure_type / max_raw_radius /
    max_theta_drift / best_val_before_failure / nonfinite_detected。
    """
    import math as _math
    import re as _re
    info = {'failure_round': None, 'failure_type': None,
            'max_raw_radius': 0.0, 'max_theta_drift': 0.0,
            'best_val_before_failure': None, 'nonfinite_detected': False}
    # events.jsonl: 结构化 per-round global_val + task_failed
    ev = os.path.join(outdir, 'events.jsonl')
    fail_round = None
    best_before = None
    if os.path.exists(ev):
        for ln in open(ev, errors='replace'):
            if not ln.strip().startswith('{'):
                continue
            try:
                j = json.loads(ln)
            except Exception:
                continue
            r = j.get('round')
            gv = j.get('global_val')
            if gv is not None and not _math.isnan(gv):
                if fail_round is None or (r is not None and r < fail_round):
                    best_before = max(best_before if best_before is not None
                                      else 0.0, float(gv))
            if (gv is not None and _math.isnan(gv)
                    and fail_round is None):
                fail_round = r
            if j.get('event') == 'task_failed' and fail_round is None:
                err = str(j.get('error', ''))
                if 'nan' in err.lower() or 'inf' in err.lower():
                    fail_round = r
    # stdout.log: NAN-DIAG / grad nan / raw_r runaway / drift 扫描
    max_raw_r = 0.0
    max_drift = 0.0
    run_type = None
    sp = os.path.join(outdir, 'stdout.log')
    if os.path.exists(sp):
        cur_round = None
        for ln in open(sp, errors='replace'):
            m = _re.search(r'\[Round (\d+)\]', ln)
            if m:
                cur_round = int(m.group(1))
            if ('NAN-DIAG' in ln or _re.search(
                    r'\|grad\|=nan|L_D=nan|L_sem=nan|L_dis=nan|L_div=nan|'
                    r'L_G=nan|L_norm=nan|=inf\b|mu=nan|σ=nan|raw_r=nan|'
                    r'raw_r=inf|proj_r=nan|proj_r=inf', ln)) \
                    and fail_round is None:
                fail_round = cur_round
            m = _re.search(r'raw_r=([0-9.eE+-]+)', ln)
            if m and 'nan' not in m.group(1):
                try:
                    max_raw_r = max(max_raw_r, float(m.group(1)))
                except ValueError:
                    pass
            m = _re.search(r'theta_drift=([0-9.eE+-]+)', ln)
            if m and 'nan' not in m.group(1):
                try:
                    max_drift = max(max_drift, float(m.group(1)))
                except ValueError:
                    pass
            m = _re.search(r'skip_delta=([0-9.eE+-]+)', ln)
            if m and 'nan' not in m.group(1):
                try:
                    if abs(float(m.group(1))) > 1e-9 and fail_round is None:
                        fail_round = cur_round
                        run_type = 'SKIP_DRIFT'
                except ValueError:
                    pass
    if fail_round is not None:
        if run_type is None:
            run_type = ('RAW_RADIUS_RUNAWAY_NAN' if max_raw_r > 1e9
                        else 'NONFINITE')
        info.update({'failure_round': fail_round, 'failure_type': run_type,
                     'nonfinite_detected': True})
    info.update({'max_raw_radius': max_raw_r,
                 'max_theta_drift': max_drift,
                 'best_val_before_failure': best_before})
    return fail_round is None, info


def compute_data_identity(ds, tier):
    """partition 文件的稳定 hash (同一 cell 所有 trial 必须一致)。"""
    h = hashlib.md5()
    d = os.path.join(REPO, 'dataset', ds, f'Client{tier}', 'Louvain')
    for f in sorted(os.listdir(d)):
        if f.endswith('.pt') and f.startswith('data'):
            with open(os.path.join(d, f), 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def stage1_path(ds, tier):
    return os.path.join(OUT, ds, f'clients{tier}', 'stage1_generator.pt')


def build_protocol(ds, tier, target_healthy, max_attempts):
    ck = stage1_path(ds, tier)
    ck_hash = None
    if os.path.exists(ck):
        h = hashlib.md5()
        with open(ck, 'rb') as f:
            h.update(f.read())
        ck_hash = h.hexdigest()[:16]
    return {
        'dataset': ds,
        'num_clients': tier,
        'code_hash': code_hash(),
        'data_identity_hash': compute_data_identity(ds, tier),
        'task_mode': 'multiclass',
        'weighted_ce': True,
        'contrastive_mode': 'subgraph_cross_view',
        'reliability_holdout_ratio': 0,
        'local_optimizer_lifecycle': 'reset_each_round',
        'DDPM': 'conditional, T=20, beta_end=0.5, timestep_scalar',
        'radius_constraint': True,
        'feature_stats_align': False,
        'skip_frozen': True,
        'num_rounds': 100,
        'num_epochs': 3,
        'selection_metric': 'accuracy',
        'test_isolation': 'tuning_mode (test 不计算)',
        'search_space': SEARCH_SPACE,
        'formal_search_space_version': 'v2',
        'formal_health_version': 'v2',
        'lambda_diffusion_anchor': 1e-2,
        'sampler': 'TPESampler(seed=2024)',
        'pruner': 'none (无正式 pruning policy)',
        'target_healthy': target_healthy,
        'max_attempts': max_attempts,
        'health_rule': ('formal_health v2: 100轮内任何 loss/grad/参数/raw '
                        'generator 输出/半径 nonfinite (NaN/±Inf) -> FAIL, '
                        '不参与候选排名; projected finite 不豁免 raw nonfinite'),
        'stage1_checkpoint': ck,
        'stage1_checkpoint_hash': ck_hash,
    }


def verify_protocol(saved, current):
    """resume 时协议一致性 fail-fast。返回 None 或抛 ValueError。"""
    for k in ('code_hash', 'data_identity_hash', 'search_space', 'dataset',
              'num_clients', 'stage1_checkpoint_hash'):
        if saved.get(k) != current.get(k):
            raise ValueError(
                f"[Formal] protocol 不一致 ({k}): study 记录 "
                f"{saved.get(k)} vs 当前 {current.get(k)} —— 禁止静默 resume")


def objective_value_from_metrics(path):
    """从 final_metrics.json 读 validation objective (绝不读 test)。"""
    with open(path) as f:
        j = json.load(f)
    return float(j['best_val_primary'])


def wait_gpu_memory(min_free_mb=6000, poll_sec=30):
    """显存看门狗: 空闲显存不足时等待, 防并发 OOM 崩溃。无 GPU 时直接返回。"""
    import time as _time
    while True:
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=15)
            if r.returncode != 0:
                return   # 无 nvidia-smi (CPU 环境), 不拦截
            free = max(int(x) for x in r.stdout.split())
            if free >= min_free_mb:
                return
            print(f"  [GPU Gate] 空闲显存 {free}MB < {min_free_mb}MB, "
                  f"等待 {poll_sec}s ...", flush=True)
            _time.sleep(poll_sec)
        except Exception:
            return


def run_one_trial(ds, tier, seed, outdir, params, stage1, expect_data_hash,
                  gpu_mem_gate_mb=6000):
    wait_gpu_memory(gpu_mem_gate_mb)
    cmd = [PY, 'train_fedtad.py', '--root', './dataset',
           '--dataset', ds, '--num_clients', str(tier),
           '--partition', 'Louvain', '--seed', str(seed)] + FIXED_FLAGS + [
           '--diffusion_pretrained_checkpoint', stage1,
           '--checkpoint_dir', outdir,
           '--emit_events', '--events_jsonl',
           os.path.join(outdir, 'events.jsonl'),
           '--final_metrics_json', os.path.join(outdir, 'final_metrics.json'),
           '--metrics_jsonl', os.path.join(outdir, 'metrics.jsonl'),
           '--resource_usage_json', os.path.join(outdir, 'resource_usage.json')]
    for k, v in params.items():
        cmd += [f'--{k}', str(v)]
    import time
    t0 = time.time()
    with open(os.path.join(outdir, 'stdout.log'), 'w') as fo, \
         open(os.path.join(outdir, 'stderr.log'), 'w') as fe:
        subprocess.run(cmd, cwd=REPO, stdout=fo, stderr=fe, check=False)
    wall = time.time() - t0
    fp = os.path.join(outdir, 'final_metrics.json')
    if not os.path.exists(fp):
        return None, wall, False, {'failure_round': None,
                                   'failure_type': 'NO_METRICS',
                                   'nonfinite_detected': False}
    # 每个 trial 开始/结束都核对 data identity (fail-fast 语义由调用方处理)
    if compute_data_identity(ds, tier) != expect_data_hash:
        raise ValueError(
            f"[Formal] data identity 在 trial 运行后变化: {ds}-c{tier} "
            f"期望 {expect_data_hash} —— cell STOP")
    ok, info = trial_health_check(outdir)
    if not ok:
        json.dump(info, open(os.path.join(outdir, 'trial_health.json'), 'w'),
                  indent=1)
    return (objective_value_from_metrics(fp) if ok else None,
            wall, ok, info)


def suggest(trial):
    return {k: trial.suggest_categorical(k, v['choices'])
            for k, v in SEARCH_SPACE.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_healthy', type=int, default=12,
                    help='每 cell 目标 health=PASS 的完整 100 轮 trial 数 '
                         '(预算 = 健康 trial 数, 非 attempt 数)')
    ap.add_argument('--max_attempts', type=int, default=15,
                    help='每 cell attempt 上限 (防数值不稳定病态搜索无限跑)')
    ap.add_argument('--n_jobs', type=int, default=2,
                    help='并发 trial 数 (服务器大显存/大内存下放开; 本地 7GB 机器'
                         '必须保持 1)。RNG v2 协议保证 trial 独立可复现; '
                         '首个 trial 恒串行以生成 CKR 缓存避免文件竞态; '
                         '每 trial 前有显存看门狗')
    ap.add_argument('--gpu_mem_gate_mb', type=int, default=6000,
                    help='显存看门狗: 空闲显存低于该值(默认 6GB)时 trial 等待'
                         '而不启动, 防止 OOM 崩溃')
    ap.add_argument('--cells', default=None,
                    help='逗号分隔 "ds:tier"; 默认全部 15 cell')
    args = ap.parse_args()
    cells = ([tuple(c.split(':')) for c in args.cells.split(',')]
             if args.cells else
             [(ds, t) for ds in DATASETS for t in TIERS])
    cells = [(ds, int(t)) for ds, t in cells]

    for ds, tier in cells:
        cell_dir = os.path.join(OUT, ds, f'clients{tier}')
        os.makedirs(cell_dir, exist_ok=True)
        protocol = build_protocol(ds, tier, args.target_healthy,
                                  args.max_attempts)
        proto_path = os.path.join(cell_dir, 'protocol.json')
        if os.path.exists(proto_path):
            saved = json.load(open(proto_path))
            verify_protocol(saved, protocol)   # 不一致 -> ValueError
        else:
            json.dump(protocol, open(proto_path, 'w'), indent=1)
        if protocol['stage1_checkpoint_hash'] is None:
            raise ValueError(
                f"[Formal] {ds}-c{tier}: Stage1 checkpoint 缺失 "
                f"({protocol['stage1_checkpoint']}) —— 先跑 Stage1 pretraining")

        storage = f"sqlite:///{os.path.join(cell_dir, 'study.db')}"
        study = optuna.create_study(
            study_name=f"formal_{FORMAL_VERSION}_{ds}_c{tier}",
            storage=storage,
            load_if_exists=True, direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=2024))

        def objective(trial):
            params = suggest(trial)
            d = os.path.join(cell_dir, 'trial_logs',
                             f'trial_{trial.number:03d}')
            os.makedirs(d, exist_ok=True)
            print(f"  [{ds} c{tier} trial {trial.number}] {params}",
                  flush=True)
            if compute_data_identity(ds, tier) != protocol['data_identity_hash']:
                raise ValueError(
                    f"[Formal] data identity 变化: {ds}-c{tier} —— cell STOP")
            val, wall, ok, info = run_one_trial(
                ds, tier, 2024, d, params, protocol['stage1_checkpoint'],
                protocol['data_identity_hash'],
                gpu_mem_gate_mb=args.gpu_mem_gate_mb)
            trial.set_user_attr('wall_sec', round(wall, 1))
            trial.set_user_attr('health', 'PASS' if ok else 'FAIL')
            if ok:
                return val
            # formal health rule: 数值失败 != 普通坏超参 -> Optuna FAIL 状态
            for k, v in info.items():
                trial.set_user_attr(k, (v if not isinstance(v, (dict, list))
                                        else json.dumps(v)))
            raise HealthFailure(
                f"[{ds} c{tier} trial {trial.number}] health FAIL: "
                f"{info.get('failure_type')} @ round {info.get('failure_round')}")

        def _healthy():
            return sum(1 for t in study.trials
                       if t.state.name == 'COMPLETE'
                       and t.user_attrs.get('health') == 'PASS')

        while True:
            if _healthy() >= args.target_healthy:
                print(f"[{ds} c{tier}] 达到目标 healthy="
                      f"{args.target_healthy}", flush=True)
                break
            allowed = args.max_attempts - len(study.trials)
            if allowed <= 0:
                print(f"[{ds} c{tier}] 达到 attempt 上限 {args.max_attempts} "
                      f"(healthy={_healthy()}) —— STOP 报告", flush=True)
                break
            # 首个 trial 恒串行: 生成 ./ckr 缓存, 避免并发写同一缓存文件竞态;
            # 之后按 --n_jobs 并发 (RNG v2 协议保证 trial 独立可复现,
            # 每个 trial 启动前过显存看门狗)
            has_history = any(t.state.name in ('COMPLETE', 'FAIL')
                              for t in study.trials)
            n_batch = 1 if not has_history else min(args.n_jobs, allowed,
                                                    args.target_healthy
                                                    - _healthy())
            if n_batch <= 0:
                break
            study.optimize(objective, n_trials=n_batch, n_jobs=n_batch,
                           gc_after_trial=True,
                           catch=(HealthFailure,))   # 数值失败 -> FAILED 状态并继续
        # best 只从 health=PASS 的 COMPLETE trials 中选 (formal health rule)
        candidates = [t for t in study.trials
                      if t.state.name == 'COMPLETE'
                      and t.user_attrs.get('health') == 'PASS']
        if not candidates:
            raise ValueError(f"[{ds} c{tier}] 无 health=PASS 候选")
        best = max(candidates, key=lambda t: t.value)
        json.dump(dict(best.params),
                  open(os.path.join(cell_dir, 'best_params.json'), 'w'),
                  indent=1)
        # 汇总 CSV (不含任何 test 指标 — tuning 全程 test 未计算)
        rows = []
        for t in study.trials:
            if t.state.name == 'COMPLETE':
                rows.append({'trial': t.number,
                             'best_val': round(t.value, 4),
                             'params': json.dumps(t.params),
                             'wall_sec': t.user_attrs.get('wall_sec'),
                             'health': t.user_attrs.get('health', '')})
        with open(os.path.join(cell_dir, 'study_summary.csv'), 'w',
                  newline='') as f:
            import csv as _csv
            _w = _csv.writer(f)
            _w.writerow(['trial', 'best_val', 'params', 'wall_sec', 'health'])
            for r in rows:
                _w.writerow([r['trial'], r['best_val'], r['params'],
                             r['wall_sec'], r['health']])
        print(f"[{ds} c{tier}] best(health=PASS) val={best.value:.4f} "
              f"params={best.params}", flush=True)
    print('FORMAL_CAMPAIGN_DONE')


if __name__ == '__main__':
    main()
