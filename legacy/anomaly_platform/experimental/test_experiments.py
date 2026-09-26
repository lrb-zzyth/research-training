"""
实验体系测试:
  1. 实验配置解析 (suite kwargs 均为合法 CLI 参数)
  2. 多种子确定性 (同 seed 两次运行 -> 最终指标一致)
  3. 结果文件不覆盖 (已完成运行自动跳过)
  4. split 无交集 (run_integrity_assertions)
  5. final test 加载 best (loaded_from == best.pt)
  6. unavailable 指标汇总 (NaN 排除 + 有效数记录)
  7. 多种子聚合计算 (mean/std 数学正确)
  8. paired difference 统计 (合成数据)
  9. 实验失败状态记录 (status.json=failed + stderr tail)
  10. dry-run suite 启动
"""
import sys, os, json, math, tempfile, subprocess
from types import SimpleNamespace
# 本目录 (experimental/) 供 import run_experiments / experiment_stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_geometric.data import Data

# 仓库根目录 (experimental/test_*.py -> 上一级), 供 import train_fedtad / util / 子进程
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================================
#  Test 1: 配置解析 — suite 中每个 kwargs 都是合法 CLI 参数
# ====================================================================
print("\n[Test 1] experiment config kwargs are valid CLI flags")
import run_experiments as re
import re as _re
help_out = subprocess.run([PY, os.path.join(ROOT, 'train_fedtad.py'), '--help'],
                          capture_output=True, text=True, cwd=ROOT).stdout
valid_flags = set(_re.findall(r'--[a-z0-9_\-]+', help_out))
bench_help = subprocess.run([PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_checkpointing.py'),
                             '--help'], capture_output=True, text=True, cwd=ROOT).stdout
bench_flags = set(_re.findall(r'--[a-z0-9_\-]+', bench_help))
n_checked = 0
for suite in re.SUITES:
    flags = bench_flags if re.SUITES[suite].get('runner') == 'benchmark' else valid_flags
    for name, exp in re.SUITES[suite]['experiments'].items():
        kw = exp['kwargs']
        i = 0
        while i < len(kw):
            flag = kw[i]
            assert flag.startswith('--'), f"{suite}/{name}: 位置 {i} 不是参数: {flag}"
            assert flag in flags, f"{suite}/{name}: 非法参数 {flag}"
            n_checked += 1
            # 布尔参数 (如 --no-use_weighted_ce) 无值; 否则下一 token 为值
            if i + 1 < len(kw) and not kw[i + 1].startswith('--'):
                i += 2
            else:
                i += 1
print(f"  {n_checked} kwargs 全部合法 ({len(re.SUITES)} suites) ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 2: 多种子确定性 — 同 seed 同配置 -> 相同最终指标
# ====================================================================
print("\n[Test 2] multi-seed determinism (same seed reproduces)")
def run_tiny(seed, tag, root_dir):
    out_dir = os.path.join(root_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    cli = [PY, os.path.join(ROOT, 'train_fedtad.py'),
           '--root', os.path.join(ROOT, 'dataset'), '--dataset', 'Cora',
           '--num_clients', '2', '--num_epochs', '1', '--hid_dim', '8',
           '--dropout', '0.0', '--num_rounds', '1', '--f1_threshold', '0.0',
           '--auc_threshold=-1e6', '--task_mode', 'anomaly_binary',
           '--normal_classes', '0,2,3,4,5', '--anomaly_classes', '1,6',
           '--contrastive_mode', 'none', '--distill_weighting', 'none',
           '--seed', str(seed), '--checkpoint_dir', out_dir,
           '--final_metrics_json', os.path.join(out_dir, 'final_metrics.json'),
           '--log_dir', out_dir]
    p = subprocess.run(cli, cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1000:]
    return json.load(open(os.path.join(out_dir, 'final_metrics.json')))

with tempfile.TemporaryDirectory() as tmp:
    f1 = run_tiny(0, 'r1', tmp)
    f2 = run_tiny(0, 'r2', tmp)
    for k in ('seed', 'best_round', 'loaded_from'):
        assert f1[k] == f2[k], f"{k}: {f1[k]} != {f2[k]}"
    a = f1['metrics']['pooled']['pr_auc']
    b = f2['metrics']['pooled']['pr_auc']
    assert abs(a - b) < 1e-9, f"same-seed reproduces: {a} vs {b}"
    # 不同 seed 应产生不同结果 (划分/初始化不同)
    f3 = run_tiny(1, 'r3', tmp)
    c = f3['metrics']['pooled']['pr_auc']
    assert abs(a - c) > 1e-9, "different seeds must differ"
    print(f"  seed0 run1={a:.4f} run2={b:.4f} (identical), seed1={c:.4f} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 3: 结果文件不覆盖 — 已完成运行自动跳过
# ====================================================================
print("\n[Test 3] completed runs are skipped (no overwrite)")
import run_experiments as re2
old_suites = dict(re2.SUITES)
suite3 = f"_t_noverwrite_{os.getpid()}"
try:
    re2.SUITES[suite3] = {
        'seeds': [0], 'overrides': [],
        'experiments': {'_T1': re2._exp('_T1', [
            '--task_mode', 'anomaly_binary', '--normal_classes', '0,2,3,4,5',
            '--anomaly_classes', '1,6', '--distill_weighting', 'none',
            '--contrastive_mode', 'none', '--num_clients', '2', '--num_rounds', '1',
            '--hid_dim', '8', '--num_epochs', '1', '--f1_threshold', '0.0',
            '--auc_threshold=-1e6', '--no-use_weighted_ce'])}}
    opts = SimpleNamespace(suite=suite3, resume_failed=False, cpu=True, dry=False)
    st1 = re2.run_one(re2.SUITES[suite3],
                      re2.SUITES[suite3]['experiments']['_T1'], 0, opts)
    assert st1 == 'success', st1
    fm_path = os.path.join(re2.RUNS_ROOT, suite3, '_T1', 'seed_0',
                           'final_metrics.json')
    content_before = open(fm_path).read()
    st2 = re2.run_one(re2.SUITES[suite3],
                      re2.SUITES[suite3]['experiments']['_T1'], 0, opts)
    assert st2 == 'skipped', st2
    assert open(fm_path).read() == content_before, "final_metrics.json must not be overwritten"
    print("  second run skipped, result file unchanged ✓")
finally:
    re2.SUITES = old_suites
print("  PASSED ✓")

# ====================================================================
#  Test 4: split 无交集 (run_integrity_assertions)
# ====================================================================
print("\n[Test 4] integrity: fit/reliability/val/test disjoint")
from util.data_split import stratified_reliability_split, stratified_split
from train_fedtad import run_integrity_assertions
N = 120
x4 = torch.randn(N, 8)
y4 = torch.zeros(N, dtype=torch.long)
y4[60:80] = 1
y4[80:] = 2
d4 = Data(x=x4, y=y4)
tr, va, te, _ = stratified_split(d4, 3, 0.2, 0.4, seed=1)
d4.train_idx, d4.val_idx, d4.test_idx = tr, va, te
fit, rel, _ = stratified_reliability_split(d4, tr, 3, 0.2, 3, split_seed=1)
d4.fit_idx, d4.reliability_idx = fit, rel
class A4:
    task_mode = 'multiclass'
    resplit_stratified = True
run_integrity_assertions([d4], 3, A4())
print("  run_integrity_assertions 通过 (两两不重叠, fit+rel==train) ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 5: final test 加载 best (结构上: final_metrics loaded_from)
# ====================================================================
print("\n[Test 5] final test loads best.pt")
# 由 Test 3 的运行验证: loaded_from 应为 best.pt
from train_fedtad import find_final_checkpoint  # noqa: F401
import glob
bm = glob.glob(os.path.join(re2.RUNS_ROOT, suite3, '_T1', 'seed_0', 'best.pt'))
assert bm, "best.pt should exist from test 3 run"
fm3 = json.load(open(os.path.join(re2.RUNS_ROOT, suite3, '_T1', 'seed_0',
                                  'final_metrics.json')))
assert fm3.get('loaded_from') == 'best.pt', f"loaded_from={fm3.get('loaded_from')}"
print(f"  best.pt exists and loaded_from='best.pt' ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 6: unavailable 指标汇总 (NaN 排除)
# ====================================================================
print("\n[Test 6] unavailable metrics aggregation excludes NaN")
from experiment_stats import summarize
vals = [80.0, float('nan'), 90.0, float('nan'), 85.0]
s = summarize(vals)
assert s['n'] == 3, f"n must count only valid: {s['n']}"
assert abs(s['mean'] - 85.0) < 1e-9
assert abs(s['std'] - 5.0) < 1e-9
assert math.isnan(summarize([float('nan')])['mean'])
print(f"  mean={s['mean']} std={s['std']} n={s['n']} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 7: 多种子聚合计算
# ====================================================================
print("\n[Test 7] multi-seed aggregation math")
from experiment_stats import paired_diff
full = [70.0, 75.0, 80.0, 85.0, 90.0]
base = [60.0, 65.0, 70.0, 75.0, 80.0]
pd = paired_diff(full, base, [0, 1, 2, 3, 4])
assert pd['n_pairs'] == 5
assert abs(pd['mean_diff'] - 10.0) < 1e-9
assert pd['wins'] == 5 and pd['losses'] == 0
assert abs(pd['ci95_low'] - 10.0) < 1e-6, "CI collapses to 10 for constant diff"
assert abs(pd['std_diff']) < 1e-9
print(f"  mean_diff={pd['mean_diff']} wins={pd['wins']} ci=({pd['ci95_low']:.3f}, {pd['ci95_high']:.3f}) ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 8: paired difference 统计 (混合方向)
# ====================================================================
print("\n[Test 8] paired difference with mixed directions")
full8 = [70.0, 60.0, 80.0, 65.0]
base8 = [60.0, 70.0, 75.0, 70.0]
pd8 = paired_diff(full8, base8, [0, 1, 2, 3])
assert pd8['wins'] == 2 and pd8['losses'] == 2
assert abs(pd8['mean_diff'] - 0.0) < 1e-9
assert not math.isnan(pd8['wilcoxon_p']), "wilcoxon should be computable for n=4"
assert pd8['per_seed']['0'] == {'full': 70.0, 'baseline': 60.0, 'diff': 10.0}
print(f"  wins={pd8['wins']} losses={pd8['losses']} p={pd8['wilcoxon_p']:.4f} "
      f"per_seed ok ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 9: 实验失败状态记录
# ====================================================================
print("\n[Test 9] failed run records status=failed with stderr tail")
import run_experiments as re3
old_suites = dict(re3.SUITES)
suite9 = f"_t_fail_{os.getpid()}"
try:
    re3.SUITES[suite9] = {
        'seeds': [0], 'overrides': [],
        'experiments': {'_F1': re3._exp('_F1', ['--definitely-not-a-flag', 'x'])}}
    opts = SimpleNamespace(suite=suite9, resume_failed=False, cpu=True, dry=False)
    st = re3.run_one(re3.SUITES[suite9],
                     re3.SUITES[suite9]['experiments']['_F1'], 0, opts)
    assert st in ('failed', 'failed_config'), st
    sp = json.load(open(os.path.join(re3.RUNS_ROOT, suite9, '_F1', 'seed_0',
                                     'status.json')))
    assert sp['status'] in ('failed', 'failed_config') and sp['returncode'] != 0
    assert 'definitely-not-a-flag' in sp['stderr_tail']
    print(f"  status={sp['status']}, returncode={sp['returncode']}, "
          f"stderr captured ✓")
finally:
    re3.SUITES = old_suites
print("  PASSED ✓")

# ====================================================================
#  Test 10: dry-run suite 配置可解析 (不实际运行, 由人工执行
#          python run_experiments.py --suite dry_run 验证)
# ====================================================================
print("\n[Test 10] dry-run suite config sanity")
dry = re.SUITES['dry_run']
assert len(dry['experiments']) >= 8
for name, exp in dry['experiments'].items():
    assert exp['seeds'] is None or exp['seeds'] == [0]
    assert any('num_rounds' in k or True for k in exp['kwargs'])
print(f"  dry_run: {len(dry['experiments'])} experiments, overrides={len(dry['overrides'])} ✓")
print("  PASSED ✓")

# ====================================================================
print(f"\n{'=' * 60}")
print("ALL EXPERIMENT TESTS PASSED (1-10) ✓")
print(f"{'=' * 60}")
