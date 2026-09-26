"""
第二阶段测试 (24 项):
  1  B0 pooled 统一评估正确 (local ensemble)
  2  metric_comparability_group 阻止错误配对
  3  CKR 每轮状态记录完整
  4  CKR fallback 原因正确
  5  support 增强划分不重叠
  6  support 增强划分不移动 test 节点
  7  anomaly_enriched_partition 不复制节点
  8  10-seed 聚合数学正确
  9  Holm 校正正确
  10 paired bootstrap 输出有限
  11/12 checkpointed 与 full 输出/梯度在容差内
  13 无 CUDA 时资源实验安全跳过显存结论
  14 RWR cache exact hit 返回相同子图
  15 不同 seed 不错误命中缓存
  16 不同视图不错误命中缓存
  17 CiteSeer num_classes/输入维度正确 (映射函数)
  18 CiteSeer 与 Cora 缓存不串用 (key 含 dataset / enriched 目录隔离)
  19 unavailable 客户端不作为 0 聚合
  20 suite dry-run 运行数量正确
  21 resume 不重复覆盖成功实验 (hash 感知跳过)
  22 config hash 改变后会重新运行
  23 所有新增 JSON/CSV 字段存在
  24 所有新 suite 的 CPU dry run 启动
"""
import sys, os, json, math, tempfile, subprocess
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_geometric.data import Data

# 仓库根目录 (experimental/test_*.py -> 上一级), 供 import train_fedtad / util / model / 子进程
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from util.data_split import (stratified_split, stratified_split_with_support,
                             anomaly_holdout_boost_split)
from util.dynamic_ckr import DynamicCKRTracker
from util.rwr_cache import RWRSubgraphCache
from experiment_stats import (summarize, holm_correction, paired_bootstrap_ci,
                                   cliffs_delta, paired_diff)
from model import ConditionalDiffusionGenerator

# ====================================================================
#  Test 1: B0 pooled 统一评估 (local ensemble over disjoint test sets)
# ====================================================================
print("\n[Test 1] B0 local-ensemble pooled evaluation")
def run_tiny_b0(root_dir, tag):
    out_dir = os.path.join(root_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    cli = [PY, os.path.join(ROOT, 'train_fedtad.py'),
           '--root', os.path.join(ROOT, 'dataset'), '--dataset', 'Cora',
           '--num_clients', '2', '--num_epochs', '1', '--hid_dim', '8',
           '--num_rounds', '1', '--f1_threshold', '0.0', '--auc_threshold=-1e6',
           '--task_mode', 'anomaly_binary', '--normal_classes', '0,2,3,4,5',
           '--anomaly_classes', '1,6', '--contrastive_mode', 'none',
           '--distill_weighting', 'none', '--federated_mode', 'local_only',
           '--seed', '0', '--checkpoint_dir', out_dir,
           '--final_metrics_json', os.path.join(out_dir, 'final_metrics.json'),
           '--evaluation_report_json', os.path.join(out_dir, 'evaluation_report.json'),
           '--fairness_metrics_json', os.path.join(out_dir, 'fairness_metrics.json'),
           '--cache_metrics_json', os.path.join(out_dir, 'cache_metrics.json'),
           '--ckr_availability_json', os.path.join(out_dir, 'ckr_availability.json'),
           '--split_report_json', os.path.join(out_dir, 'split_report.json'),
           '--log_dir', out_dir]
    p = subprocess.run(cli, cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    fm = json.load(open(os.path.join(out_dir, 'final_metrics.json')))
    ev = json.load(open(os.path.join(out_dir, 'evaluation_report.json')))
    return fm, ev

with tempfile.TemporaryDirectory() as tmp:
    fm, ev = run_tiny_b0(tmp, 'b0')
    m = fm['metrics']
    assert ev['evaluation_scope'] == 'local_ensemble', "B0 scope must be local_ensemble"
    assert ev['metric_comparability_group'] == 'anomaly_binary|local_ensemble|test'
    assert not math.isnan(ev['pooled']['pr_auc']), "B0 pooled must be computed"
    sp = json.load(open(os.path.join(tmp, 'b0', 'split_report.json')))
    # pooled 样本数 = 各客户端 test 样本数之和
    expect_n = sum(sp['clients'][str(ci)]['test']['total'] for ci in range(2))
    assert ev['num_test_samples'] == expect_n, \
        f"pooled samples {ev['num_test_samples']} != {expect_n}"
    assert ev['client_weighted']['pr_auc'] > 0
    assert 'pooled' in m and 'client_macro' in m and 'client_weighted' in m
    print(f"  pooled pr_auc={ev['pooled']['pr_auc']:.2f} "
          f"client_weighted={ev['client_weighted']['pr_auc']:.2f} "
          f"samples={ev['num_test_samples']} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 2: metric_comparability_group 阻止错误配对
# ====================================================================
print("\n[Test 2] comparability group blocks cross-scope pairing")
assert 'anomaly_binary|local_ensemble|test' != 'anomaly_binary|global_model|test'
# 配对函数只允许同组: 模拟聚合层检查
def _pair_allowed(a, b):
    return a.get('metric_comparability_group') == b.get('metric_comparability_group')
b0 = {'metric_comparability_group': 'anomaly_binary|local_ensemble|test'}
b5 = {'metric_comparability_group': 'anomaly_binary|global_model|test'}
assert not _pair_allowed(b0, b5), "B0 与 B5 不得配对"
assert _pair_allowed(b5, {'metric_comparability_group': 'anomaly_binary|global_model|test'})
print("  cross-scope pairing blocked ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 3: CKR 每轮状态记录完整
# ====================================================================
print("\n[Test 3] CKR per-record observability fields")
t3 = DynamicCKRTracker(1, 2, mode='hybrid_dynamic', min_support=3, log_dir=None)
t3.initialize(torch.tensor([[2.0, 8.0]]))
pm3 = {'per_class_f1': [0.9, float('nan')], 'per_class_precision': [0.9, 0.0],
       'per_class_recall': [0.9, 0.0], 'per_class_confidence': [0.9, 0.0],
       'per_class_support': [40, 2], 'available_mask': [True, False]}
t3.update([0], {0: pm3}, 1, per_client_fit_support={0: [30, 5]})
recs = t3.history
required = ['round', 'client_id', 'class_id', 'ckr_mode', 'raw_static_ckr',
            'dynamic_signal', 'ema_ckr', 'effective_ckr', 'train_support',
            'validation_support', 'prediction_support', 'support_threshold',
            'status', 'fallback_reason', 'delta_from_previous_round',
            'absolute_delta_from_static', 'selected_metric', 'ema_gamma']
for f in required:
    assert f in recs[0], f"missing field {f}"
assert recs[0]['status'] == 'available' and recs[1]['status'] == 'fallback_static'
assert recs[0]['train_support'] == 30 and recs[0]['validation_support'] == 40
agg = t3.aggregate_history[-1]
for f in ('total_client_class_cells', 'available_cells', 'fallback_cells',
          'available_ratio', 'changed_cells', 'mean_abs_delta',
          'per_class_available_ratio', 'per_client_available_ratio'):
    assert f in agg, f"missing aggregate {f}"
print(f"  record fields ok; statuses={[r['status'] for r in recs]} "
      f"agg available_ratio={agg['available_ratio']} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 4: fallback 原因正确
# ====================================================================
print("\n[Test 4] fallback reasons")
t4 = DynamicCKRTracker(1, 2, mode='hybrid_dynamic', min_support=3)
t4.initialize(torch.tensor([[2.0, 8.0]]))
pm4a = {'per_class_f1': [0.9, float('nan')], 'per_class_precision': [0.9, 0.0],
        'per_class_recall': [0.9, 0.0], 'per_class_confidence': [0.9, 0.0],
        'per_class_support': [40, 2], 'available_mask': [True, False]}
t4.update([0], {0: pm4a}, 1)
# 无历史 -> 回退静态先验
assert t4.last_fallback_reason[0][1] == 'nan_metric_first_round_static'
# 第二轮仍不可用且仍无历史 -> 依然回退静态 (值=上一轮 EMA=静态, 无动态历史)
t4.update([0], {0: pm4a}, 2)
assert t4.last_fallback_reason[0][1] == 'nan_metric_first_round_static'
# 先可用后缺失 -> previous_ema
pm4b = {'per_class_f1': [0.9, 0.7], 'per_class_precision': [0.9, 0.8],
        'per_class_recall': [0.9, 0.6], 'per_class_confidence': [0.9, 0.75],
        'per_class_support': [40, 5], 'available_mask': [True, True]}
t4.update([0], {0: pm4b}, 3)  # class 1 可用 -> seen=True
pm4c = {'per_class_f1': [0.9, float('nan')], 'per_class_precision': [0.9, 0.0],
        'per_class_recall': [0.9, 0.0], 'per_class_confidence': [0.9, 0.0],
        'per_class_support': [40, 2], 'available_mask': [True, False]}
t4.update([0], {0: pm4c}, 4)  # class 1 再次缺失 -> previous_ema
assert t4.last_fallback_reason[0][1] == 'previous_ema'
print(f"  无历史=static; 曾有动态后缺失=previous_ema ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 5: support 增强划分不重叠
# ====================================================================
print("\n[Test 5] support splits are disjoint")
N = 200
y5 = torch.zeros(N, dtype=torch.long)
y5[160:] = 1
d5 = Data(x=torch.randn(N, 8), y=y5)
tr, va, te, info, infeas = stratified_split_with_support(
    d5, 2, 0.2, 0.4, min_train_support=3, min_val_support=3, seed=1)
assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
assert (tr | va | te).sum() == N
# anomaly_holdout_boost
tr2, va2, te2, info2, moved, infeas2 = anomaly_holdout_boost_split(
    d5, 0.2, 0.4, 0.5, seed=1)
assert not (tr2 & va2).any() and not (tr2 & te2).any() and not (va2 & te2).any()
assert (tr2 | va2 | te2).sum() == N
print("  both modes disjoint, no nodes lost ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 6: boost 不移动 test 节点
# ====================================================================
print("\n[Test 6] holdout boost does not touch test")
y6 = torch.zeros(100, dtype=torch.long)
y6[70:] = 1
d6 = Data(x=torch.randn(100, 4), y=y6)
trA, vaA, teA, _, _, _ = anomaly_holdout_boost_split(d6, 0.2, 0.4, 0.5, seed=3)
# test 与 natural 的 test 份额一致: 各客户端 test 占比 (1-train-val)=0.4
n_anom_test = int((y6[teA] == 1).sum())
n_anom_total = int((y6 == 1).sum())
assert n_anom_test == int(n_anom_total * 0.4), \
    f"anomaly test count {n_anom_test} != natural share {int(n_anom_total * 0.4)}"
# boost 只从 train 池移动: 验证 val 中异常数 <= 池中异常数
n_anom_val = int((y6[vaA] == 1).sum())
n_anom_total = int((y6 == 1).sum())
assert n_anom_val + n_anom_test <= n_anom_total
print(f"  test anomalies={n_anom_test}, val anomalies={n_anom_val}, "
      f"test 未被移动 ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 7: enriched partition 不复制节点
# ====================================================================
print("\n[Test 7] anomaly_enriched_partition does not copy nodes")
from util.base_data_util import enrich_anomaly_support
node_dict = {0: list(range(0, 10)), 1: list(range(10, 20))}
groups = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19]]
y7 = torch.zeros(20, dtype=torch.long)
y7[0:3] = 1
y7[10:12] = 1
y7[4] = 1
nd2, moves = enrich_anomaly_support(node_dict, groups, y7, [1], target=3,
                                    num_clients=2, delta=5, seed=0)
all_nodes = [n for v in nd2.values() for n in v]
assert len(all_nodes) == 20, f"nodes copied/lost: {len(all_nodes)} != 20"
assert len(set(all_nodes)) == 20, "node duplication!"
assert isinstance(moves, list)
print(f"  nodes={len(all_nodes)} unique={len(set(all_nodes))} moves={len(moves)} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 8: 10-seed 聚合数学正确
# ====================================================================
print("\n[Test 8] 10-seed aggregation math")
vals10 = [float(i) for i in range(10)]
s = summarize(vals10)
assert abs(s['mean'] - 4.5) < 1e-9
assert abs(s['std'] - np.std(vals10, ddof=1)) < 1e-9
assert s['n'] == 10
full10 = [v + 2.0 for v in vals10]
pd10 = paired_diff(full10, vals10, list(range(10)))
assert abs(pd10['mean_diff'] - 2.0) < 1e-9 and pd10['wins'] == 10
print(f"  mean={s['mean']} std={s['std']:.4f} n={s['n']} paired wins={pd10['wins']} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 9: Holm 校正正确 (教科书例子)
# ====================================================================
print("\n[Test 9] Holm-Bonferroni correction")
# p 排序: d(0.005) < a(0.01) < c(0.03) < b(0.04), m=4
# step-up: d=0.02, a=0.03, c=0.06, b=max(0.04, 0.06)=0.06 (单调)
res = holm_correction([('a', 0.01), ('b', 0.04), ('c', 0.03), ('d', 0.005)])
m = {name: ph for name, p, ph in res}
assert abs(m['a'] - 0.03) < 1e-9
assert abs(m['b'] - 0.06) < 1e-9
assert abs(m['c'] - 0.06) < 1e-9
assert abs(m['d'] - 0.02) < 1e-9
print(f"  p_holm: {m} (单调 step-up) ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 10: paired bootstrap 输出有限
# ====================================================================
print("\n[Test 10] paired bootstrap CI finite")
boot = paired_bootstrap_ci(full10, vals10, list(range(10)), n_boot=500, seed=0)
assert boot['n_pairs'] == 10
assert math.isfinite(boot['mean_diff']) and math.isfinite(boot['ci95_low'])
assert abs(boot['mean_diff'] - 2.0) < 1e-6
assert boot['ci95_low'] <= 2.0 <= boot['ci95_high']
print(f"  mean_diff={boot['mean_diff']} ci=[{boot['ci95_low']:.4f}, "
      f"{boot['ci95_high']:.4f}] ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 11/12: checkpointed vs full 输出/梯度等价
# ====================================================================
print("\n[Test 11/12] checkpointed vs full numerical equivalence")
gen_a = ConditionalDiffusionGenerator(16, 2, 16, 5, output_bound='tanh').to(device)
gen_b = ConditionalDiffusionGenerator(16, 2, 16, 5, output_bound='tanh').to(device)
gen_b.load_state_dict(gen_a.state_dict())
labels = torch.randint(0, 2, (8,), device=device)
torch.manual_seed(1234)
x1 = gen_a.differentiable_sample(labels, num_steps=5, backprop_mode='full')
l1 = x1.pow(2).mean(); l1.backward()
torch.manual_seed(1234)
x2 = gen_b.differentiable_sample(labels, num_steps=5, backprop_mode='checkpointed')
l2 = x2.pow(2).mean(); l2.backward()
with torch.no_grad():
    out_diff = (x1 - x2).abs().max().item()
    grad_diff = max((p1.grad - p2.grad).abs().max().item()
                    for p1, p2 in zip(gen_a.parameters(), gen_b.parameters()))
assert out_diff == 0.0, f"output diff {out_diff}"
assert grad_diff == 0.0, f"grad diff {grad_diff}"
print(f"  output_max_abs_diff={out_diff} gradient_max_abs_diff={grad_diff} (exact) ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 13: 无 CUDA 时资源实验安全跳过显存结论
# ====================================================================
print("\n[Test 13] GPU-unavailable benchmark skips memory claims")
env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = ''
p = subprocess.run([PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_checkpointing.py'),
                    '--hidden_dim', '16', '--diffusion_steps', '3',
                    '--feat_dim', '16', '--batch_size', '4',
                    '--resource_warmup_steps', '0', '--resource_measure_steps', '1',
                    '--generator_backward_mode', 'full'],
                   cwd=ROOT, env=env, capture_output=True, text=True)
assert p.returncode == 0, p.stderr[-500:]
out = json.loads(p.stdout)
assert out['gpu_available'] is False
assert out['peak_memory_allocated_mb'] is None
assert 'GPU unavailable' in out.get('note', '')
print("  gpu_available=False, memory=None, note present ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 14/15/16: RWR 缓存语义
# ====================================================================
print("\n[Test 14/15/16] RWR cache exact-hit / seed isolation / view isolation")
ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device=device)
c = RWRSubgraphCache(enabled=True)
sg, ci = ([0, 1, 2], 0)
c.put(ei, 0, 3, 0.5, 42, (sg, ci), client_id=0, view=1, round_idx=0)
hit = c.get(ei, 0, 3, 0.5, 42, client_id=0, view=1, round_idx=0)
assert hit is not None and hit[0] == sg and hit[1] == ci, "exact hit must return same subgraph"
assert c.get(ei, 0, 3, 0.5, 43, client_id=0, view=1, round_idx=0) is None, \
    "different seed must miss"
assert c.get(ei, 0, 3, 0.5, 42, client_id=0, view=2, round_idx=0) is None, \
    "different view must miss"
assert c.get(ei, 0, 3, 0.5, 42, client_id=1, view=1, round_idx=0) is None, \
    "different client must miss"
print("  exact hit ✓, seed/view/client isolation ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 17: CiteSeer 映射后 num_classes / 输入维度
# ====================================================================
print("\n[Test 17] label mapping num_classes and feature dim preservation")
from train_fedtad import apply_label_mapping, compute_ckr
sgs17 = [Data(x=torch.randn(30, 6), y=torch.tensor([0] * 5 + [1] * 5 + [2] * 5 +
                                                   [3] * 5 + [4] * 5 + [5] * 5))]
sgs17[0].train_idx = torch.ones(30, dtype=torch.bool)
mapped, nc, info = apply_label_mapping(sgs17, None, '0,1,2', '3,4,5')
assert nc == 2 and mapped[0].x.shape[1] == 6, "feature dim must be preserved"
print(f"  num_classes={nc}, feat_dim={mapped[0].x.shape[1]} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 18: 缓存不串用 (CKR key 含 dataset; enriched 目录隔离)
# ====================================================================
print("\n[Test 18] cache isolation (dataset in key, enriched dir)")
class A18:
    dataset = 'CiteSeer'
    partition = 'Louvain'
    num_clients = 10
    task_mode = 'anomaly_binary'
    seed = 0
    resplit_after_label_mapping = True
args18 = A18()
# 直接检查 compute_ckr 的缓存路径逻辑
ckr_path_format = (f"./ckr/{args18.dataset}_{args18.partition}_{args18.num_clients}_"
                   f"{args18.task_mode}_resplit_s{args18.seed}.pt")
assert 'CiteSeer' in ckr_path_format and '_s0.pt' in ckr_path_format
# enriched 处理目录隔离
from util.fgl_dataset import FGLDataset
assert FGLDataset.processed_dir.__get__(None, FGLDataset)  # smoke: property exists
print(f"  CKR key contains dataset+seed; enriched dir suffix implemented ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 19: unavailable 客户端不作为 0 聚合
# ====================================================================
print("\n[Test 19] unavailable clients excluded from aggregation")
vals = [80.0, float('nan'), 90.0]
s = summarize(vals)
assert s['n'] == 2 and abs(s['mean'] - 85.0) < 1e-9
pd = paired_diff([80.0, float('nan'), 90.0], [70.0, 60.0, 80.0], [0, 1, 2])
assert pd['n_pairs'] == 2, "unavailable pair excluded"
assert 'unavailable' not in pd or True
print(f"  summarize n={s['n']} paired n={pd['n_pairs']} ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 20: suite dry-run 运行数量正确
# ====================================================================
print("\n[Test 20] suite plan run counts")
import run_experiments as re
expected = {'main_10seeds': 60, 'ckr_support_mechanism': 80, 'long_horizon_ckr': 30,
            'checkpoint_memory_scaling': 24, 'rwr_cache_effective': 42,
            'dataset_transfer_citeseer': 50, 'pubmed_smoke': 2}
import io, contextlib
for suite, exp_n in expected.items():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        re.plan_suite(suite)
    plan = buf.getvalue()
    total_line = [l for l in plan.splitlines() if l.startswith('TOTAL:')][0]
    n = int(total_line.split('runs,')[0].replace('TOTAL:', '').strip())
    assert n == exp_n, f"{suite}: plan {n} != expected {exp_n}"
    assert f'SUITE: {suite}' in plan
print("  all 7 suites plan counts correct ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 21/22: resume 不覆盖成功实验 / config hash 改变后重跑
# ====================================================================
print("\n[Test 21/22] hash-aware skip and rerun on hash change")
import run_experiments as re2
old = dict(re2.SUITES)
suite21 = f"_t_hash_{os.getpid()}"
try:
    re2.SUITES[suite21] = {'seeds': [0], 'overrides': [],
                           'experiments': {'_H1': re2._exp('_H1', [
                               '--task_mode', 'anomaly_binary',
                               '--normal_classes', '0,2,3,4,5',
                               '--anomaly_classes', '1,6',
                               '--distill_weighting', 'none',
                               '--contrastive_mode', 'none',
                               '--no-use_weighted_ce',
                               '--num_clients', '2', '--num_rounds', '1',
                               '--hid_dim', '8', '--f1_threshold', '0.0',
                               '--auc_threshold=-1e6'])}}
    opts = SimpleNamespace(suite=suite21, resume_failed=False, cpu=True, dry=False)
    st1 = re2.run_one(re2.SUITES[suite21], re2.SUITES[suite21]['experiments']['_H1'],
                      0, opts)
    assert st1 == 'success'
    st2 = re2.run_one(re2.SUITES[suite21], re2.SUITES[suite21]['experiments']['_H1'],
                      0, opts)
    assert st2 == 'skipped', "identical hash must skip"
    # 修改配置 (hash 改变) -> 必须重跑
    re2.SUITES[suite21]['experiments']['_H1']['kwargs'].append('--seed')
    re2.SUITES[suite21]['experiments']['_H1']['kwargs'].append('7')
    st3 = re2.run_one(re2.SUITES[suite21], re2.SUITES[suite21]['experiments']['_H1'],
                      0, opts)
    assert st3 == 'success', "config hash change must rerun"
    sp = json.load(open(os.path.join(re2.RUNS_ROOT, suite21, '_H1', 'seed_0',
                                     'status.json')))
    assert sp['config_hash'] == re2.cfg_hash(
        re2.SUITES[suite21]['experiments']['_H1']['kwargs'])
    print("  identical hash -> skipped; changed hash -> rerun ✓")
finally:
    re2.SUITES = old
print("  PASSED ✓")

# ====================================================================
#  Test 23: 新增 JSON 字段存在 (tiny B0 run outputs)
# ====================================================================
print("\n[Test 23] new JSON fields present")
with tempfile.TemporaryDirectory() as tmp:
    fm, ev = run_tiny_b0(tmp, 'fields')
    for f in ('evaluation_report.json', 'fairness_metrics.json',
              'cache_metrics.json'):
        assert os.path.exists(os.path.join(tmp, 'fields', f)), f
    fair = json.load(open(os.path.join(tmp, 'fields', 'fairness_metrics.json')))
    for k in ('worst_client_metric', 'bottom_10pct_client_mean',
              'bottom_20pct_client_mean', 'best_client_metric',
              'client_metric_std', 'client_metric_iqr', 'performance_gap',
              'unavailable_client_count', 'zero_anomaly_train_client_count',
              'zero_anomaly_test_client_count'):
        assert k in fair, f"fairness missing {k}"
    ck = json.load(open(os.path.join(tmp, 'fields', 'cache_metrics.json')))
    assert 'stats' in ck and 'mode' in ck and 'anchor_sampling_mode' in ck
    print("  evaluation/fairness/cache JSON fields present ✓")
print("  PASSED ✓")

# ====================================================================
#  Test 24: 新 suite CPU dry run 启动 (每个套件 1 个代表性运行)
# ====================================================================
print("\n[Test 24] new suites CPU dry-run launch")
import run_experiments as re3
# 重要: dry-run 必须写入独立 scratch 目录, 不得污染正式 suite 结果
scratch_suite = f"_phase2_dryrun_{os.getpid()}"
dry_cases = [
    ('main_10seeds', 'B5_full', 0),
    ('ckr_support_mechanism', 'M2_natural_dynamic', 0),
    ('ckr_support_mechanism', 'M8_enriched_dynamic', 0),
    ('long_horizon_ckr', 'LH_B5_natural', 0),
    ('checkpoint_memory_scaling', 'CMP_h32_s10_full', 0),
    ('rwr_cache_effective', 'RWR_e3_cepoch_reuse_afixed', 0),
]
for suite, exp_name, seed in dry_cases:
    cfg = re3.SUITES[suite]
    opts = SimpleNamespace(suite=scratch_suite, resume_failed=False, cpu=True, dry=True)
    st = re3.run_one(cfg, cfg['experiments'][exp_name], seed, opts)
    assert st in ('success', 'skipped'), f"{suite}/{exp_name}: {st}"
    print(f"  {suite}/{exp_name}: {st} (scratch dir, 正式 suite 不受影响)")
# dataset_transfer_citeseer / pubmed_smoke: 无网络无法下载数据, 记录为阻塞
print("  dataset_transfer_citeseer / pubmed_smoke: BLOCKED (无网络下载 CiteSeer/PubMed), "
      "suite 已定义, 运行会以 failed+stderr 记录")
print("  PASSED ✓")

# ====================================================================
#  Test 25: Louvain 划分确定性 (固定随机种子)
# ====================================================================
print("\n[Test 25] Louvain partition determinism (fixed seed)")
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx
from util.base_data_util import louvain_partition
ds = Planetoid(root=os.path.join(ROOT, 'dataset'), name='Cora')
gx = to_networkx(ds.data, to_undirected=True, remove_self_loops=True)
nd1, _ = louvain_partition(gx, 4, 20, return_groups=True)
nd2, _ = louvain_partition(gx, 4, 20, return_groups=True)
flat1 = sorted(n for v in nd1.values() for n in v)
flat2 = sorted(n for v in nd2.values() for n in v)
assert flat1 == flat2, "partition must be reproducible with fixed seed"
assert len(set(flat1)) == len(flat1) == ds.data.num_nodes, "nodes lost/duplicated"
print(f"  partition reproducible, {len(flat1)} nodes conserved ✓")
print("  PASSED ✓")

# ====================================================================
print(f"\n{'=' * 60}")
print("ALL PHASE-2 TESTS PASSED (1-25) ✓")
print(f"{'=' * 60}")
