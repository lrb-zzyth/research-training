"""
正式战役 readiness 测试: formal guard / test isolation / data identity /
protocol resume 一致性。CPU, 不启动任何训练。
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'experiments', 'accuracy_benchmark'))

from train_fedtad import validate_formal_campaign
import formal_campaign as fc


def _good_args(**kw):
    base = dict(task_mode='multiclass', use_weighted_ce=True,
                reliability_holdout_ratio=0, selection_metric='accuracy',
                local_optimizer_lifecycle='reset_each_round',
                radius_constraint=True, feature_stats_align=False,
                diffusion_freeze_skip_scale=True,
                federated_diffusion_pretrain=False,
                contrastive_mode='subgraph_cross_view', distill_loss_type='kl',
                diffusion_skip_mode='timestep_scalar', diffusion_steps=20,
                diffusion_beta_end=0.5, use_posterior_variance=True,
                lambda_diffusion_anchor=1e-2, generator_sampling_steps=0,
                generator_backprop_mode='checkpointed', checkpoint_segments=1,
                fake_graph_topology='knn', tuning_mode=True,
                final_test_only=False,
                # guard 这里只验证“路径存在”；用测试文件自身避免依赖本机 runs/。
                diffusion_pretrained_checkpoint=__file__)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_formal_guard_passes_valid():
    validate_formal_campaign(_good_args())   # 不抛 = PASS
    print("formal guard (合法配置通过): PASS")


def test_formal_guard_fail_fast():
    cases = [('use_weighted_ce', False), ('reliability_holdout_ratio', 0.2),
             ('selection_metric', 'macro_f1'),
             ('local_optimizer_lifecycle', 'persistent'),
             ('radius_constraint', False), ('feature_stats_align', True),
             ('diffusion_freeze_skip_scale', False),
             ('federated_diffusion_pretrain', True)]
    for k, v in cases:
        try:
            validate_formal_campaign(_good_args(**{k: v}))
            raise AssertionError(f'guard 未触发: {k}={v}')
        except ValueError:
            pass
    # Stage1 checkpoint 不存在
    try:
        validate_formal_campaign(_good_args(
            diffusion_pretrained_checkpoint='/nonexistent/x.pt'))
        raise AssertionError('guard 未触发: checkpoint 缺失')
    except ValueError:
        pass
    print("formal guard (8 类违规 fail-fast): PASS")


def test_objective_reads_val_only():
    """objective 只读 best_val_primary: 人为改变 test 数字, objective 不变。"""
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, 'a.json')
        p2 = os.path.join(d, 'b.json')
        m1 = {'best_val_primary': 78.5,
              'metrics': {'pooled': {'accuracy': 77.0}}}
        m2 = {'best_val_primary': 78.5,
              'metrics': {'pooled': {'accuracy': 55.0}}}   # test 完全不同
        json.dump(m1, open(p1, 'w'))
        json.dump(m2, open(p2, 'w'))
        assert fc.objective_value_from_metrics(p1) == 78.5
        assert fc.objective_value_from_metrics(p2) == 78.5
    print("test isolation (objective 不读 test): PASS")


def test_data_identity_fail_fast():
    # 用第一个实际存在的 cached partition 作示例 (服务器部署可能只有部分数据集)
    cand = [('Cora', 5), ('PubMed', 10)]
    h1 = None
    ds = tier = None
    for _ds, _tier in cand:
        if os.path.isdir(os.path.join(
                fc.REPO, 'dataset', _ds,
                'Client%d' % _tier, 'Louvain')):
            ds, tier = _ds, _tier
            h1 = fc.compute_data_identity(ds, tier)
            break
    if h1 is None:
        print('data identity fail-fast: SKIP (无 cached partition)')
        return
    h2 = fc.compute_data_identity(ds, tier)
    assert h1 == h2 and len(h1) == 16, '同一划分 hash 必须稳定'
    saved = {'code_hash': 'x', 'data_identity_hash': 'DIFFERENT',
             'search_space': fc.SEARCH_SPACE, 'dataset': ds,
             'num_clients': tier, 'stage1_checkpoint_hash': 'y'}
    try:
        fc.verify_protocol(saved, dict(saved, data_identity_hash=h1))
        raise AssertionError('data identity 不一致未 fail-fast')
    except ValueError:
        pass
    print('data identity fail-fast: PASS (%s-c%d)' % (ds, tier))


def test_protocol_resume_mismatch():
    saved = {'code_hash': 'oldcode', 'data_identity_hash': 'same',
             'search_space': fc.SEARCH_SPACE, 'dataset': 'PubMed',
             'num_clients': 10, 'stage1_checkpoint_hash': 'same',
             'formal_health_version': 'v2', 'formal_search_space_version': 'v2'}
    # code hash 变化
    try:
        fc.verify_protocol(saved, dict(saved, code_hash='newcode'))
        raise AssertionError('code hash 变化未 fail-fast')
    except ValueError:
        pass
    # 搜索空间变化
    changed = dict(fc.SEARCH_SPACE, distill_steps={'choices': [1, 3, 5, 25]})
    try:
        fc.verify_protocol(saved, dict(saved, search_space=changed))
        raise AssertionError('搜索空间变化未 fail-fast')
    except ValueError:
        pass
    # 完全一致 -> 通过
    fc.verify_protocol(saved, dict(saved))
    print("protocol resume mismatch fail-fast: PASS")


def test_formal_health_rule():
    """health FAIL 的 trial 不得成为合法候选 (即使历史 best_val 在崩溃前)。"""
    with tempfile.TemporaryDirectory() as d:
        # 失败 trial: round10 val=0.90, round90 NaN + grad nan
        with open(os.path.join(d, 'events.jsonl'), 'w') as f:
            f.write(json.dumps({'event': 'global_metric', 'round': 10,
                                'global_val': 0.90}) + '\n')
            f.write(json.dumps({'event': 'global_metric', 'round': 90,
                                'global_val': float('nan')}) + '\n')
        with open(os.path.join(d, 'stdout.log'), 'w') as f:
            f.write('[Round 90]\n  [NAN-DIAG] round 90 client 0 (ce=nan)\n')
            f.write('  step 0: L_sem=1.0 L_dis=2.0 |grad|=nan raw_r=1e17\n')
        ok, info = fc.trial_health_check(d)
        assert ok is False
        assert info['failure_round'] == 90
        assert info['failure_type'] == 'RAW_RADIUS_RUNAWAY_NAN'
        assert info['best_val_before_failure'] == 0.90
        assert info['nonfinite_detected'] is True
        # 干净 trial -> PASS
        d2 = os.path.join(d, 'clean')
        os.makedirs(d2)
        with open(os.path.join(d2, 'events.jsonl'), 'w') as f:
            f.write(json.dumps({'event': 'global_metric', 'round': 10,
                                'global_val': 0.88}) + '\n')
        with open(os.path.join(d2, 'stdout.log'), 'w') as f:
            f.write('[Round 10]\n  step 0: L_sem=1.0 |grad|=0.5 raw_r=7.3\n')
        ok2, info2 = fc.trial_health_check(d2)
        assert ok2 is True and info2['nonfinite_detected'] is False
    print("formal health rule (NaN/runaway -> FAIL, 干净 -> PASS): PASS")


def test_formal_health_v2_raw_inf():
    """formal_health v2: raw 流 nonfinite (即使 projected finite) 必须 FAIL。"""
    with tempfile.TemporaryDirectory() as d:
        # Case 1: raw_radius=inf + projected_radius finite -> FAIL (此前漏检核心)
        with open(os.path.join(d, 'stdout.log'), 'w') as f:
            f.write('[Round 10]\n')
            f.write('  step 0: L_sem=1.0 L_dis=0.5 |grad|=0.4 '
                    'raw_r=inf proj_r=7.33 anchor=0.01\n')
        ok, info = fc.trial_health_check(d)
        assert ok is False and info['failure_round'] == 10, \
            'raw_r=inf 必须 FAIL (projected finite 不豁免)'
        # Case 2: raw feature 含 inf
        with open(os.path.join(d, 'stdout.log'), 'w') as f:
            f.write('[Round 5]\n  step 0: fake_x μ=inf σ=nan raw_r=inf\n')
        ok2, _ = fc.trial_health_check(d)
        assert ok2 is False, 'raw feature inf 必须 FAIL'
        # Case 3: 全部 finite -> PASS
        with open(os.path.join(d, 'stdout.log'), 'w') as f:
            f.write('[Round 5]\n  step 0: L_sem=1.0 L_dis=0.5 |grad|=0.4 '
                    'raw_r=7.3 proj_r=7.3 anchor=0.01\n')
        with open(os.path.join(d, 'events.jsonl'), 'w') as f:
            f.write(json.dumps({'event': 'global_metric', 'round': 5,
                                'global_val': 80.0}) + '\n')
        ok3, info3 = fc.trial_health_check(d)
        assert ok3 is True and info3['nonfinite_detected'] is False
    print("formal_health v2 (raw inf/特征 inf/全 finite): PASS")


def test_candidate_filter_excludes_health_fail():
    """Case 4: Optuna state=COMPLETE 但 health=FAIL 的 trial 必须被候选过滤排除。"""
    class FakeTrial:
        def __init__(self, number, state, health, value):
            self.number = number
            self.state = types.SimpleNamespace(name=state)
            self.user_attrs = {'health': health}
            self.value = value
    trials = [FakeTrial(6, 'COMPLETE', 'FAIL', 83.68),   # 历史误标, 正式排除
              FakeTrial(1, 'COMPLETE', 'PASS', 83.59)]
    candidates = [t for t in trials
                  if t.state.name == 'COMPLETE'
                  and t.user_attrs.get('health') == 'PASS']
    assert [t.number for t in candidates] == [1], \
        'COMPLETE+health FAIL 不得进入候选'
    best = max(candidates, key=lambda t: t.value)
    assert best.number == 1
    print("candidate filter (COMPLETE+FAIL 排除): PASS")


if __name__ == '__main__':
    test_formal_guard_passes_valid()
    test_formal_guard_fail_fast()
    test_objective_reads_val_only()
    test_data_identity_fail_fast()
    test_protocol_resume_mismatch()
    test_formal_health_rule()
    test_formal_health_v2_raw_inf()
    test_candidate_filter_excludes_health_fail()
    print("\nALL FORMAL CAMPAIGN TESTS PASSED")
