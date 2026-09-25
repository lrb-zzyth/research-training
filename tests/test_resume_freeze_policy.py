"""
Bug A regression: resume 时 freeze 策略 fail-fast, 且进入下一轮真实 generator
update 的 trainability toggle 后 skip 仍保持冻结/可训练与 checkpoint 一致。

CPU only, 合成配置。
"""
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ConditionalDiffusionGenerator
from train_fedtad import (resolve_resume_freeze_policy,
                          configure_generator_trainability)
from util.checkpoint import save_checkpoint, load_checkpoint


def _gen():
    return ConditionalDiffusionGenerator(
        feat_dim=8, num_classes=3, hidden_dim=16, num_steps=4,
        beta_start=1e-4, beta_end=0.5, output_bound='tanh',
        skip_mode='timestep_scalar', use_posterior_variance=True)


def _save_ckpt(d, freeze):
    """按给定 freeze 策略构造 generator/optimizer 并存 checkpoint (模拟 Stage2 存档)。"""
    gen = _gen()
    configure_generator_trainability(gen, True, freeze_skip_scale=freeze)
    opt = torch.optim.Adam([p for p in gen.parameters() if p.requires_grad],
                           lr=1e-3)
    args = types.SimpleNamespace(diffusion_freeze_skip_scale=freeze)
    save_checkpoint(d, 'best.pt', gen, gen_optimizer=opt, args=args,
                    round_idx=1, task_mode='multiclass', num_classes=3,
                    feat_dim=8)
    return gen, opt


def _simulate_resume_and_round_toggle(d, cli_freeze):
    """真实 resume 路径: resolve policy → 重建 optimizer → load → 模拟 [8] 块 toggle。"""
    ck_peek = torch.load(os.path.join(d, 'best.pt'), map_location='cpu',
                         weights_only=False)
    policy = resolve_resume_freeze_policy(ck_peek.get('args'), cli_freeze)
    del ck_peek
    gen = _gen()
    configure_generator_trainability(gen, True, freeze_skip_scale=policy)
    opt = torch.optim.Adam([p for p in gen.parameters() if p.requires_grad],
                           lr=1e-3)
    load_checkpoint(os.path.join(d, 'best.pt'), generator=gen,
                    gen_optimizer=opt)
    # 真实主循环 [8] 块进入 generator update 前的 toggle (使用当前 CLI 策略)
    configure_generator_trainability(gen, True, freeze_skip_scale=cli_freeze)
    return gen, opt


def test_A1_checkpoint_freeze_true_cli_true():
    """checkpoint freeze=True + CLI freeze=True → 成功, 下一轮 toggle 后仍冻结。"""
    with tempfile.TemporaryDirectory() as d:
        _save_ckpt(d, freeze=True)
        gen, opt = _simulate_resume_and_round_toggle(d, cli_freeze=True)
        skip = gen.skip_log_scale.weight
        assert skip.requires_grad is False, "A1: toggle 后 skip 必须仍冻结"
        param_ids = {id(p) for grp in opt.param_groups for p in grp['params']}
        assert id(skip) not in param_ids, "A1: optimizer 不得包含 skip"
    print("BugA A1 (freeze=True/CLI=True): PASS")


def test_A2_checkpoint_freeze_true_cli_false():
    """checkpoint freeze=True + CLI freeze=False → resume 立即 fail-fast。"""
    with tempfile.TemporaryDirectory() as d:
        _save_ckpt(d, freeze=True)
        try:
            _simulate_resume_and_round_toggle(d, cli_freeze=False)
            raise AssertionError("A2: 策略不一致时必须 raise ValueError")
        except ValueError as e:
            assert 'diffusion_freeze_skip_scale' in str(e)
            assert 'True' in str(e) and 'False' in str(e)
    print("BugA A2 (freeze=True/CLI=False fail-fast): PASS")


def test_A3_checkpoint_freeze_false_cli_false():
    """checkpoint freeze=False + CLI freeze=False → 成功, skip 保持可训练。"""
    with tempfile.TemporaryDirectory() as d:
        _save_ckpt(d, freeze=False)
        gen, opt = _simulate_resume_and_round_toggle(d, cli_freeze=False)
        skip = gen.skip_log_scale.weight
        assert skip.requires_grad is True, "A3: 无冻结策略时 skip 应可训练"
        param_ids = {id(p) for grp in opt.param_groups for p in grp['params']}
        assert id(skip) in param_ids, "A3: optimizer 应包含可训练 skip"
    print("BugA A3 (freeze=False/CLI=False): PASS")


def test_legacy_checkpoint_without_field():
    """legacy checkpoint (args 无该字段): 不 raise, 返回 CLI 策略, 明确警告。"""
    policy = resolve_resume_freeze_policy(
        {'num_rounds': 100}, cli_freeze=True)   # 无 freeze 字段
    assert policy is True
    policy2 = resolve_resume_freeze_policy(None, cli_freeze=False)
    assert policy2 is False
    print("BugA legacy checkpoint path: PASS")


if __name__ == '__main__':
    test_A1_checkpoint_freeze_true_cli_true()
    test_A2_checkpoint_freeze_true_cli_false()
    test_A3_checkpoint_freeze_false_cli_false()
    test_legacy_checkpoint_without_field()
    print("\nALL RESUME FREEZE POLICY TESTS PASSED")
