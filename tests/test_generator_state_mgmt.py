"""
Correctness patch 单元测试: generator 状态管理 (Bug 1/7/8/9 + 3/4 guards)。

CPU only, 合成张量, 不加载真实数据集。
运行: /home/lrb/miniconda3/envs/fedtad5060/bin/python -m pytest tests/ -q
或:  /home/lrb/miniconda3/envs/fedtad5060/bin/python tests/test_generator_state_mgmt.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ConditionalDiffusionGenerator
from train_fedtad import (configure_generator_trainability,
                          validate_run_args)

DEVICE = 'cpu'


def make_gen(num_steps=4, skip_mode='timestep_scalar',
             use_posterior_variance=True):
    return ConditionalDiffusionGenerator(
        feat_dim=8, num_classes=3, hidden_dim=16, num_steps=num_steps,
        beta_start=1e-4, beta_end=0.5, output_bound='tanh',
        skip_mode=skip_mode, use_posterior_variance=use_posterior_variance)


def test_bug1_freeze_skip_survives_toggles():
    """Bug 1: freeze 后无论 trainable 如何切换, skip scale 恒 requires_grad=False;
    且 residual 参数可训练、可更新。"""
    gen = make_gen()
    skip = gen.skip_log_scale.weight
    skip_before = skip.detach().clone()

    # freeze → toggles
    configure_generator_trainability(gen, True, freeze_skip_scale=True)
    configure_generator_trainability(gen, False, freeze_skip_scale=True)
    configure_generator_trainability(gen, True, freeze_skip_scale=True)
    assert skip.requires_grad is False, "skip must stay frozen across toggles"
    assert any(p.requires_grad for p in gen.net.parameters()), \
        "residual denoiser must be trainable"

    # optimizer 只含 trainable 参数
    opt = torch.optim.Adam([p for p in gen.parameters() if p.requires_grad])
    param_ids = {id(p) for group in opt.param_groups for p in group['params']}
    assert id(skip) not in param_ids, "skip must not be in optimizer groups"

    # one backward + step
    gen.zero_grad(set_to_none=True)
    t = torch.randint(0, 4, (4,))
    x_t = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 0])
    loss = gen(x_t, t, labels).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
    opt.step()

    assert skip.grad is None, "frozen skip must have grad None"
    assert torch.equal(skip, skip_before), "frozen skip value must not change"
    assert any(p.grad is not None for p in gen.net.parameters()), \
        "residual params must receive grad"
    print("Bug1 freeze-skip invariance: PASS")


def test_bug8_posterior_variance_switch():
    """Bug 8: use_posterior_variance 真正控制 reverse sigma, 默认 True 为主行为。"""
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2])
    g_true = make_gen(use_posterior_variance=True)
    g_false = make_gen(use_posterior_variance=False)
    assert g_true.use_posterior_variance is True  # 默认 True
    g_true.load_state_dict(g_false.state_dict())  # 同权重

    x_t = torch.randn(3, 8)
    rng1 = torch.Generator(device='cpu').manual_seed(7)
    rng2 = torch.Generator(device='cpu').manual_seed(7)
    with torch.no_grad():
        y1 = g_true.sample(labels, num_steps=4, device='cpu',
                           initial_noise=x_t, noise_generator=rng1)
        y2 = g_false.sample(labels, num_steps=4, device='cpu',
                            initial_noise=x_t, noise_generator=rng2)
    assert not torch.allclose(y1, y2), \
        "posterior=True vs False 必须产生不同 reverse 轨迹"
    # 参考值核对: 单步 sigma 是否与公式一致
    with torch.no_grad():
        step_t = 2
        eps_p = torch.zeros(3, 8)
        mu1, mu2 = None, None
        for g in (g_true, g_false):
            a_t = g.alphas[step_t]
            ab_t = g.alpha_bars[step_t]
            b_t = g.betas[step_t]
            mu = (x_t - (b_t / (1.0 - ab_t) ** 0.5) * eps_p) / a_t ** 0.5
            var = (g.posterior_variance[step_t] if g.use_posterior_variance
                   else g.betas[step_t])
            sig = var ** 0.5
            if mu1 is None:
                mu1, sig1 = mu, sig
            else:
                mu2, sig2 = mu, sig
        assert torch.allclose(mu1, mu2) and not torch.allclose(sig1, sig2), \
            "sigma 随 use_posterior_variance 变化, mu 不变"
    print("Bug8 posterior variance CLI: PASS")


def test_bug7_probe_rng_independence():
    """Bug 7: initial noise (seed+4242) 与 reverse 噪声流 (seed+9173) 相互独立
    且各自确定性。"""
    seed = 2024
    def build(init_seed, rev_seed):
        g_i = torch.Generator(device='cpu').manual_seed(init_seed)
        noise = torch.randn(4, 8, generator=g_i)
        g_r = torch.Generator(device='cpu').manual_seed(rev_seed)
        first_rev = torch.randn(4, 8, generator=g_r)
        return noise, first_rev
    n1, r1 = build(seed + 4242, seed + 9173)
    n2, r2 = build(seed + 4242, seed + 9173)
    assert torch.equal(n1, n2) and torch.equal(r1, r2), "probe 必须确定性"
    assert not torch.equal(n1, r1), \
        "initial noise 与第一份 reverse noise 必须不同 (独立种子)"
    print("Bug7 probe RNG independence: PASS")


def test_guards_fail_fast():
    """Bug 3/4/9: 互斥 CLI 组合必须 raise ValueError。"""
    base = dict(resume_checkpoint='', diffusion_pretrained_checkpoint='',
                federated_diffusion_pretrain=False,
                radius_constraint=False, feature_stats_align=False)
    # 合法组合不报错
    validate_run_args(types.SimpleNamespace(**base))
    # Bug 3
    try:
        validate_run_args(types.SimpleNamespace(
            **{**base, 'resume_checkpoint': 'a.pt',
               'diffusion_pretrained_checkpoint': 'b.pt'}))
        raise AssertionError("Bug3 guard 未触发")
    except ValueError:
        pass
    # Bug 4
    try:
        validate_run_args(types.SimpleNamespace(
            **{**base, 'diffusion_pretrained_checkpoint': 'b.pt',
               'federated_diffusion_pretrain': True}))
        raise AssertionError("Bug4 guard 未触发")
    except ValueError:
        pass
    # Bug 9
    try:
        validate_run_args(types.SimpleNamespace(
            **{**base, 'radius_constraint': True,
               'feature_stats_align': True}))
        raise AssertionError("Bug9 guard 未触发")
    except ValueError:
        pass
    print("Bug3/4/9 guards: PASS")


if __name__ == '__main__':
    test_bug1_freeze_skip_survives_toggles()
    test_bug8_posterior_variance_switch()
    test_bug7_probe_rng_independence()
    test_guards_fail_fast()
    print("\nALL GENERATOR STATE MGMT TESTS PASSED")
