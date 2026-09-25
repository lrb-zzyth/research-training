"""
Correctness patch Stage2 smoke test (合成配置, CPU):

  fresh pretrained 初始化 → 冻结 skip → 建 optimizer → 一次对抗 backward
  → checkpoint 保存 → 重建 → resume

验证:
  skip requires_grad=False / grad=None / 值不变
  residual 参数可训练
  optimizer 参数组匹配
  pretrain_ref / radius_bank 恢复
  resume 不重新加载 Stage1 pretrained checkpoint (互斥 guard)
  resume 不重新运行 federated pretrain (guard)
"""
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ConditionalDiffusionGenerator, GCN
from train_fedtad import (init_stage2_from_pretrained,
                          configure_generator_trainability,
                          validate_run_args)
from util.checkpoint import save_checkpoint, load_checkpoint

DEVICE = 'cpu'
FEAT, CLS, STEPS = 8, 3, 4


def _args(**kw):
    base = dict(generator_lr=1e-3, diffusion_freeze_skip_scale=True,
                radius_constraint=True, radius_bank_size=8,
                diffusion_residual_skip=False, diffusion_skip_mode='timestep_scalar',
                diffusion_steps=STEPS, diffusion_beta_start=1e-4,
                diffusion_beta_end=0.5, use_posterior_variance=True,
                resume_checkpoint='', diffusion_pretrained_checkpoint='',
                federated_diffusion_pretrain=False,
                feature_stats_align=False,
                local_optimizer_lifecycle='persistent',
                allow_legacy_resume_without_local_optimizer_state=False,
                seed=2024)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _gen():
    return ConditionalDiffusionGenerator(
        feat_dim=FEAT, num_classes=CLS, hidden_dim=16, num_steps=STEPS,
        beta_start=1e-4, beta_end=0.5, output_bound='tanh',
        skip_mode='timestep_scalar', use_posterior_variance=True)


def test_stage2_smoke():
    torch.manual_seed(0)
    # ---- 1. 伪造一个 Stage1 R20 权重 (随机训练几步的 denoiser) ----
    gen0 = _gen()
    opt0 = torch.optim.Adam(gen0.parameters(), lr=1e-3)
    for _ in range(5):
        opt0.zero_grad()
        t = torch.randint(0, STEPS, (6,))
        x0 = torch.randn(6, FEAT)
        eps = torch.randn(6, FEAT)
        ab = gen0.alpha_bars[t].unsqueeze(-1)
        x_t = ab ** 0.5 * x0 + (1.0 - ab) ** 0.5 * eps
        loss = torch.nn.functional.mse_loss(gen0(x_t, t, torch.tensor([0, 1, 2] * 2)), eps)
        loss.backward()
        opt0.step()
    with tempfile.TemporaryDirectory() as d:
        r20_path = os.path.join(d, 'r20.pt')
        torch.save(gen0.state_dict(), r20_path)

        # ---- 2. Fresh Stage2 初始化 ----
        args = _args(diffusion_pretrained_checkpoint=r20_path)
        gen = _gen()
        pretrain_ref, radius_bank, gen_opt = init_stage2_from_pretrained(
            gen, r20_path, args, DEVICE, CLS)
        skip = gen.skip_log_scale.weight
        skip_val = skip.detach().clone()
        assert skip.requires_grad is False
        assert radius_bank is not None and radius_bank.shape[0] == CLS
        param_ids = {id(p) for grp in gen_opt.param_groups for p in grp['params']}
        assert id(skip) not in param_ids

        # ---- 3. 一次对抗 backward (简化: 任意标量 loss) ----
        gen.zero_grad(set_to_none=True)
        t = torch.randint(0, STEPS, (4,))
        labels = torch.tensor([0, 1, 2, 0])
        loss = gen(torch.randn(4, FEAT), t, labels).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
        gen_opt.step()
        assert skip.grad is None and torch.equal(skip, skip_val)
        assert any(p.grad is not None for p in gen.net.parameters())

        # ---- 4. 保存 checkpoint (含 freeze 策略 / ref / bank / 本地 opt) ----
        global_model = GCN(FEAT, 8, CLS, 0.0)
        global_opt = torch.optim.Adam(global_model.parameters(), lr=1e-3)
        local_opt = torch.optim.Adam(global_model.parameters(), lr=1e-3)
        save_checkpoint(d, 'best.pt', global_model, generator=gen,
                        global_optimizer=global_opt, gen_optimizer=gen_opt,
                        args=args, round_idx=3, best_metric=50.0,
                        task_mode='multiclass', num_classes=CLS, feat_dim=FEAT,
                        generator_cfg=dict(skip_mode='timestep_scalar',
                                           use_posterior_variance=True),
                        pretrain_ref=pretrain_ref, radius_bank=radius_bank,
                        local_optimizers=[local_opt])

        # ---- 5. 重建 + resume (模拟 resume 路径的 optimizer 参数组重建) ----
        ck_peek = torch.load(os.path.join(d, 'best.pt'), map_location='cpu',
                             weights_only=False)
        saved_freeze = bool(ck_peek['args']['diffusion_freeze_skip_scale'])
        assert saved_freeze is True
        gen2 = _gen()
        configure_generator_trainability(gen2, True,
                                         freeze_skip_scale=saved_freeze)
        gen_opt2 = torch.optim.Adam([p for p in gen2.parameters()
                                     if p.requires_grad], lr=1e-3)
        gm2 = GCN(FEAT, 8, CLS, 0.0)
        go2 = torch.optim.Adam(gm2.parameters(), lr=1e-3)
        lo2 = torch.optim.Adam(gm2.parameters(), lr=1e-3)
        ck = load_checkpoint(os.path.join(d, 'best.pt'), global_model=gm2,
                             generator=gen2, global_optimizer=go2,
                             gen_optimizer=gen_opt2, local_optimizers=[lo2])
        assert ck['round'] == 3
        assert gen2.skip_log_scale.weight.requires_grad is False
        assert torch.equal(gen2.skip_log_scale.weight, skip_val)
        assert ck.get('pretrain_ref') is not None
        assert ck.get('radius_bank') is not None
        assert len(gen_opt2.param_groups) == len(gen_opt.param_groups)

        # ---- 6. resume 与 fresh 互斥 / 不二次预训练 (guards) ----
        for bad in (dict(resume_checkpoint='x.pt',
                         diffusion_pretrained_checkpoint='y.pt'),
                    dict(diffusion_pretrained_checkpoint='y.pt',
                         federated_diffusion_pretrain=True)):
            try:
                validate_run_args(_args(**bad))
                raise AssertionError(f"guard 未触发: {bad}")
            except ValueError:
                pass
    print("Stage2 smoke (fresh→backward→save→resume→guards): PASS")


if __name__ == '__main__':
    test_stage2_smoke()
    print("\nSTAGE2 SMOKE PASSED")
