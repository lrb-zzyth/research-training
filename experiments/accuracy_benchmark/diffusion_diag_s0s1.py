#!/usr/bin/env python3
"""
扩散去噪器分层诊断 (S0 / S1) —— 定位 "denoise MSE 打不过零预测基线" 的根因层次。

S0 固定 batch 单客户端过拟合:
  固定 x0 / labels / t / epsilon, 在同一个 batch 上反复训练 denoiser。
  目的: 排除 FedAvg / 客户端异质性 / 随机 batch 后, 网络本身能否学会
  一个固定的 epsilon prediction 任务。~1-2 分钟。

S1 单客户端真实随机 denoise 训练:
  随机 batch / t / epsilon, NO FedAvg, NO adversarial, 划小 held-out 子集
  (仅用于 diffusion 诊断, 不影响正式 test protocol)。
  记录 train/held-out MSE + 按 timestep 三分桶 (early/mid/late)。

用法:
  python experiments/accuracy_benchmark/diffusion_diag_s0s1.py --stage s0
  python experiments/accuracy_benchmark/diffusion_diag_s0s1.py --stage s1
  python experiments/accuracy_benchmark/diffusion_diag_s0s1.py --stage all
"""
import argparse
import json
import math
import sys

import torch

sys.path.insert(0, '.')
from model import ConditionalDiffusionGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_client(ds='PubMed', client=0, tier=10, n_sub=256):
    """加载一个真实客户端的 train 节点特征。返回 (x, y)。"""
    d = torch.load(f'dataset/{ds}/Client{tier}/Louvain/data{client}.pt',
                   map_location='cpu', weights_only=False)
    tr = torch.where(d.train_idx)[0]
    x = d.x[tr].float()
    y = d.y[tr]
    if n_sub is not None and x.shape[0] > n_sub:
        idx = torch.randperm(x.shape[0])[:n_sub]
        x, y = x[idx], y[idx]
    print(f"[data] client{client}: {x.shape[0]} train nodes, "
          f"F={x.shape[1]}, x std={x.std():.4f}, "
          f"x norm median={x.norm(dim=1).median():.4f}")
    return x.to(DEVICE), y.to(DEVICE)


def make_gen(num_steps=20, beta_end=0.5, hidden=256, lr=1e-3,
             residual_skip=False, skip_mode='none'):
    """与正式 pretrain 相同的网络 / 调度 / 优化器。"""
    gen = ConditionalDiffusionGenerator(
        feat_dim=500, num_classes=3, hidden_dim=hidden,
        num_steps=num_steps, beta_start=1e-4, beta_end=beta_end,
        output_bound='tanh', residual_skip=residual_skip,
        skip_mode=skip_mode).to(DEVICE)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    return gen, opt


def zero_baseline(eps):
    """零预测的 MSE = mean(eps^2) (理论 ≈1.0, 实际按 batch 算)。"""
    return float((eps ** 2).mean().item())


def fixed_batch_run(x, y, t_val, steps=2000, report_every=200):
    """S0-A: 固定 x0/labels/t/epsilon, 同 batch 反复训练。"""
    gen, opt = make_gen()
    gen.train()
    torch.manual_seed(2024)
    B = x.shape[0]
    t = torch.full((B,), t_val, dtype=torch.long, device=DEVICE)
    eps = torch.randn_like(x)
    alpha_bar_t = gen.alpha_bars[t_val]
    x_t = (math.sqrt(alpha_bar_t.item()) * x
           + math.sqrt(1.0 - alpha_bar_t.item()) * eps)
    z_base = zero_baseline(eps)
    records = []
    for step in range(steps):
        opt.zero_grad()
        eps_pred = gen(x_t, t, y)
        loss = torch.nn.functional.mse_loss(eps_pred, eps)
        loss.backward()
        gn = math.sqrt(sum(p.grad.norm().item() ** 2
                           for p in gen.parameters() if p.grad is not None))
        opt.step()
        if step % report_every == 0 or step == steps - 1:
            rec = {'step': step, 'mse': float(loss.item()),
                   'grad_norm': round(gn, 4),
                   'eps_pred_mean': float(eps_pred.mean().item()),
                   'eps_pred_std': float(eps_pred.std().item()),
                   'eps_true_mean': float(eps.mean().item()),
                   'eps_true_std': float(eps.std().item())}
            records.append(rec)
            print(f"    step {step:5d}: MSE={loss.item():.5f} "
                  f"(zero={z_base:.4f}, ratio={loss.item() / z_base:.3f}) "
                  f"|grad|={gn:.4f} eps_pred μ={eps_pred.mean().item():+.4f} "
                  f"σ={eps_pred.std().item():.4f}")
    return records, z_base


def run_s0():
    print("=" * 70)
    print("S0: 固定 batch 单客户端过拟合 (PubMed client0, 256 nodes)")
    print("=" * 70)
    x, y = load_client()
    out = {}
    for t_val in (5, 10, 15):   # 主测试 t=10; 5/15 作辅助 (SNR 差异探针)
        print(f"\n--- fixed t={t_val} (alpha_bar={gen_ab(t_val):.4f}) ---")
        recs, z_base = fixed_batch_run(x, y, t_val)
        out[t_val] = {'records': recs, 'zero_baseline': z_base}
        final = recs[-1]
        ratio = final['mse'] / z_base
        print(f"  [S0 t={t_val}] initial MSE={recs[0]['mse']:.4f} -> "
              f"final MSE={final['mse']:.4f} | zero={z_base:.4f} | "
              f"ratio={ratio:.3f} | "
              f"{'PASS' if ratio < 0.5 else 'FAIL'}")
    json.dump(out, open('runs/accuracy_benchmark/DIAG_S0.json', 'w'),
              indent=1, default=str)
    print("\nS0_DONE")


def gen_ab(t):
    return ConditionalDiffusionGenerator(
        feat_dim=500, num_classes=3, hidden_dim=256,
        num_steps=20, beta_end=0.5,
        output_bound='tanh').alpha_bars[t].item()


def run_s1():
    print("=" * 70)
    print("S1: 单客户端随机 t/epsilon denoise 训练 (NO FedAvg)")
    print("=" * 70)
    x, y = load_client(n_sub=512)
    n = x.shape[0]
    n_tr = int(n * 0.8)
    torch.manual_seed(2024)
    perm = torch.randperm(n)
    tr_idx, ho_idx = perm[:n_tr], perm[n_tr:]
    gen, opt = make_gen()
    gen.train()
    T = gen.num_steps
    bs = 128
    steps = 2000
    z_base_tr, z_base_ho = None, None
    out = []
    for step in range(steps):
        idx = torch.randint(0, n_tr, (bs,), device='cpu')   # CPU 索引 (tr_idx 在 CPU)
        xb = x[tr_idx[idx]]
        yb = y[tr_idx[idx]]
        t = torch.randint(0, T, (bs,), device=DEVICE)
        eps = torch.randn_like(xb)
        alpha_bar_t = gen.alpha_bars[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * xb + torch.sqrt(1.0 - alpha_bar_t) * eps
        opt.zero_grad()
        eps_pred = gen(x_t, t, yb)
        loss = torch.nn.functional.mse_loss(eps_pred, eps)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            rec = {'step': step, 'train_mse': float(loss.item())}
            # held-out 评估: 8 次随机 (t, eps) 平均, 并按 t 三分桶
            gen.eval()
            with torch.no_grad():
                ho_mse = {'all': 0.0, 'early': 0.0, 'mid': 0.0, 'late': 0.0}
                ho_cnt = {'all': 0, 'early': 0, 'mid': 0, 'late': 0}
                eps_ho = torch.randn_like(x[ho_idx])
                for _ in range(8):
                    t_ho = torch.randint(0, T, (len(ho_idx),), device=DEVICE)
                    ab = gen.alpha_bars[t_ho].unsqueeze(-1)
                    x_t_ho = (torch.sqrt(ab) * x[ho_idx]
                              + torch.sqrt(1.0 - ab) * eps_ho)
                    pred = gen(x_t_ho, t_ho, y[ho_idx])
                    mse = (pred - eps_ho) ** 2
                    ho_mse['all'] += mse.mean().item()
                    ho_cnt['all'] += 1
                    for key, (lo, hi) in {'early': (0, T // 3),
                                          'mid': (T // 3, 2 * T // 3),
                                          'late': (2 * T // 3, T)}.items():
                        mask = (t_ho >= lo) & (t_ho < hi)
                        if mask.any():
                            ho_mse[key] += mse[mask].mean().item()
                            ho_cnt[key] += 1
                if z_base_ho is None:
                    z_base_ho = (eps_ho ** 2).mean().item()
                    z_base_tr = 1.0   # train 侧每步重抽 eps, 用理论值
                for k in ho_mse:
                    rec[f'ho_mse_{k}'] = ho_mse[k] / max(1, ho_cnt[k])
                rec['zero_baseline_ho'] = z_base_ho
            gen.train()
            ratio = rec['ho_mse_all'] / z_base_ho
            out.append(rec)
            print(f"    step {step:5d}: train={rec['train_mse']:.4f} "
                  f"ho={rec['ho_mse_all']:.4f} (zero={z_base_ho:.4f}, "
                  f"ratio={ratio:.3f}) | early={rec['ho_mse_early']:.4f} "
                  f"mid={rec['ho_mse_mid']:.4f} late={rec['ho_mse_late']:.4f}")
    json.dump(out, open('runs/accuracy_benchmark/DIAG_S1.json', 'w'),
              indent=1, default=str)
    last = out[-1]
    ratio = last['ho_mse_all'] / z_base_ho
    print(f"\n[S1] train={last['train_mse']:.4f} held-out={last['ho_mse_all']:.4f}"
          f" zero={z_base_ho:.4f} ratio={ratio:.3f} | "
          f"{'PASS' if ratio < 1.0 and last['ho_mse_all'] < last['train_mse'] * 2 else 'FAIL'}")
    print("S1_DONE")


def run_s1ab(steps=600):
    """A/B: 当前架构 vs 直通残差 (600 本地步, 单客户端)。看 late-t MSE。"""
    x, y = load_client(n_sub=512)
    n = x.shape[0]
    n_tr = int(n * 0.8)
    torch.manual_seed(2024)
    perm = torch.randperm(n)
    tr_idx, ho_idx = perm[:n_tr], perm[n_tr:]
    # 固定 held-out (x0, t, eps, x_t) 供两种架构公平对比
    T = 20
    ho_t = torch.randint(0, T, (len(ho_idx),), device=DEVICE)
    ho_eps = torch.randn_like(x[ho_idx])
    ho_ab = None
    results = {}
    for name, skip in [('A_no_skip', False), ('B_residual_skip', True)]:
        gen, opt = make_gen(residual_skip=skip)
        gen.train()
        bs = 128
        print(f"\n=== S1-A/B [{name}] {steps} 步 ===")
        for step in range(steps):
            idx = torch.randint(0, n_tr, (bs,), device='cpu')
            xb, yb = x[tr_idx[idx]], y[tr_idx[idx]]
            t = torch.randint(0, T, (bs,), device=DEVICE)
            eps = torch.randn(xb.shape, device=DEVICE)
            ab = gen.alpha_bars[t].unsqueeze(-1)
            x_t = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * eps
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(gen(x_t, t, yb), eps)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == steps - 1:
                gen.eval()
                with torch.no_grad():
                    ab_ho = gen.alpha_bars[ho_t].unsqueeze(-1)
                    x_t_ho = (torch.sqrt(ab_ho) * x[ho_idx]
                              + torch.sqrt(1.0 - ab_ho) * ho_eps)
                    pred = gen(x_t_ho, ho_t, y[ho_idx])
                    mse_all = float(((pred - ho_eps) ** 2).mean().item())
                    bucket = {}
                    ident = {}
                    for bk, (lo, hi) in {'early': (0, 6), 'mid': (6, 13),
                                         'late': (13, 20)}.items():
                        mask = (ho_t >= lo) & (ho_t < hi)
                        bucket[bk] = float(((pred[mask] - ho_eps[mask]) ** 2)
                                          .mean().item())
                        ident[bk] = float(((x_t_ho[mask]
                                            / torch.sqrt(1.0 - ab_ho[mask])
                                            - ho_eps[mask]) ** 2).mean().item())
                    # x0 重构 RMSE (late)
                    late_mask = ho_t >= 13
                    x0_hat = ((x_t_ho[late_mask]
                               - torch.sqrt(1.0 - ab_ho[late_mask])
                               * pred[late_mask])
                              / torch.sqrt(ab_ho[late_mask]))
                    x0_rmse = float(((x0_hat - x[ho_idx][late_mask]) ** 2)
                                    .mean().item()) ** 0.5
                gen.train()
                late_ratio = bucket['late'] / max(ident['late'], 1e-12)
                print(f"    step {step:4d}: ho_mse={mse_all:.4f} "
                      f"early={bucket['early']:.4f} mid={bucket['mid']:.4f} "
                      f"late={bucket['late']:.4f} (identity={ident['late']:.5f}, "
                      f"model/identity={late_ratio:.1f}x) late_x0_rmse={x0_rmse:.3f}")
        results[name] = dict(bucket=bucket, ident_late=ident['late'],
                             late_ratio=late_ratio, late_x0_rmse=x0_rmse)
    print("\n=== A/B 汇总 ===")
    for name, r in results.items():
        print(f"  {name}: late_mse={r['bucket']['late']:.4f} "
              f"(identity={r['ident_late']:.5f}, ratio={r['late_ratio']:.1f}x) "
              f"late_x0_rmse={r['late_x0_rmse']:.3f}")
    json.dump(results, open('runs/accuracy_benchmark/DIAG_S1AB.json', 'w'),
              indent=1, default=str)
    print("S1AB_DONE")


def run_s1c(steps=600):
    """C: timestep_scalar skip (c(t)=exp(g_t)·x_t + body)。与 B 同协议。
    早停门控: step 200 时 early MSE 未明显低于 B 同期的 0.4531 -> 停止。"""
    x, y = load_client(n_sub=512)
    n = x.shape[0]
    n_tr = int(n * 0.8)
    torch.manual_seed(2024)
    perm = torch.randperm(n)
    tr_idx, ho_idx = perm[:n_tr], perm[n_tr:]
    T = 20
    ho_t = torch.randint(0, T, (len(ho_idx),), device=DEVICE)
    ho_eps = torch.randn_like(x[ho_idx])

    gen, opt = make_gen(skip_mode='timestep_scalar')
    gen.train()
    bs = 128
    recs = []
    print(f"\n=== S1-C [timestep_scalar skip] {steps} 步 ===")
    B_early_200 = 0.4531   # B (fixed) 在 step 200 的 early MSE (历史 reference)
    for step in range(steps):
        idx = torch.randint(0, n_tr, (bs,), device='cpu')
        xb, yb = x[tr_idx[idx]], y[tr_idx[idx]]
        t = torch.randint(0, T, (bs,), device=DEVICE)
        eps = torch.randn(xb.shape, device=DEVICE)
        ab = gen.alpha_bars[t].unsqueeze(-1)
        x_t = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * eps
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(gen(x_t, t, yb), eps)
        loss.backward()
        opt.step()
        if step in (0, 50, 100, 200, 400) or step == steps - 1:
            gen.eval()
            with torch.no_grad():
                ab_ho = gen.alpha_bars[ho_t].unsqueeze(-1)
                x_t_ho = (torch.sqrt(ab_ho) * x[ho_idx]
                          + torch.sqrt(1.0 - ab_ho) * ho_eps)
                pred = gen(x_t_ho, ho_t, y[ho_idx])
                mse_all = float(((pred - ho_eps) ** 2).mean().item())
                bucket = {}
                for bk, (lo, hi) in {'early': (0, 6), 'mid': (6, 13),
                                     'late': (13, 20)}.items():
                    mask = (ho_t >= lo) & (ho_t < hi)
                    bucket[bk] = float(((pred[mask] - ho_eps[mask]) ** 2)
                                      .mean().item())
            gen.train()
            recs.append(dict(step=step, overall=mse_all, **bucket))
            print(f"    step {step:4d}: overall={mse_all:.4f} "
                  f"early={bucket['early']:.4f} mid={bucket['mid']:.4f} "
                  f"late={bucket['late']:.4f}")
            if step == 200 and bucket['early'] >= B_early_200:
                print(f"  [早停门控] step 200 early={bucket['early']:.4f} "
                      f"未明显低于 B(0.4531), 停止")
                break
    # learned c(t) vs diagnostic optimal c*(t)
    print("\nlearned c(t) vs diagnostic optimal c*(t):")
    print(f"{'t':>3} {'alpha_bar':>10} {'c(t)':>10} {'c*(t)':>10}")
    ct = torch.exp(gen.skip_log_scale.weight).squeeze(-1)
    ct_table = {}
    with torch.no_grad():
        ab_ho = gen.alpha_bars[ho_t].unsqueeze(-1)
        x_t_ho = (torch.sqrt(ab_ho) * x[ho_idx]
                  + torch.sqrt(1.0 - ab_ho) * ho_eps)
        for tv in range(T):
            mask = ho_t == tv
            if not mask.any():
                continue
            num = float((x_t_ho[mask] * ho_eps[mask]).sum().item())
            den = float((x_t_ho[mask] ** 2).sum().item())
            ct_table[tv] = {'c_learned': float(ct[tv].item()),
                            'c_star': num / max(den, 1e-12)}
    for tv in sorted(ct_table):
        print(f"{tv:3d} {gen.alpha_bars[tv].item():10.5f} "
              f"{ct_table[tv]['c_learned']:10.4f} {ct_table[tv]['c_star']:10.4f}")
    json.dump({'records': recs, 'c_table': ct_table},
              open('runs/accuracy_benchmark/DIAG_S1C.json', 'w'), indent=1,
              default=str)
    last = recs[-1]
    print(f"\n=== C 汇总 === overall={last['overall']:.4f} "
          f"early={last['early']:.4f} mid={last['mid']:.4f} "
          f"late={last['late']:.4f}")
    print("S1C_DONE")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='s0',
                    choices=['s0', 's1', 's1ab', 's1c', 'all'])
    args = ap.parse_args()
    if args.stage in ('s0', 'all'):
        run_s0()
    if args.stage in ('s1', 'all'):
        run_s1()
    if args.stage in ('s1ab', 'all'):
        run_s1ab()
    if args.stage in ('s1c', 'all'):
        run_s1c()
