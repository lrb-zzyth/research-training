#!/usr/bin/env python3
"""
S3 诊断: analytical baselines + architecture 事实 + x0 reconstruction amplification。

无新训练: 重建训练代码同款固定 diag subset, 加载 pretrain_diagnostic_only 保存的
pretrained_generator.pt, 计算:
  S3-A  zero / identity / optimal-scalar / model 四种 epsilon predictor 的 MSE (分桶 + 末 3 步)
  S3-C  oracle x0 reconstruction (应到浮点误差) + 模型重构 + 放大因子一致性核验

用法:
  python experiments/accuracy_benchmark/diffusion_diag_s3.py \
      --ckpt runs/accuracy_benchmark/DIAG_pubmed10_long30/pretrained_generator.pt
"""
import argparse
import json
import math
import sys

import torch

sys.path.insert(0, '.')
from model import ConditionalDiffusionGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rebuild_diag(seed=2024, ds='PubMed', tier=10, n_clients=10):
    """与 train_fedtad.federated_diffusion_pretrain 完全相同的固定 diag subset。"""
    pooled = {'x0': [], 'y': [], 't': [], 'eps': [], 'xt': [], 'ci': []}
    for ci in range(n_clients):
        d = torch.load(f'dataset/{ds}/Client{tier}/Louvain/data{ci}.pt',
                       map_location='cpu', weights_only=False)
        tr = torch.where(d.train_idx)[0]
        x = d.x[tr].float().to(DEVICE)
        y = d.y[tr].to(DEVICE)
        n = x.shape[0]
        n_diag = min(128, n)
        g_cpu = torch.Generator(device='cpu').manual_seed(seed + ci * 1000 + 7)
        g_dev = torch.Generator(device=DEVICE).manual_seed(seed + ci * 1000 + 7)
        idx = torch.randperm(n, generator=g_cpu)[:n_diag]
        x0d = x[idx]
        yd = y[idx]
        td = torch.randint(0, 20, (n_diag,), generator=g_dev, device=DEVICE)
        epsd = torch.randn(x0d.shape, generator=g_dev, device=DEVICE,
                           dtype=x0d.dtype)
        ab = gen_buffers['alpha_bars'][td].unsqueeze(-1)
        xtd = torch.sqrt(ab) * x0d + torch.sqrt(1.0 - ab) * epsd
        for k, v in [('x0', x0d), ('y', yd), ('t', td), ('eps', epsd),
                     ('xt', xtd)]:
            pooled[k].append(v)
        pooled['ci'] += [ci] * n_diag
    out = {}
    for k in ['x0', 'y', 't', 'eps', 'xt']:
        out[k] = torch.cat(pooled[k], dim=0)
    return out


def bucket_mask(t):
    return {'early': t < 6, 'mid': (t >= 6) & (t < 13), 'late': t >= 13}


def mse(a, b):
    return float(((a - b) ** 2).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    global gen_buffers
    gen = ConditionalDiffusionGenerator(
        feat_dim=500, num_classes=3, hidden_dim=256,
        num_steps=20, beta_start=1e-4, beta_end=0.5,
        output_bound='tanh').to(DEVICE)
    gen.load_state_dict(torch.load(args.ckpt, map_location=DEVICE,
                                   weights_only=False))
    gen_buffers = {k: v for k, v in gen.state_dict().items()}
    gen.eval()

    D = rebuild_diag()
    x0, y, t, eps, xt = D['x0'], D['y'], D['t'], D['eps'], D['xt']
    ab = gen.alpha_bars[t].unsqueeze(-1)
    n = t.shape[0]
    print(f"[S3] diag pool: {n} 节点 (10 客户端 × 128 固定), "
          f"t 分布={torch.bincount(t).tolist()}")

    # ---- S3-A: 四种 predictor ----
    eps_identity = xt / torch.sqrt(1.0 - ab)
    with torch.no_grad():
        eps_model = gen(xt, t, y)
    print("\n=== S3-A Analytical Baselines ===")
    print(f"{'Bucket':<8} {'Zero':>9} {'Identity':>10} {'OptScalar':>10} "
          f"{'Model':>9} {'Model/Identity':>14}")
    rows = {}
    for bname, mask in bucket_mask(t).items():
        xb, eb, tb, mb = xt[mask], eps[mask], t[mask], eps_model[mask]
        ab_b = gen.alpha_bars[tb].unsqueeze(-1)
        z = mse(torch.zeros_like(eb), eb)
        i = mse(xb / torch.sqrt(1.0 - ab_b), eb)
        # 最优标量线性: c = Σ(x_t·ε) / Σ(x_t²) (逐 bucket)
        c_opt = float((xb * eb).sum() / (xb.pow(2).sum() + 1e-12))
        s = mse(c_opt * xb, eb)
        m = mse(mb, eb)
        ratio = m / max(i, 1e-12)
        rows[bname] = dict(zero=z, identity=i, opt_scalar=s, model=m,
                           ratio=ratio)
        print(f"{bname:<8} {z:9.4f} {i:10.4f} {s:10.4f} {m:9.4f} "
              f"{ratio:14.1f}x")
    # 末 3 个 timestep 单独列出
    print("\n末 3 个 timestep (identity 应接近 0):")
    print(f"{'t':>3} {'alpha_bar':>10} {'identity MSE':>13} "
          f"{'model MSE':>10} {'model/identity':>15}")
    for tv in (17, 18, 19):
        mask = t == tv
        xb, eb, mb = xt[mask], eps[mask], eps_model[mask]
        abv = gen.alpha_bars[tv].item()
        i = mse(xb / math.sqrt(1.0 - abv), eb)
        m = mse(mb, eb)
        print(f"{tv:3d} {abv:10.5f} {i:13.5f} {m:10.4f} "
              f"{m / max(i, 1e-12):15.1f}x")

    # ---- S3-C: x0 reconstruction ----
    print("\n=== S3-C x0 Reconstruction (amplification check) ===")
    x0_oracle = (xt - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
    oracle_err = ((x0_oracle - x0) ** 2).mean().item()
    print(f"oracle reconstruction MSE vs x0: {oracle_err:.3e} "
          f"(应到浮点误差量级)")
    x0_hat = (xt - torch.sqrt(1.0 - ab) * eps_model) / torch.sqrt(ab)
    amp = torch.sqrt((1.0 - ab) / ab)          # 理论放大因子
    pred_err = (amp * (eps - eps_model)) ** 2  # 理论逐元素误差
    actual_err = (x0_hat - x0) ** 2
    print(f"{'Bucket':<8} {'eps RMSE':>9} {'amp(RMS)':>9} "
          f"{'pred x0 RMSE':>13} {'actual x0 RMSE':>15} {'x0_hat σ':>9} "
          f"{'x0_hat med r':>12}")
    for bname, mask in bucket_mask(t).items():
        e_rmse = float(((eps[mask] - eps_model[mask]) ** 2).mean().item()) ** 0.5
        a_rms = float((amp[mask] ** 2).mean().item()) ** 0.5
        p_rmse = float(pred_err[mask].mean().item()) ** 0.5
        a_rmse = float(actual_err[mask].mean().item()) ** 0.5
        xh = x0_hat[mask]
        print(f"{bname:<8} {e_rmse:9.4f} {a_rms:9.3f} {p_rmse:13.4f} "
              f"{a_rmse:15.4f} {xh.std().item():9.4f} "
              f"{xh.norm(dim=1).median().item():12.4f}")
    json.dump({'S3A': rows,
               'oracle_recon_mse': oracle_err},
              open('runs/accuracy_benchmark/DIAG_S3.json', 'w'), indent=1)
    print("\nS3_DONE")


if __name__ == '__main__':
    main()
