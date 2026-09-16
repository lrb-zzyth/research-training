"""
公开代理数据 DDPM 预训练 (可选, --generator_init proxy_pretrained)

只允许使用公开代理数据或合成代理特征, 不得读取任何客户端私有子图。
合成代理数据 (--proxy_dataset synthetic) 为逐类高斯簇, 完全本地生成, 不联网。

标准 DDPM 噪声预测训练:
  x_t = sqrt(alpha_bar_t) * x0_public + sqrt(1 - alpha_bar_t) * epsilon
  L_diff = MSE(epsilon_theta(x_t, t, [label]), epsilon)

特征维度适配:
  - proxy_feature_dim != target_feature_dim 时使用显式 FeatureAdapter
  - 不得静默截断/补零/reshape
  - adapter 参数保存进 checkpoint

类别语义:
  - 公开类别与目标类别语义不一致时使用 unconditional 预训练 (默认建议)
  - unconditional 模式生成器使用 num_classes=1, 不假设类别 ID 语义
  - 联邦阶段通过教师语义损失注入目标类别信息

proxy checkpoint 保存:
  generator state / feature adapter state / optimizer state /
  proxy dataset 名称 / 特征维度 / diffusion schedule / 模式 / num_classes / seed
"""
import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import ConditionalDiffusionGenerator


# =====================================================================
#  FeatureAdapter: 公开代理特征 -> 目标 GNN 输入特征
# =====================================================================

class FeatureAdapter(nn.Module):
    """显式特征适配器。维度相同为 Identity, 不同时用 MLP, 不静默变换。"""

    def __init__(self, proxy_dim, target_dim, hidden_dim=128):
        super().__init__()
        self.proxy_dim = int(proxy_dim)
        self.target_dim = int(target_dim)
        if proxy_dim == target_dim:
            self.net = nn.Identity()
            self.is_identity = True
        else:
            self.net = nn.Sequential(
                nn.Linear(proxy_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, target_dim),
            )
            self.is_identity = False

    def forward(self, x):
        return self.net(x)


# =====================================================================
#  合成代理数据 (逐类高斯簇, 无需联网)
# =====================================================================

def make_synthetic_proxy_data(n_samples, feat_dim, num_classes, seed=0, scale=3.0):
    """
    合成代理特征: 每类一个单位范数中心 + 高斯噪声。
    类别无真实语义, 仅作为 DDPM 预训练的可信特征流形。
    """
    rng = np.random.RandomState(int(seed))
    mus = rng.randn(num_classes, feat_dim).astype(np.float32)
    mus /= (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-8)
    labels = rng.randint(0, num_classes, size=n_samples)
    x = mus[labels] * scale + rng.randn(n_samples, feat_dim).astype(np.float32) * 0.5
    return torch.tensor(x, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# =====================================================================
#  参数
# =====================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--proxy_dataset', type=str, default='synthetic',
                    choices=['synthetic'], help='当前仅支持本地合成代理数据')
parser.add_argument('--proxy_n_samples', type=int, default=5000)
parser.add_argument('--proxy_feature_dim', type=int, default=1433)
parser.add_argument('--target_feature_dim', type=int, default=1433)
parser.add_argument('--proxy_pretrain_mode', type=str, default='unconditional',
                    choices=['unconditional', 'conditional'])
parser.add_argument('--proxy_epochs', type=int, default=50)
parser.add_argument('--proxy_lr', type=float, default=1e-3)
parser.add_argument('--proxy_batch_size', type=int, default=256)
parser.add_argument('--proxy_checkpoint', type=str, default='./proxy_ckpt/proxy_diffusion.pt')

# 生成器结构 (与联邦阶段一致)
parser.add_argument('--diffusion_steps', type=int, default=10)
parser.add_argument('--diffusion_hidden', type=int, default=256)
parser.add_argument('--diffusion_beta_start', type=float, default=1e-4)
parser.add_argument('--diffusion_beta_end', type=float, default=0.02)
parser.add_argument('--generator_output_bound', type=str, default='tanh',
                    choices=['tanh', 'clamp', 'none'])
parser.add_argument('--num_classes', type=int, default=7,
                    help='conditional 模式下代理类别数 (无语义, 仅结构)')
parser.add_argument('--gpu_id', type=str, default='0')


def train_proxy_diffusion(args, device):
    """标准 DDPM 噪声预测训练。返回 generator, adapter, 训练历史。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gen_num_classes = args.num_classes if args.proxy_pretrain_mode == 'conditional' else 1

    generator = ConditionalDiffusionGenerator(
        feat_dim=args.target_feature_dim,
        num_classes=gen_num_classes,
        hidden_dim=args.diffusion_hidden,
        num_steps=args.diffusion_steps,
        beta_start=args.diffusion_beta_start,
        beta_end=args.diffusion_beta_end,
        output_bound=args.generator_output_bound,
    ).to(device)
    adapter = FeatureAdapter(args.proxy_feature_dim, args.target_feature_dim).to(device)
    optimizer = torch.optim.Adam(
        list(generator.parameters()) + list(adapter.parameters()), lr=args.proxy_lr)

    x, y = make_synthetic_proxy_data(
        args.proxy_n_samples, args.proxy_feature_dim, args.num_classes, seed=args.seed)
    x, y = x.to(device), y.to(device)
    n = x.shape[0]

    betas = generator.betas  # [T]
    alpha_bars = generator.alpha_bars  # [T]

    history = []
    n_batches = max(1, math.ceil(n / args.proxy_batch_size))
    for epoch in range(args.proxy_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.proxy_batch_size:(b + 1) * args.proxy_batch_size]
            if len(idx) == 0:
                continue
            x0_proxy = x[idx]
            y_batch = y[idx]

            x0 = adapter(x0_proxy)  # 显式适配到目标维度
            B = x0.shape[0]
            t = torch.randint(0, args.diffusion_steps, (B,), device=device)

            noise = torch.randn_like(x0)
            sqrt_ab = alpha_bars[t].unsqueeze(1) ** 0.5
            sqrt_1m = (1.0 - alpha_bars[t]).clamp(min=0.0).unsqueeze(1) ** 0.5
            x_t = sqrt_ab * x0 + sqrt_1m * noise

            if args.proxy_pretrain_mode == 'conditional':
                labels_gen = y_batch
            else:
                labels_gen = torch.zeros(B, device=device, dtype=torch.long)

            eps_pred = generator(x_t, t, labels_gen)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        avg = epoch_loss / n
        history.append(avg)
        if (epoch + 1) % 10 == 0 or epoch == args.proxy_epochs - 1:
            print(f"[Proxy DDPM] epoch {epoch + 1}/{args.proxy_epochs}: "
                  f"L_diff={avg:.6f}")

    return generator, adapter, optimizer, history


def save_proxy_checkpoint(path, generator, adapter, optimizer, args, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'meta': {
            'proxy_dataset': args.proxy_dataset,
            'proxy_feature_dim': int(args.proxy_feature_dim),
            'target_feature_dim': int(args.target_feature_dim),
            'mode': args.proxy_pretrain_mode,
            'num_classes': int(args.num_classes),
            'gen_num_classes': int(generator.num_classes),
            'diffusion_steps': int(args.diffusion_steps),
            'diffusion_hidden': int(args.diffusion_hidden),
            'beta_start': float(args.diffusion_beta_start),
            'beta_end': float(args.diffusion_beta_end),
            'output_bound': generator.output_bound,
            'seed': int(args.seed),
            'proxy_epochs': int(args.proxy_epochs),
        },
        'generator_state': generator.state_dict(),
        'adapter_state': adapter.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'history': history,
    }
    torch.save(ckpt, path)
    print(f"[Proxy DDPM] checkpoint saved: {path}")


def load_proxy_pretrained_generator(checkpoint_path, generator, device='cpu'):
    """
    联邦训练加载代理 checkpoint。

    校验:
      1. target_feature_dim 必须等于目标生成器 feat_dim, 否则明确报错
      2. conditional 模式必须 num_classes 一致 (类别 ID 语义对齐)
    unconditional 模式下类别嵌入允许维度不同 (联邦阶段注入类别语义)。

    Returns:
        (ckpt dict, skipped_keys list)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt['meta']

    if int(meta['target_feature_dim']) != generator.feat_dim:
        raise RuntimeError(
            f"Proxy checkpoint 特征维度不兼容: checkpoint target_feature_dim="
            f"{meta['target_feature_dim']}, 目标生成器 feat_dim={generator.feat_dim}. "
            f"请重新预训练或更换代理 checkpoint。")

    if meta['mode'] == 'conditional' and int(meta['num_classes']) != generator.num_classes:
        raise RuntimeError(
            f"Proxy checkpoint conditional 模式 num_classes={meta['num_classes']} "
            f"与目标生成器 {generator.num_classes} 不一致。"
            f"类别 ID 语义可能不对齐, 请使用 unconditional 预训练。")

    if int(meta['diffusion_steps']) != generator.num_steps:
        print(f"  [Proxy] ⚠ diffusion_steps 不同 (proxy={meta['diffusion_steps']}, "
              f"target={generator.num_steps}), 时间嵌入归一化尺度存在近似差异")

    # 只加载形状匹配的参数 (unconditional 时 class_embed 允许跳过)
    own = generator.state_dict()
    proxy_state = ckpt['generator_state']
    to_load = {k: v for k, v in proxy_state.items()
               if k in own and own[k].shape == v.shape}
    skipped = [k for k in proxy_state if k not in to_load]
    generator.load_state_dict(to_load, strict=False)
    if skipped:
        print(f"  [Proxy] 跳过不匹配参数: {skipped}")

    return ckpt, skipped


def main(argv=None):
    args = parser.parse_args(argv)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')
    print(f"[Proxy DDPM] device={device} mode={args.proxy_pretrain_mode} "
          f"proxy_dim={args.proxy_feature_dim} target_dim={args.target_feature_dim}")

    generator, adapter, optimizer, history = train_proxy_diffusion(args, device)
    save_proxy_checkpoint(args.proxy_checkpoint, generator, adapter, optimizer,
                          args, history)

    print(f"[Proxy DDPM] 训练完成. L_diff 首末: "
          f"{history[0]:.6f} -> {history[-1]:.6f}")
    return args.proxy_checkpoint


if __name__ == "__main__":
    main()
