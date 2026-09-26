"""
生成器反向传播显存/时间微基准 (checkpoint_memory_scaling suite 的运行器)

对相同 seed 与相同输入, 对比 full 与 checkpointed 反向传播:
  - max_memory_allocated / max_memory_reserved (CUDA; CPU 下标记 gpu_available=false)
  - forward / backward / total step 时间
  - 生成器梯度范数、最终损失
  - 输出张量最大绝对差、梯度最大绝对差 (数值等价性)

无 CUDA 时不伪造显存数据: 只运行功能对比并标记 GPU unavailable。
"""
import argparse
import json
import os
import sys
import time
import math
import torch

# 仓库根目录 (experimental/benchmark_checkpointing.py -> 上一级), 供 import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ConditionalDiffusionGenerator


def run_mode(generator, labels, num_steps, mode, segments, device, measure_steps):
    """返回 (loss, grad_norm, fwd_time, bwd_time, step_time, peak_alloc, peak_reserved)。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_fwd = total_bwd = 0.0
    for step in range(measure_steps):
        generator.zero_grad()
        tf0 = time.time()
        fake_x = generator.differentiable_sample(
            labels, num_steps=num_steps, backprop_mode=mode,
            checkpoint_segments=segments)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_fwd += time.time() - tf0
        loss = fake_x.pow(2).mean()
        tb0 = time.time()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_bwd += time.time() - tb0
    total_step = time.time() - t0
    gn = math.sqrt(sum(p.grad.norm().item() ** 2
                       for p in generator.parameters() if p.grad is not None))
    peak_alloc = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    return {'loss': loss.item(), 'grad_norm': gn,
            'forward_time_sec': total_fwd / measure_steps,
            'backward_time_sec': total_bwd / measure_steps,
            'step_time_sec': total_step / measure_steps,
            'peak_memory_allocated_mb': (peak_alloc / 1024 ** 2
                                         if peak_alloc is not None else None),
            'peak_memory_reserved_mb': (peak_reserved / 1024 ** 2
                                        if peak_reserved is not None else None)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--diffusion_steps', type=int, default=10)
    ap.add_argument('--feat_dim', type=int, default=1433)
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--generator_backward_mode', type=str, default='full',
                    choices=['full', 'checkpointed'])
    ap.add_argument('--checkpoint_segments', type=int, default=1)
    ap.add_argument('--resource_warmup_steps', type=int, default=1)
    ap.add_argument('--resource_measure_steps', type=int, default=2)
    ap.add_argument('--output_json', type=str, default='')
    ap.add_argument('--gpu_id', type=str, default='0')
    opts = ap.parse_args(argv)

    torch.manual_seed(opts.seed)
    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')
    gpu_available = torch.cuda.is_available()

    labels = torch.randint(0, opts.num_classes, (opts.batch_size,), device=device)

    results = {'seed': opts.seed,
               'hidden_dim': opts.hidden_dim,
               'diffusion_steps': opts.diffusion_steps,
               'feat_dim': opts.feat_dim,
               'batch_size': opts.batch_size,
               'checkpoint_segments': opts.checkpoint_segments,
               'gpu_available': gpu_available,
               'device': str(device)}

    gen_full = ConditionalDiffusionGenerator(
        feat_dim=opts.feat_dim, num_classes=opts.num_classes,
        hidden_dim=opts.hidden_dim, num_steps=opts.diffusion_steps,
        output_bound='tanh').to(device)
    gen_ckpt = ConditionalDiffusionGenerator(
        feat_dim=opts.feat_dim, num_classes=opts.num_classes,
        hidden_dim=opts.hidden_dim, num_steps=opts.diffusion_steps,
        output_bound='tanh').to(device)
    gen_ckpt.load_state_dict(gen_full.state_dict())
    for p in gen_full.parameters():
        p.requires_grad_(True)
    for p in gen_ckpt.parameters():
        p.requires_grad_(True)

    # warmup (排除 CUDA 初始化/分配开销)
    for _ in range(opts.resource_warmup_steps):
        fx = gen_full.differentiable_sample(labels, num_steps=opts.diffusion_steps,
                                            backprop_mode='full')
        fx.pow(2).mean().backward()

    # 相同输入: 固定扩散噪声 (两种模式必须使用相同 x_T 与逐步噪声)
    torch.manual_seed(999)
    ref_noise = [torch.randn(opts.batch_size, opts.feat_dim, device=device)
                 for _ in range(opts.diffusion_steps)]

    def sample_with_fixed_noise(gen, mode):
        with torch.no_grad():
            x_t = torch.randn(opts.batch_size, opts.feat_dim, device=device)
        x_t = x_t.detach()
        for t in reversed(range(opts.diffusion_steps)):
            t_tensor = torch.full((opts.batch_size,), t, device=device, dtype=torch.long)
            eps_pred = gen(x_t, t_tensor, labels)
            alpha_t = gen.alphas[t]
            beta_t = gen.betas[t]
            alpha_bar_t = gen.alpha_bars[t]
            noise = ref_noise[t] if t > 0 else torch.zeros_like(x_t)
            mu = (1.0 / torch.sqrt(alpha_t + 1e-8)) * (
                x_t - (beta_t / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)) * eps_pred)
            sigma = torch.sqrt(beta_t + 1e-8)
            x_prev = mu + sigma * noise
            if mode == 'checkpointed':
                import torch.utils.checkpoint as cp
                # 数值等价性测试使用逐步重计算 (与训练路径一致)
                pass
            x_t = x_prev.detach()
        return x_t

    # 数值等价性: 相同固定噪声下的输出与梯度
    with torch.no_grad():
        fx_full = gen_full.differentiable_sample(
            labels, num_steps=opts.diffusion_steps, backprop_mode='full')
    # 用相同噪声序列重放: 通过可微路径对比
    torch.manual_seed(777)
    x_full, x_ckpt = None, None
    if opts.generator_backward_mode == 'full':
        results['mode'] = 'full'
    else:
        results['mode'] = 'checkpointed'
    # 主测量: 仅测目标模式
    res = run_mode(gen_full if results['mode'] == 'full' else gen_ckpt,
                   labels, opts.diffusion_steps, results['mode'],
                   opts.checkpoint_segments, device, opts.resource_measure_steps)
    results.update(res)

    # 数值等价性对比 (同一模型同一输入: full vs checkpointed 输出/梯度差)
    torch.manual_seed(4242)
    g1 = ConditionalDiffusionGenerator(
        feat_dim=opts.feat_dim, num_classes=opts.num_classes,
        hidden_dim=opts.hidden_dim, num_steps=opts.diffusion_steps,
        output_bound='tanh').to(device)
    g2 = ConditionalDiffusionGenerator(
        feat_dim=opts.feat_dim, num_classes=opts.num_classes,
        hidden_dim=opts.hidden_dim, num_steps=opts.diffusion_steps,
        output_bound='tanh').to(device)
    g2.load_state_dict(g1.state_dict())
    # 固定输入与噪声: 直接逐模式可微采样 (模型参数相同 -> 随机噪声也相同,
    # 只要 RNG 状态一致; 这里分别 reset RNG 保证完全相同的输入流)
    torch.manual_seed(31337)
    x1 = g1.differentiable_sample(labels, num_steps=opts.diffusion_steps,
                                  backprop_mode='full')
    loss1 = x1.pow(2).mean()
    loss1.backward()
    torch.manual_seed(31337)
    x2 = g2.differentiable_sample(labels, num_steps=opts.diffusion_steps,
                                  backprop_mode='checkpointed')
    loss2 = x2.pow(2).mean()
    loss2.backward()

    with torch.no_grad():
        out_diff = (x1 - x2).abs().max().item()
        grad_diffs = []
        for p1, p2 in zip(g1.parameters(), g2.parameters()):
            grad_diffs.append((p1.grad - p2.grad).abs().max().item())
        grad_diff = max(grad_diffs) if grad_diffs else 0.0
    results['numerical_equivalence'] = {
        'output_max_abs_diff': out_diff,
        'gradient_max_abs_diff': grad_diff,
        'loss_full': loss1.item(),
        'loss_checkpointed': loss2.item(),
        'loss_max_abs_diff': abs(loss1.item() - loss2.item()),
        'equivalence': ('exact' if out_diff == 0.0 and grad_diff == 0.0
                        else 'allclose' if out_diff < 1e-5 and grad_diff < 1e-5
                        else 'not_equivalent'),
    }
    # 梯度范数 (full vs checkpointed, 同一输入)
    gn1 = math.sqrt(sum(p.grad.norm().item() ** 2 for p in g1.parameters()
                        if p.grad is not None))
    gn2 = math.sqrt(sum(p.grad.norm().item() ** 2 for p in g2.parameters()
                        if p.grad is not None))
    results['numerical_equivalence']['grad_norm_full'] = gn1
    results['numerical_equivalence']['grad_norm_checkpointed'] = gn2

    if not gpu_available:
        results['note'] = ('GPU unavailable: memory figures omitted (no fabrication); '
                           'functional/numerical comparison only')
        for k in ('peak_memory_allocated_mb', 'peak_memory_reserved_mb'):
            results[k] = None

    if opts.output_json:
        os.makedirs(os.path.dirname(opts.output_json) or '.', exist_ok=True)
        with open(opts.output_json, 'w') as f:
            json.dump(results, f, indent=1, default=str)
    print(json.dumps(results, indent=1, default=str))
    return results


if __name__ == '__main__':
    main()
