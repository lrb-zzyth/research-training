"""
Checkpoint Manager for FedTAD

保存/恢复:
- global_model, generator, optimizers
- DynamicCKRTracker state
- RNG states (python/random/numpy/torch CPU/CUDA)
- 全部关键超参数 (args) + 任务元数据 (task_mode/num_classes/feat_dim/生成器配置)

规则:
- 验证主指标提升且非 NaN 时保存 best.pt
- --save_last_checkpoint 时每轮保存 last.pt
- 训练结束默认加载 best.pt 做最终 test
- 从 checkpoint 恢复训练时校验元数据, 不兼容则明确报错
"""
import os
import json
import glob
import torch
import random
import numpy as np


def _build_meta(args=None, task_mode='multiclass', num_classes=7, feat_dim=None,
                normal_classes='', anomaly_classes='', selection_metric='macro_f1',
                generator_cfg=None):
    return {
        'task_mode': task_mode,
        'num_classes': int(num_classes),
        'feat_dim': int(feat_dim) if feat_dim is not None else None,
        'normal_classes': normal_classes,
        'anomaly_classes': anomaly_classes,
        'selection_metric': selection_metric,
        'generator_cfg': generator_cfg or {},
    }


def save_checkpoint(save_dir, filename, global_model, generator=None,
                    global_optimizer=None, gen_optimizer=None,
                    ckr_tracker=None, rng_state=None, args=None,
                    round_idx=0, best_metric=0.0, task_mode='multiclass',
                    num_classes=7, feat_dim=None, normal_classes='',
                    anomaly_classes='', selection_metric='macro_f1',
                    generator_cfg=None, is_best=True):
    """保存完整 checkpoint (模型 + 优化器 + CKR tracker + RNG + 元数据)。"""
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        'round': round_idx,
        'best_metric': best_metric,
        'meta': _build_meta(args, task_mode, num_classes, feat_dim,
                            normal_classes, anomaly_classes, selection_metric,
                            generator_cfg),
        'global_model_state': global_model.state_dict() if global_model else None,
        'global_optimizer_state': global_optimizer.state_dict() if global_optimizer else None,
    }
    if generator is not None:
        ckpt['generator_state'] = generator.state_dict()
    if gen_optimizer is not None:
        ckpt['gen_optimizer_state'] = gen_optimizer.state_dict()
    if ckr_tracker is not None:
        ckpt['ckr_tracker_state'] = ckr_tracker.state_dict()
    if rng_state is not None:
        ckpt['rng_state'] = rng_state
    if args is not None:
        ckpt['args'] = vars(args) if hasattr(args, '__dict__') else str(args)

    # RNG states
    ckpt['python_random_state'] = random.getstate()
    ckpt['numpy_random_state'] = np.random.get_state()
    ckpt['torch_cpu_rng_state'] = torch.get_rng_state()
    if torch.cuda.is_available():
        try:
            ckpt['torch_cuda_rng_state'] = torch.cuda.get_rng_state()
            ckpt['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass

    save_name = filename if filename else ('best.pt' if is_best else f'round_{round_idx}.pt')
    path = os.path.join(save_dir, save_name)
    torch.save(ckpt, path)

    meta_path = os.path.join(save_dir, f'{save_name}.meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'round': round_idx,
            'best_metric': best_metric,
            'task_mode': task_mode,
            'num_classes': int(num_classes),
            'is_best': is_best,
            'file': save_name,
        }, f)
    return path


def validate_checkpoint_meta(ckpt, expected_meta=None):
    """
    校验 checkpoint 元数据。不兼容时抛出 RuntimeError (明确报错)。

    expected_meta: dict 包含 task_mode / num_classes / feat_dim (可选)
    """
    if expected_meta is None:
        return
    meta = ckpt.get('meta', {})
    for key, exp in expected_meta.items():
        if exp is None:
            continue
        got = meta.get(key)
        if key == 'num_classes' and got is not None:
            exp, got = int(exp), int(got)
        if got is not None and got != exp:
            raise RuntimeError(
                f"Checkpoint 元数据不兼容: 期望 {key}={exp}, checkpoint 为 {got}. "
                f"请使用匹配的任务配置或重新训练。")


def load_checkpoint(path, global_model=None, generator=None,
                    global_optimizer=None, gen_optimizer=None,
                    ckr_tracker=None, expected_meta=None,
                    restore_rng=True, map_location='cpu'):
    """
    加载 checkpoint。默认恢复 RNG 状态, 保证恢复后训练连续性。

    Returns:
        ckpt dict (包含 'round', 'meta' 等)
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    validate_checkpoint_meta(ckpt, expected_meta)

    if global_model is not None and ckpt.get('global_model_state'):
        global_model.load_state_dict(ckpt['global_model_state'])
    if generator is not None and ckpt.get('generator_state'):
        generator.load_state_dict(ckpt['generator_state'])
    if global_optimizer is not None and ckpt.get('global_optimizer_state'):
        global_optimizer.load_state_dict(ckpt['global_optimizer_state'])
    if gen_optimizer is not None and ckpt.get('gen_optimizer_state'):
        gen_optimizer.load_state_dict(ckpt['gen_optimizer_state'])
    if ckr_tracker is not None and ckpt.get('ckr_tracker_state'):
        ckr_tracker.load_state_dict(ckpt['ckr_tracker_state'])

    if restore_rng:
        # 注意: torch.load(map_location='cuda') 会把 RNG 状态张量也搬到 GPU,
        # set_rng_state 需要 CPU ByteTensor, 这里显式 .cpu()
        if ckpt.get('torch_cpu_rng_state') is not None:
            torch.set_rng_state(ckpt['torch_cpu_rng_state'].cpu())
        if ckpt.get('python_random_state') is not None:
            random.setstate(ckpt['python_random_state'])
        if ckpt.get('numpy_random_state') is not None:
            np.random.set_state(ckpt['numpy_random_state'])
        if ckpt.get('torch_cuda_rng_state') is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(ckpt['torch_cuda_rng_state'].cpu())
                if ckpt.get('torch_cuda_rng_state_all') is not None:
                    torch.cuda.set_rng_state_all(
                        [s.cpu() for s in ckpt['torch_cuda_rng_state_all']])
            except Exception:
                pass

    return ckpt


def find_best_checkpoint(checkpoint_dir):
    """返回 best.pt 路径, 如果存在。"""
    best_path = os.path.join(checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        return best_path
    return None


def find_last_checkpoint(checkpoint_dir):
    """返回最新的 round_N.pt 路径。"""
    rounds = []
    for f in glob.glob(os.path.join(checkpoint_dir, 'round_*.pt')):
        try:
            r = int(os.path.basename(f).replace('round_', '').replace('.pt', ''))
            rounds.append((r, f))
        except ValueError:
            continue
    if not rounds:
        return None
    rounds.sort(key=lambda x: x[0])
    return rounds[-1][1]


def find_final_checkpoint(checkpoint_dir):
    """训练结束加载哪个 checkpoint: 优先 best.pt, 否则最新 round_N.pt。"""
    best = find_best_checkpoint(checkpoint_dir)
    if best is not None:
        return best
    return find_last_checkpoint(checkpoint_dir)
