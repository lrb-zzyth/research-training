"""
Reliability Holdout Split 与标签映射后重划分

1. stratified_reliability_split
   从 train_idx 中按类别分层划分 fit_idx 和 reliability_idx。
   某类别样本太少时优先保留在 fit_idx, reliability 标记为 unavailable。

2. stratified_split / stratified_split_with_support
   按新标签重新分层划分 train/val/test (支持最小 support 约束与不可满足报告)。

3. anomaly_holdout_boost_split
   提高异常类在 validation 中的占比 (仅用于可靠性估计),
   不从 test 移动节点, 记录移动数量。

4. split_report
   生成每个客户端、每个类别的 fit/reliability support 与可用性报告。
"""
import torch


def _to_mask(idx, n, device):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


def stratified_reliability_split(data, train_idx, num_classes,
                                 holdout_ratio=0.2,
                                 min_support=3,
                                 split_seed=42):
    """
    从 train_idx 中分层划分 fit_idx 和 reliability_idx。

    规则:
      - 按类别比例划分 (holdout_ratio)
      - 某类别总样本 <= min_support 时全部保留在 fit_idx
      - 某类别划分后 fit < min_support 时缩小 reliability 份额
      - 不得为了 reliability_idx 抽走某类别唯一的训练样本
      - split_seed 决定划分 (与全局 RNG 无关, 可复现)

    Returns:
        fit_idx: bool mask [N]
        reliability_idx: bool mask [N]
        split_info: dict {class_id: {'total','fit','reliability'}}
    """
    device = data.x.device
    N = data.x.shape[0]
    y = data.y.cpu()

    if train_idx.dtype == torch.bool:
        train_nodes = torch.where(train_idx.cpu())[0]
    else:
        train_nodes = train_idx.cpu()

    fit_mask = torch.zeros(N, device=device, dtype=torch.bool)
    rel_mask = torch.zeros(N, device=device, dtype=torch.bool)
    info = {}

    g = torch.Generator()
    g.manual_seed(int(split_seed))

    for c in range(num_classes):
        class_nodes = train_nodes[y[train_nodes] == c]
        n_total = len(class_nodes)
        n_hold = 0
        if n_total > 0:
            n_hold = int(n_total * holdout_ratio)
            # 类别样本太少: 全部保留在 fit_idx
            if n_total <= min_support:
                n_hold = 0
            else:
                # 确保 fit 至少保留 min_support 个
                n_fit_min = min_support
                if n_total - n_hold < n_fit_min:
                    n_hold = max(0, n_total - n_fit_min)

        if n_hold > 0:
            perm = torch.randperm(n_total, generator=g)
            rel_idx = class_nodes[perm[:n_hold]]
            fit_idx = class_nodes[perm[n_hold:]]
            rel_mask[_to_mask(rel_idx, N, device)] = True
            fit_mask[_to_mask(fit_idx, N, device)] = True
        else:
            fit_mask[_to_mask(class_nodes, N, device)] = True

        info[c] = {
            'total': int(n_total),
            'fit': int(n_total - n_hold),
            'reliability': int(n_hold),
        }

    return fit_mask, rel_mask, info


def stratified_split(data, num_classes, train_ratio, val_ratio, seed=42):
    """
    标签映射后按新标签重新分层划分 train/val/test。

    Returns:
        train_mask, val_mask, test_mask: bool masks [N]
        split_info: dict {class_id: {'train','val','test'}}
    """
    tr, va, te, info, _ = stratified_split_with_support(
        data, num_classes, train_ratio, val_ratio,
        min_train_support=0, min_val_support=0, seed=seed, allow_infeasible=True)
    return tr, va, te, info


def stratified_split_with_support(data, num_classes, train_ratio, val_ratio,
                                  min_train_support=0, min_val_support=0,
                                  seed=42, allow_infeasible=False):
    """
    按类别分层划分 train/val/test, 支持最小 support 约束。

    某类别样本不足以满足 min_train_support/min_val_support 时:
      - 记录不可满足原因 (infeasible_report)
      - allow_infeasible=False 时抛出 ValueError (不静默伪造样本)
      - allow_infeasible=True 时按可行份额划分并记录原因

    Returns:
        train_mask, val_mask, test_mask: bool masks [N]
        split_info: dict {class_id: {'train','val','test'}}
        infeasible_report: list[str]
    """
    device = data.x.device
    N = data.x.shape[0]
    y = data.y.cpu()

    train_mask = torch.zeros(N, device=device, dtype=torch.bool)
    val_mask = torch.zeros(N, device=device, dtype=torch.bool)
    test_mask = torch.zeros(N, device=device, dtype=torch.bool)
    info = {}
    infeasible = []

    g = torch.Generator()
    g.manual_seed(int(seed))

    for c in range(num_classes):
        class_nodes = torch.where(y == c)[0]
        n_total = len(class_nodes)
        if n_total == 0:
            info[c] = {'train': 0, 'val': 0, 'test': 0}
            if min_train_support > 0 or min_val_support > 0:
                infeasible.append(f"class {c}: 无样本")
            continue
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        if n_train < min_train_support:
            infeasible.append(
                f"class {c}: train {n_train} < min_train_support {min_train_support}")
        if n_val < min_val_support:
            infeasible.append(
                f"class {c}: val {n_val} < min_val_support {min_val_support}")
        if not allow_infeasible and (n_train < min_train_support
                                     or n_val < min_val_support):
            raise ValueError(
                f"不可满足最小 support: class {c} 总样本 {n_total}, "
                f"train {n_train}/{min_train_support}, val {n_val}/{min_val_support}. "
                f"使用 --allow_support_infeasible 可继续 (将如实记录)。")

        perm = torch.randperm(n_total, generator=g)
        train_idx = class_nodes[perm[:n_train]]
        val_idx = class_nodes[perm[n_train:n_train + n_val]]
        test_idx = class_nodes[perm[n_train + n_val:]]

        train_mask[_to_mask(train_idx, N, device)] = True
        val_mask[_to_mask(val_idx, N, device)] = True
        test_mask[_to_mask(test_idx, N, device)] = True

        info[c] = {'train': int(n_train), 'val': int(n_val),
                   'test': int(n_test)}

    return train_mask, val_mask, test_mask, info, infeasible


def anomaly_holdout_boost_split(data, train_ratio, val_ratio, anomaly_val_ratio,
                                seed=42, min_train_support=0, allow_infeasible=False):
    """
    anomaly_binary: 提高异常类在 validation 中的占比 (仅用于可靠性估计)。

    规则:
      - test 先按类别份额切出, 不从 test 移动节点
      - val 中异常类目标占比 = anomaly_val_ratio (受可用异常数上限约束)
      - 其余节点进入 train
      - 所有划分受 seed 控制, train/val/test 两两不重叠

    Returns:
        train_mask, val_mask, test_mask
        split_info: {class_id: {'train','val','test'}}
        moved_report: {class_1: {'natural_val', 'boosted_val',
                                 'moved_from_train', 'infeasible_reason'}}
        infeasible: list[str]
    """
    device = data.x.device
    N = data.x.shape[0]
    y = data.y.cpu()

    train_mask = torch.zeros(N, device=device, dtype=torch.bool)
    val_mask = torch.zeros(N, device=device, dtype=torch.bool)
    test_mask = torch.zeros(N, device=device, dtype=torch.bool)
    info = {}
    moved = {}
    infeasible = []

    g = torch.Generator()
    g.manual_seed(int(seed))

    for c in range(2):
        class_nodes = torch.where(y == c)[0]
        n_total = len(class_nodes)
        if n_total == 0:
            info[c] = {'train': 0, 'val': 0, 'test': 0}
            continue
        perm = torch.randperm(n_total, generator=g)
        n_test = int(n_total * (1 - train_ratio - val_ratio))
        test_idx = class_nodes[perm[:n_test]]
        pool = class_nodes[perm[n_test:]]  # train+val 池
        test_mask[_to_mask(test_idx, N, device)] = True

        n_pool = len(pool)
        if c == 0:
            # 正常类: val 份额与 natural 相同
            n_val = min(int(n_total * val_ratio), n_pool)
            val_idx = pool[:n_val]
            train_idx = pool[n_val:]
            moved[f'class_{c}'] = {'natural_val': int(n_val),
                                   'boosted_val': int(n_val),
                                   'moved_from_train': 0,
                                   'infeasible_reason': None}
        else:
            # 异常类: val 目标占比 anomaly_val_ratio
            n_val_normal_est = int(n_total * val_ratio)
            n_val_target = int(anomaly_val_ratio / max(1e-8, 1 - anomaly_val_ratio)
                               * n_val_normal_est)
            n_val_natural = min(int(n_total * val_ratio), n_pool)
            reason = None
            if n_val_target > n_pool:
                reason = (f"anomaly 可用样本不足: 目标 val {n_val_target} > 池 {n_pool}")
                infeasible.append(reason)
                if not allow_infeasible:
                    raise ValueError(reason + ". 使用 --allow_support_infeasible 可继续。")
                n_val_target = n_pool
            if n_pool - n_val_target < min_train_support:
                reason = (f"boost 后 train 异常 {n_pool - n_val_target} "
                          f"< min_train_support {min_train_support}")
                infeasible.append(reason)
                if not allow_infeasible:
                    raise ValueError(reason + ". 使用 --allow_support_infeasible 可继续。")
                n_val_target = n_pool - min_train_support
            val_idx = pool[:n_val_target]
            train_idx = pool[n_val_target:]
            moved[f'class_{c}'] = {'natural_val': int(n_val_natural),
                                   'boosted_val': int(n_val_target),
                                   'moved_from_train': int(n_val_target - n_val_natural),
                                   'infeasible_reason': reason}

        val_mask[_to_mask(val_idx, N, device)] = True
        train_mask[_to_mask(train_idx, N, device)] = True
        info[c] = {'train': int(len(train_idx)), 'val': int(len(val_idx)),
                   'test': int(len(test_idx))}

    return train_mask, val_mask, test_mask, info, moved, infeasible


def split_report(fit_mask, rel_mask, y, num_classes, min_support):
    """
    输出每个类别: fit support / reliability support / 动态指标是否可用 / 回退原因。

    Returns:
        dict {class_id: {'fit': int, 'reliability': int,
                          'available': bool, 'fallback_reason': str}}
    """
    report = {}
    y_cpu = y.cpu()
    for c in range(num_classes):
        n_fit = int((fit_mask.cpu() & (y_cpu == c)).sum())
        n_rel = int((rel_mask.cpu() & (y_cpu == c)).sum())
        available = n_rel >= min_support
        if n_rel == 0:
            reason = 'no_reliability_samples' if n_fit > 0 else 'class_absent'
        elif not available:
            reason = f'insufficient_support_{n_rel}<{min_support}'
        else:
            reason = ''
        report[c] = {
            'fit': n_fit,
            'reliability': n_rel,
            'available': available,
            'fallback_reason': reason,
        }
    return report
