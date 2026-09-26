"""
实验统计工具: mean±std / 配对差值 / 95% CI / wins / Wilcoxon。

约定: NaN 视为缺失 (unavailable), 聚合时从该指标统计中排除并记录有效数。
"""
import math
import numpy as np
from scipy import stats


def clean(values):
    """移除 NaN/None (unavailable), 返回 (列表, 有效数量)。"""
    vals = []
    for v in values:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(float('nan'))
    valid = [v for v in vals if not math.isnan(v)]
    return valid, len(valid)


def summarize(values):
    """
    mean ± std (含 95% 置信区间与 n)。
    Returns: dict(mean, std, n, ci95_low, ci95_high, sem) 或全 NaN。
    """
    valid, n = clean(values)
    if n == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0,
                'ci95_low': float('nan'), 'ci95_high': float('nan'),
                'sem': float('nan')}
    arr = np.array(valid)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    t = stats.t.ppf(0.975, df=max(1, n - 1))
    return {'mean': mean, 'std': std, 'n': n,
            'ci95_low': mean - t * sem, 'ci95_high': mean + t * sem,
            'sem': sem}


def paired_diff(full_values, baseline_values, seed_list=None):
    """
    配对差值 d_s = full_s - baseline_s (同种子配对)。

    Returns: dict(mean_diff, std_diff, ci95_low, ci95_high, n_pairs,
                   wins, losses, ties, wilcoxon_p, per_seed)
    """
    valid, n = clean(full_values)
    if n == 0:
        return {'mean_diff': float('nan'), 'std_diff': float('nan'),
                'ci95_low': float('nan'), 'ci95_high': float('nan'),
                'n_pairs': 0, 'wins': 0, 'losses': 0, 'ties': 0,
                'wilcoxon_p': float('nan'), 'per_seed': {}}
    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')

    diffs = []
    wins = losses = ties = 0
    per_seed = {}
    for i, f in enumerate(full_values):
        b = baseline_values[i] if i < len(baseline_values) else float('nan')
        fv, bv = _to_float(f), _to_float(b)
        if math.isnan(fv) or math.isnan(bv):
            continue
        d = fv - bv
        diffs.append(d)
        if seed_list is not None:
            per_seed[str(seed_list[i])] = {'full': fv, 'baseline': bv, 'diff': d}
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
        else:
            ties += 1
    if not diffs:
        return {'mean_diff': float('nan'), 'std_diff': float('nan'),
                'ci95_low': float('nan'), 'ci95_high': float('nan'),
                'n_pairs': 0, 'wins': wins, 'losses': losses, 'ties': ties,
                'wilcoxon_p': float('nan'), 'per_seed': per_seed}
    arr = np.array(diffs)
    mean_d = float(arr.mean())
    std_d = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std_d / math.sqrt(len(arr))
    t = stats.t.ppf(0.975, df=max(1, len(arr) - 1))
    p = float('nan')
    try:
        if len(arr) >= 3 and np.any(arr != 0):
            _, p = stats.wilcoxon(arr)
    except Exception:
        p = float('nan')
    return {'mean_diff': mean_d, 'std_diff': std_d,
            'ci95_low': mean_d - t * sem, 'ci95_high': mean_d + t * sem,
            'n_pairs': len(arr), 'wins': wins, 'losses': losses, 'ties': ties,
            'wilcoxon_p': float(p), 'per_seed': per_seed}


def fmt_mean_std(values, ndigits=2):
    """格式化 'mean ± std (n=N)'。"""
    s = summarize(values)
    if s['n'] == 0:
        return 'unavailable'
    return f"{s['mean']:.{ndigits}f} ± {s['std']:.{ndigits}f} (n={s['n']})"


def paired_bootstrap_ci(full_values, baseline_values, seed_list=None,
                        n_boot=2000, alpha=0.05, seed=0):
    """
    配对 bootstrap 95% CI (按种子对重采样)。
    Returns: dict(mean_diff, median_diff, ci95_low, ci95_high, n_pairs, seed)
    """
    rng = np.random.RandomState(int(seed))
    pairs = []
    for i, f in enumerate(full_values):
        b = baseline_values[i] if i < len(baseline_values) else float('nan')
        try:
            fv, bv = float(f), float(b)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(fv) or math.isnan(bv)):
            pairs.append((fv, bv))
    if len(pairs) < 2:
        return {'mean_diff': float('nan'), 'median_diff': float('nan'),
                'ci95_low': float('nan'), 'ci95_high': float('nan'),
                'n_pairs': len(pairs), 'seed': seed}
    arr = np.array(pairs)
    diffs = arr[:, 0] - arr[:, 1]
    boot_means = []
    n = len(pairs)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means.append(diffs[idx].mean())
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {'mean_diff': float(diffs.mean()), 'median_diff': float(np.median(diffs)),
            'ci95_low': lo, 'ci95_high': hi, 'n_pairs': n, 'seed': seed}


def holm_correction(p_values):
    """
    Holm-Bonferroni 多重比较校正。
    p_values: list of (name, p)
    Returns: list of (name, p, p_holm)
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i][1])
    m = len(p_values)
    out = [None] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        # Holm step-up: p_i * (m - rank), 并保证沿排序单调非递减
        adj = min(1.0, p_values[idx][1] * (m - rank))
        adj = max(adj, prev)
        prev = adj
        out[idx] = (p_values[idx][0], p_values[idx][1], adj)
    return out


def cliffs_delta(full_values, baseline_values):
    """
    配对 Cliff's delta (用配对差的符号对)。
    Returns: dict(delta, interpretation)
    """
    pairs = []
    for i, f in enumerate(full_values):
        b = baseline_values[i] if i < len(baseline_values) else float('nan')
        try:
            fv, bv = float(f), float(b)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(fv) or math.isnan(bv)):
            pairs.append((fv, bv))
    if not pairs:
        return {'delta': float('nan'), 'n': 0, 'interpretation': 'unavailable'}
    n = len(pairs)
    wins = sum(1 for f, b in pairs if f > b)
    losses = sum(1 for f, b in pairs if f < b)
    delta = (wins - losses) / n
    if abs(delta) < 0.147:
        interp = 'negligible'
    elif abs(delta) < 0.33:
        interp = 'small'
    elif abs(delta) < 0.474:
        interp = 'medium'
    else:
        interp = 'large'
    return {'delta': float(delta), 'n': n, 'interpretation': interp}


def sensitivity_analysis(full_values, baseline_values):
    """
    敏感性分析: leave-one-pair-out 的配对均值差范围。
    Returns: dict(min_diff, max_diff, range, n)
    """
    pairs = []
    for i, f in enumerate(full_values):
        b = baseline_values[i] if i < len(baseline_values) else float('nan')
        try:
            fv, bv = float(f), float(b)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(fv) or math.isnan(bv)):
            pairs.append(fv - bv)
    if not pairs:
        return {'min_diff': float('nan'), 'max_diff': float('nan'),
                'range': float('nan'), 'n': 0}
    if len(pairs) < 2:
        return {'min_diff': float('nan'), 'max_diff': float('nan'),
                'range': float('nan'), 'n': len(pairs)}
    loo = []
    total = sum(pairs)
    for i in range(len(pairs)):
        loo.append((total - pairs[i]) / (len(pairs) - 1))
    return {'min_diff': float(min(loo)), 'max_diff': float(max(loo)),
            'range': float(max(loo) - min(loo)), 'n': len(pairs)}


def blocked_paired_analysis(full_values, baseline_values, partition_ids,
                            model_seeds=None, n_boot=2000, seed=0):
    """
    blocked paired analysis (阶段三):
    full/baseline 按 (partition, model_seed) 配对。

    Returns:
      per_block: [{partition, n, mean_diff, wins}]
      overall: mean/std/median diff, model-seed CI (t),
               partition-cluster bootstrap CI, wilcoxon, sign test,
               cohen_dz, prob_improvement, prob_gain_gt {0.1,0.25,0.5,1.0},
               variance decomposition (partition/model_seed/residual)
    """
    import math as _m
    rng = np.random.RandomState(int(seed))
    pairs = []  # (partition, model_seed, diff)
    for i, f in enumerate(full_values):
        b = baseline_values[i] if i < len(baseline_values) else float('nan')
        try:
            fv, bv = float(f), float(b)
        except (TypeError, ValueError):
            continue
        if _m.isnan(fv) or _m.isnan(bv):
            continue
        p = partition_ids[i] if i < len(partition_ids) else None
        m = model_seeds[i] if model_seeds and i < len(model_seeds) else None
        pairs.append((p, m, fv - bv))

    if len(pairs) < 2:
        return {'per_block': [], 'overall': {
            'n_pairs': len(pairs), 'mean_diff': float('nan'),
            'median_diff': float('nan'), 'ci95_low': float('nan'),
            'ci95_high': float('nan'), 'partition_cluster_ci_low': float('nan'),
            'partition_cluster_ci_high': float('nan'), 'wilcoxon_p': float('nan'),
            'sign_test_p': float('nan'), 'cohen_dz': float('nan'),
            'prob_improvement': float('nan'), 'prob_gain_gt': {},
            'variance_partition': float('nan'), 'variance_model_seed': float('nan'),
            'variance_residual': float('nan'), 'n_partitions': 0}}

    diffs = np.array([d for _, _, d in pairs])
    # per-block
    per_block = []
    by_block = {}
    for p, m, d in pairs:
        by_block.setdefault(p, []).append(d)
    for p in sorted(by_block, key=lambda x: (x is None, x)):
        bd = np.array(by_block[p])
        per_block.append({'partition': p, 'n': len(bd),
                          'mean_diff': float(bd.mean()),
                          'wins': int((bd > 1e-9).sum())})
    n = len(diffs)
    mean_d = float(diffs.mean())
    median_d = float(np.median(diffs))
    std_d = float(diffs.std(ddof=1)) if n > 1 else 0.0
    t = stats.t.ppf(0.975, df=max(1, n - 1))
    sem = std_d / _m.sqrt(n)

    # partition-cluster bootstrap: 以 partition 为抽样单元
    blocks = sorted({p for p, _, _ in pairs if p is not None})
    block_diffs = {p: np.array([d for pp, _, d in pairs if pp == p])
                   for p in blocks}
    boot_means = []
    if len(blocks) >= 2:
        for _ in range(n_boot):
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            means = [block_diffs[b].mean() for b in chosen]
            boot_means.append(float(np.mean(means)))
        pcl = float(np.percentile(boot_means, 2.5))
        pch = float(np.percentile(boot_means, 97.5))
    else:
        pcl = pch = float('nan')

    wilcoxon_p = float('nan')
    try:
        if n >= 3 and np.any(diffs != 0):
            _, wilcoxon_p = stats.wilcoxon(diffs)
    except Exception:
        pass
    # sign test: binom
    sign_p = float('nan')
    n_pos = int((diffs > 1e-9).sum())
    n_neg = int((diffs < -1e-9).sum())
    if n_pos + n_neg >= 3:
        from scipy.stats import binomtest
        sign_p = float(binomtest(n_pos, n_pos + n_neg, 0.5).pvalue)
    cohen_dz = float(mean_d / std_d) if std_d > 0 else float('nan')
    prob_imp = n_pos / n
    thresholds = {str(t_): float((diffs > t_).mean())
                  for t_ in (0.1, 0.25, 0.5, 1.0)}

    # 方差分解: 单因素 ANOVA on diff ~ partition
    vp = vm = vr = float('nan')
    if len(blocks) >= 2:
        groups = [block_diffs[b] for b in blocks]
        n_all = sum(len(g) for g in groups)
        grand = float(np.concatenate(groups).mean())
        ss_between = sum(len(g) * (float(g.mean()) - grand) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        vp = ss_between / max(1, len(blocks) - 1)
        vr = ss_within / max(1, n_all - len(blocks))
        vm = float('nan')  # model_seed 方差需嵌套设计, 此处并入残差
    return {
        'per_block': per_block,
        'overall': {
            'n_pairs': n, 'mean_diff': mean_d, 'median_diff': median_d,
            'std_diff': std_d,
            'ci95_low': mean_d - t * sem, 'ci95_high': mean_d + t * sem,
            'partition_cluster_ci_low': pcl, 'partition_cluster_ci_high': pch,
            'wilcoxon_p': float(wilcoxon_p), 'sign_test_p': float(sign_p),
            'cohen_dz': cohen_dz, 'prob_improvement': prob_imp,
            'prob_gain_gt': thresholds,
            'variance_partition': vp, 'variance_model_seed': vm,
            'variance_residual': vr, 'n_partitions': len(blocks),
        }}


def mean_abs_round_change(ckr_records):
    """
    动态 CKR 轮间波动: ΔR^t = (1/KC) Σ|R_{k,c}^t - R_{k,c}^{t-1}|
    ckr_records: 按轮分组的 {round: {(k,c): ema}}
    Returns: list of (round, delta) 按轮排序。
    """
    rounds = sorted(ckr_records)
    out = []
    for t1, t2 in zip(rounds[:-1], rounds[1:]):
        r1, r2 = ckr_records[t1], ckr_records[t2]
        keys = set(r1) & set(r2)
        if not keys:
            continue
        delta = sum(abs(r2[k] - r1[k]) for k in keys) / len(keys)
        out.append((t2, delta))
    return out
