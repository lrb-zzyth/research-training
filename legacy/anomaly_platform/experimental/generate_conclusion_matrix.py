"""
生成 conclusion_matrix.csv (阶段三): 每个声明附证据、数据规模、效应量、CI、
校正后 p 值、实际意义与状态。
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, 'runs')


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = []

    def add(claim, evidence, datasets, n_part, n_seed, effect, ci,
            p_holm, practical, status, limitations):
        rows.append({'claim': claim, 'evidence': evidence,
                     'datasets': datasets, 'number_of_partitions': n_part,
                     'number_of_model_seeds': n_seed, 'effect_size': effect,
                     'confidence_interval': ci, 'corrected_p_value': p_holm,
                     'practical_significance': practical, 'status': status,
                     'limitations': limitations})

    # 1) enriched 复现 (blocked paired)
    bp = load_csv(os.path.join(RUNS, 'ckr_enriched_independent_replication',
                               'blocked_paired_comparisons.csv'))
    ov = next((r for r in bp if r.get('partition') == 'OVERALL'), None)
    if ov and ov.get('n_pairs', '0') != '0' and ov.get('n_pairs') not in (None, ''):
        add('dynamic CKR 在独立 partition 块上的复现',
            f"5 partition blocks × {ov['n_pairs']} 严格配对 (frozen split, "
            f"data/init hash 一致)",
            'Cora', 5, ov['n_pairs'],
            f"mean_diff={ov['mean_diff']}pp",
            f"model-seed CI [{ov['ci95_low']}, {ov['ci95_high']}], "
            f"partition-cluster CI [{ov['partition_cluster_ci_low']}, "
            f"{ov['partition_cluster_ci_high']}]",
            ov.get('wilcoxon_p', ''), '见阈值分析',
            'supported' if (ov.get('partition_cluster_ci_low') and
                            float(ov.get('partition_cluster_ci_low', 0)) > 0
                            and float(ov.get('mean_diff', 0)) >= 0.25)
            else 'conditionally_supported'
            if (ov.get('partition_cluster_ci_low') and
                float(ov.get('partition_cluster_ci_low', 0)) > 0)
            else 'unsupported',
            '仅 Cora; enriched 划分改变了 Louvain 基础抽取')
    else:
        add('dynamic CKR 在独立 partition 块上的复现', '结果未就绪',
            'Cora', 0, 0, '', '', '', '', 'pending', '')

    # 2) 蒸馏因果链
    chain_mr = load_csv(os.path.join(RUNS, 'distillation_causal_chain',
                                     'main_results.csv'))
    by_name = {}
    for r in chain_mr:
        by_name[r['name']] = r
    if by_name:
        d0 = by_name.get('D0', {}).get('pooled_pr_auc_mean')
        d7 = by_name.get('D7', {}).get('pooled_pr_auc_mean')
        d3 = by_name.get('D3', {}).get('pooled_pr_auc_mean')
        if d0 and d7:
            add('蒸馏 (static CKR) 相对 B3 的增益',
                f"D7({d7}) vs D0({d0}) pooled PR-AUC",
                'Cora', 1, 5, f"{float(d7) - float(d0):+.2f}pp",
                '见 paired_comparisons.csv', '见 paired_comparisons.csv',
                '≥0.25pp 才算实际意义', 'pending', '单 partition')
        if d0 and d3:
            add('动态 CKR 蒸馏相对 B3 的增益',
                f"D3({d3}) vs D0({d0})", 'Cora', 1, 5,
                f"{float(d3) - float(d0):+.2f}pp", '', '', '', 'pending', '')

    # 3) 公平性
    fair_pareto = load_csv(os.path.join(RUNS, 'worst_client_fairness',
                                        'fairness_pareto.csv'))
    if fair_pareto:
        add('公平性机制对最差客户端的影响',
            'F0-F5 worst/pooled Pareto (fairness_pareto.csv)',
            'Cora', 1, 5, '', '', '', '', 'pending',
            '需同时检查 pooled 损失')

    # 4) 信息诊断
    diag_mr = load_csv(os.path.join(RUNS, 'ckr_information_diagnostics',
                                    'main_results.csv'))
    if diag_mr:
        add('shuffled/inverse 权重对照', '见 main_results.csv',
            'Cora', 1, 5, '', '', '', '', 'pending', '')

    # 5) 支持度网格
    grid_mr = load_csv(os.path.join(RUNS, 'ckr_support_gain_grid',
                                    'main_results.csv'))
    if grid_mr:
        add('支持度 × 动态收益关系', 'grid main_results.csv',
            'Cora', 1, 3, '', '', '', '', 'pending', '')

    # 6) 跨数据集
    ext = load_csv(os.path.join(RUNS, 'dataset_transfer_citeseer',
                                'blocked_external_resources.csv'))
    add('CiteSeer 跨数据集结论', '外部数据阻塞 (网络不可达)',
        'CiteSeer', 0, 0, '', '', '', '', 'blocked',
        f"{len(ext)} 条 external_blocked 记录")

    out = os.path.join(ROOT, 'conclusion_matrix.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} claims)")
    for r in rows:
        print(f"  [{r['status']}] {r['claim']}")


if __name__ == '__main__':
    main()
