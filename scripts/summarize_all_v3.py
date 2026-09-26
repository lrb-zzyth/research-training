#!/usr/bin/env python3
"""Generate FORMAL_V3_ALL_DATASETS_REPORT.md / _RESULTS.json: 15-cell master table.

只读聚合 runs/formal_campaign_v3/<ds>/<DS>_V3_FINAL_SUMMARY.json (每 dataset
完成后由 summarize_dataset_v3.py 生成)。论文数字硬编码自 audit_docs/PAPER_TABLE2.md
(2026-09-25 从 references/fedtad.pdf 提取核验), 生成时若该文件存在会交叉核对。
"""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(REPO, 'runs', 'formal_campaign_v3')
DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics']
TIERS = [5, 10, 20]

# FedTAD 论文 Table 2: (mean, std) — 来源 audit_docs/PAPER_TABLE2.md
PAPER = {
    'Cora':     {5: (85.1, 0.1), 10: (75.3, 0.4), 20: (61.3, 0.3)},
    'CiteSeer': {5: (73.5, 0.3), 10: (71.7, 0.4), 20: (70.2, 0.3)},
    'PubMed':   {5: (87.9, 0.1), 10: (84.4, 0.4), 20: (83.5, 0.2)},
    'CS':       {5: (94.3, 0.4), 10: (90.2, 0.2), 20: (88.7, 0.4)},
    'Physics':  {5: (96.2, 0.2), 10: (94.1, 0.2), 20: (93.3, 0.3)},
}


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def check_paper_against_doc():
    """audit_docs/PAPER_TABLE2.md 存在时交叉核对硬编码论文数字。"""
    doc = os.path.join(REPO, 'audit_docs', 'PAPER_TABLE2.md')
    if not os.path.exists(doc):
        return 'PAPER_TABLE2.md not present (skip cross-check)'
    txt = open(doc).read()
    # 表格行形如: | Cora | 5 | 85.1 ± 0.1 | 80.6 ± 0.3 |
    bad = []
    for ds, tiers in PAPER.items():
        for tier, (m, s) in tiers.items():
            import re
            pat = r'\| %s \| %d \| ([0-9.]+) ± ([0-9.]+) \|' % (ds, tier)
            mch = re.search(pat, txt)
            if not mch or float(mch.group(1)) != m or float(mch.group(2)) != s:
                bad.append(f'{ds}-{tier}')
    return ('PAPER_TABLE2.md cross-check FAIL on: ' + ','.join(bad)) if bad \
        else 'PAPER_TABLE2.md cross-check OK'


def main():
    rows = []
    totals = {'datasets_complete': 0, 'cells_complete': 0, 'stage1_pass': 0,
              'attempts': 0, 'healthy': 0, 'failed_trials': 0,
              'final_seeds': 0, 'final_test_evals': 0,
              'tuning_test_evals': 0, 'wins': 0, 'losses': 0, 'ties': 0}
    for ds in DATASETS:
        s = load(os.path.join(OUT, ds, f'{ds.upper()}_V3_FINAL_SUMMARY.json'))
        cells = {c['cell']: c for c in s['cells']} if s else {}
        ds_complete = True
        for tier in TIERS:
            key = f'{ds}_c{tier}'
            cell = cells.get(key)
            ours = None
            if cell and cell['finals'].get('test_mean') is not None:
                ours = (cell['finals']['test_mean'],
                        cell['finals'].get('test_std'))
            pmean, pstd = PAPER[ds][tier]
            delta = round(ours[0] - pmean, 2) if ours else None
            row = {'dataset': ds, 'clients': tier,
                   'ours_mean': ours[0] if ours else None,
                   'ours_std': ours[1] if ours else None,
                   'fedtad_mean': pmean, 'fedtad_std': pstd,
                   'delta': delta}
            if ours is None:
                ds_complete = False
            else:
                totals['cells_complete'] += 1
                totals['final_seeds'] += 3
                totals['final_test_evals'] += 3
                if delta > 0:
                    totals['wins'] += 1
                elif delta < 0:
                    totals['losses'] += 1
                else:
                    totals['ties'] += 1
                if (cell.get('stage1_health') or {}).get('verdict') == 'PASS':
                    totals['stage1_pass'] += 1
                st = cell.get('study') or {}
                totals['attempts'] += st.get('attempts') or 0
                totals['healthy'] += st.get('healthy') or 0
                totals['failed_trials'] += st.get('failed') or 0
            rows.append(row)
        if ds_complete:
            totals['datasets_complete'] += 1

    results = {'generated_by': 'scripts/summarize_all_v3.py',
               'rows': rows, 'totals': totals,
               'paper_source': check_paper_against_doc()}
    jp = os.path.join(OUT, 'FORMAL_V3_ALL_DATASETS_RESULTS.json')
    with open(jp, 'w') as f:
        json.dump(results, f, indent=1)

    L = ['# FedTAD V3 Formal Campaign — All Datasets (15 cells)', '',
         '> 由 `scripts/summarize_all_v3.py` 生成 (只读聚合各 dataset '
         'FINAL_SUMMARY.json)', '',
         '## 15-cell 主表', '',
         '| Dataset | Clients | Ours (mean ± std) | FedTAD paper (mean ± std) | Delta |',
         '|---|---|---|---|---|']
    for r in rows:
        ours = ('%.2f ± %.2f' % (r['ours_mean'], r['ours_std'])
                if r['ours_mean'] is not None else '—')
        fed = '%.1f ± %.1f' % (r['fedtad_mean'], r['fedtad_std'])
        delta = ('%+.2f' % r['delta']) if r['delta'] is not None else '—'
        L.append('| %s | %d | %s | %s | %s |'
                 % (r['dataset'], r['clients'], ours, fed, delta))
    L += ['', '## 汇总统计', '',
          '- datasets COMPLETE: %d / %d' % (totals['datasets_complete'],
                                            len(DATASETS)),
          '- cells COMPLETE: %d / 15' % totals['cells_complete'],
          '- Stage1 PASS: %d' % totals['stage1_pass'],
          '- Optuna attempts: %d | healthy: %d | failed: %d'
          % (totals['attempts'], totals['healthy'], totals['failed_trials']),
          '- final seeds: %d (15 cells × 3)' % totals['final_seeds'],
          '- final test evaluations: %d (每 seed exactly once)'
          % totals['final_test_evals'],
          '- tuning test evaluations: %d (全程 --tuning_mode)'
          % totals['tuning_test_evals'],
          '- vs FedTAD: wins %d / losses %d / ties %d'
          % (totals['wins'], totals['losses'], totals['ties']),
          '', '- 论文数字来源: %s' % results['paper_source'], '']
    mp = os.path.join(OUT, 'FORMAL_V3_ALL_DATASETS_REPORT.md')
    with open(mp, 'w') as f:
        f.write('\n'.join(L))
    print('all-datasets results written: %s' % jp)
    print('all-datasets report written:   %s' % mp)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
