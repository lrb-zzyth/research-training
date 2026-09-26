#!/usr/bin/env python3
"""Generate <DS>_V3_FINAL_SUMMARY.json/.md for a completed V3 dataset campaign.

Read-only over runs/formal_campaign_v3/<ds>/clients<tier>/.
Usage: python scripts/summarize_dataset_v3.py [Cora]
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def test_accuracy(fm):
    """Extract fedtad_official test accuracy from final_metrics.json."""
    m = (fm or {}).get('metrics') or {}
    for agg in ('fedtad_official', 'overall'):
        a = m.get(agg) or {}
        if a.get('accuracy') is not None:
            return float(a['accuracy'])
    return None


def phase_times(cell_dir):
    out = {}
    for ph in ('stage1', 'tuning', 'finals'):
        p = os.path.join(cell_dir, f'.{ph}_times')
        wall, starts, ends = [], [], []
        if os.path.exists(p):
            for ln in open(p):
                parts = ln.split()
                if len(parts) < 3:
                    continue
                if parts[0] == ph and parts[1] == 'start':
                    starts.append(int(parts[2]))
                elif parts[0] == ph and parts[1] == 'end':
                    ends.append(int(parts[2]))
            for s, e in zip(starts, ends):
                wall.append(max(0, e - s))
        out[ph] = {'launches': len(starts), 'wall_sec_total': sum(wall),
                   'wall_sec_list': wall}
    return out


def summarize(ds):
    out = os.path.join(REPO, 'runs', 'formal_campaign_v3', ds)
    tiers = [5, 10, 20]
    cells = []
    for tier in tiers:
        d = os.path.join(out, f'clients{tier}')
        c = {'cell': f'{ds}_c{tier}', 'dir': d}
        proto = load(os.path.join(d, 'protocol.json'))
        c['protocol'] = proto
        c['code_hash'] = (proto or {}).get('code_hash')
        c['data_identity_hash'] = (proto or {}).get('data_identity_hash')
        c['stage1_checkpoint_hash'] = (proto or {}).get('stage1_checkpoint_hash')
        c['kl_direction'] = (proto or {}).get('kl_direction')

        sh = load(os.path.join(d, 'stage1', 'stage1_health.json'))
        c['stage1_health'] = sh

        # --- Optuna study ---
        db = os.path.join(d, 'study.db')
        study = {'attempts': None, 'healthy': None, 'failed': None,
                 'best_trial': None, 'best_val': None, 'best_round': None,
                 'best_params': None, 'wall_sec_healthy_sum': None}
        if os.path.exists(db):
            try:
                import optuna
                st = optuna.load_study(
                    study_name=f'formal_v3_{ds}_c{tier}',
                    storage='sqlite:///{0}'.format(db))
                trials = st.trials
                healthy = [t for t in trials
                           if t.state.name == 'COMPLETE'
                           and t.user_attrs.get('health') == 'PASS']
                failed = [t for t in trials
                          if t.state.name in ('FAIL', 'PRUNED')]
                study['attempts'] = len(trials)
                study['healthy'] = len(healthy)
                study['failed'] = len(failed)
                study['wall_sec_healthy_sum'] = round(
                    sum(t.user_attrs.get('wall_sec', 0) or 0 for t in healthy), 1)
                if healthy:
                    best = max(healthy, key=lambda t: t.value)
                    fm = load(os.path.join(
                        d, 'trial_logs', f'trial_{best.number:03d}',
                        'final_metrics.json'))
                    study['best_trial'] = best.number
                    study['best_val'] = round(best.value, 4)
                    study['best_round'] = (fm or {}).get('best_round')
                    study['best_params'] = best.params
            except Exception as e:
                study['load_error'] = str(e)
        c['study'] = study
        c['best_params_file'] = load(os.path.join(d, 'best_params.json'))
        c['stability_failure'] = None
        sf = os.path.join(d, '.stability_failure')
        if os.path.exists(sf):
            c['stability_failure'] = open(sf).read().strip()

        # --- Finals ---
        finals = load(os.path.join(d, 'final', 'final_3seed_summary.json'))
        c['finals'] = {}
        if finals:
            for seed_s, fm in sorted(finals.items()):
                entry = {'best_val': (fm or {}).get('best_val_primary'),
                         'best_round': (fm or {}).get('best_round'),
                         'test_acc': test_accuracy(fm)}
                ru = load(os.path.join(d, 'final', f'seed_{seed_s}',
                                       'resource_usage.json'))
                entry['wall_sec'] = (ru or {}).get('total_wall_sec')
                entry['gen_peak_mem_mb_max'] = (ru or {}).get(
                    'gen_peak_mem_mb_max')
                c['finals'][seed_s] = entry
            tests = [e['test_acc'] for e in c['finals'].values()
                     if e['test_acc'] is not None]
            if tests:
                mean = sum(tests) / len(tests)
                var = sum((t - mean) ** 2 for t in tests) / len(tests)
                c['finals']['test_mean'] = round(mean, 4)
                c['finals']['test_std'] = round(var ** 0.5, 4)

        # --- Runtime / resources ---
        c['phase_times'] = phase_times(d)
        c['trial_peak_gen_mem_mb'] = None
        tl = os.path.join(d, 'trial_logs')
        peaks = []
        if os.path.isdir(tl):
            for td in sorted(os.listdir(tl)):
                ru = load(os.path.join(tl, td, 'resource_usage.json'))
                if ru and ru.get('gen_peak_mem_mb_max') is not None:
                    peaks.append(ru['gen_peak_mem_mb_max'])
        if peaks:
            c['trial_peak_gen_mem_mb'] = max(peaks)
        cells.append(c)

    return {'dataset': ds, 'generated_by': 'scripts/summarize_dataset_v3.py',
            'cells': cells}


def md(summary):
    ds = summary['dataset']
    L = ['# {0} V3 Formal Dataset Summary'.format(ds), '']
    L.append('> 由 `scripts/summarize_dataset_v3.py` 生成 (只读聚合 runs/ 产物)')
    L.append('')
    for c in summary['cells']:
        tier = c['cell']
        L.append('## {0}'.format(tier))
        L.append('')
        proto = c['protocol'] or {}
        L.append('- code hash: `{0}`'.format(c.get('code_hash')))
        L.append('- data identity: `{0}`'.format(c.get('data_identity_hash')))
        L.append('- Stage1 checkpoint hash: `{0}`'.format(
            c.get('stage1_checkpoint_hash')))
        L.append('- KL direction: {0}'.format(c.get('kl_direction')))
        sh = c['stage1_health'] or {}
        L.append('- Stage1 health: {0}'.format(sh.get('verdict', 'N/A')))
        st = c['study']
        L.append('- Optuna: attempts={0} healthy={1} failed={2}'.format(
            st.get('attempts'), st.get('healthy'), st.get('failed')))
        if st.get('best_val') is not None:
            L.append('- best (health=PASS) trial {0}: val={1} @ round {2}'.format(
                st['best_trial'], st['best_val'], st.get('best_round')))
            L.append('- best hyperparameters: `{0}`'.format(
                json.dumps(st['best_params'], sort_keys=True)))
        if c.get('stability_failure'):
            L.append('- **FORMAL NUMERICAL STABILITY FAILURE**: {0}'.format(
                c['stability_failure']))
        finals = c['finals']
        if finals.get('test_mean') is not None:
            L.append('- finals (3 seeds): test {0} ± {1}'.format(
                finals['test_mean'], finals.get('test_std')))
            for seed_s, e in sorted(finals.items()):
                if seed_s.startswith('20'):
                    L.append('  - seed {0}: test={1} (val {2} @ round {3})'.format(
                        seed_s, e.get('test_acc'), e.get('best_val'),
                        e.get('best_round')))
        pt = c['phase_times']
        for ph, info in pt.items():
            if info['wall_sec_list']:
                L.append('- {0} wall (s): {1} (launches {2})'.format(
                    ph, info['wall_sec_total'], info['launches']))
        L.append('')
    return '\n'.join(L)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else 'Cora'
    summary = summarize(ds)
    out = os.path.join(REPO, 'runs', 'formal_campaign_v3', ds)
    jp = os.path.join(out, '{0}_V3_FINAL_SUMMARY.json'.format(ds.upper()))
    mp = os.path.join(out, '{0}_V3_FINAL_SUMMARY.md'.format(ds.upper()))
    with open(jp, 'w') as f:
        json.dump(summary, f, indent=1)
    with open(mp, 'w') as f:
        f.write(md(summary))
    print('summary written: {0}'.format(jp))
    print('summary written: {0}'.format(mp))


if __name__ == '__main__':
    main()
