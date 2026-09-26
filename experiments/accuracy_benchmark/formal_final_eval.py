#!/usr/bin/env python3
"""Formal v3 final evaluation: fixed best hyperparameters, 3 fixed seeds, serial.

Per seed: 100 rounds, validation selects best checkpoint, test is evaluated exactly
once after loading that checkpoint (--final_test_only). No Optuna and no test-based
selection.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import formal_campaign as fc


def run_seed(ds, tier, seed, params, cell_dir):
    outdir = os.path.join(cell_dir, 'final', f'seed_{seed}')
    os.makedirs(outdir, exist_ok=True)
    stage1 = fc.stage1_path(ds, tier)
    flags = [x for x in fc.FIXED_FLAGS if x != '--tuning_mode'] + ['--final_test_only']
    cmd = [fc.PY, 'train_fedtad.py', '--root', './dataset', '--dataset', ds,
           '--num_clients', str(tier), '--partition', 'Louvain', '--seed', str(seed)]
    cmd += flags + [
        '--diffusion_pretrained_checkpoint', stage1,
        '--checkpoint_dir', outdir,
        '--emit_events', '--events_jsonl', os.path.join(outdir, 'events.jsonl'),
        '--final_metrics_json', os.path.join(outdir, 'final_metrics.json'),
        '--metrics_jsonl', os.path.join(outdir, 'metrics.jsonl'),
        '--resource_usage_json', os.path.join(outdir, 'resource_usage.json')]
    for k, v in params.items():
        cmd += [f'--{k}', str(v)]
    with open(os.path.join(outdir, 'stdout.log'), 'w') as fo, \
         open(os.path.join(outdir, 'stderr.log'), 'w') as fe:
        proc = subprocess.run(cmd, cwd=fc.REPO, stdout=fo, stderr=fe, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'{ds} c{tier} seed{seed} failed: exit={proc.returncode}')
    ok, health = fc.trial_health_check(outdir)
    if not ok:
        raise RuntimeError(f'{ds} c{tier} seed{seed} health FAIL: {health}')
    fp = os.path.join(outdir, 'final_metrics.json')
    if not os.path.exists(fp):
        raise RuntimeError(f'{ds} c{tier} seed{seed}: missing final_metrics.json')
    with open(fp) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, help='e.g. Cora:5')
    ap.add_argument('--seeds', default='2024,2025,2026')
    args = ap.parse_args()
    ds, tier_s = args.cell.split(':', 1)
    tier = int(tier_s)
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    cell_dir = os.path.join(fc.OUT, ds, f'clients{tier}')
    proto_path = os.path.join(cell_dir, 'protocol.json')
    params_path = os.path.join(cell_dir, 'best_params.json')
    if not os.path.exists(proto_path) or not os.path.exists(params_path):
        raise RuntimeError('formal tuning must finish first (protocol.json + best_params.json required)')
    with open(proto_path) as f:
        proto = json.load(f)
    if proto.get('code_hash') != fc.code_hash():
        raise RuntimeError('code hash changed since tuning; do not mix final evaluation with another code version')
    if proto.get('data_identity_hash') != fc.compute_data_identity(ds, tier):
        raise RuntimeError('data identity changed since tuning')
    with open(params_path) as f:
        params = json.load(f)
    results = {}
    for seed in seeds:
        print(f'[FINAL] {ds} c{tier} seed={seed}', flush=True)
        results[str(seed)] = run_seed(ds, tier, seed, params, cell_dir)
    out = os.path.join(cell_dir, 'final', 'final_3seed_summary.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'FINAL_EVAL_DONE: {out}')


if __name__ == '__main__':
    main()
