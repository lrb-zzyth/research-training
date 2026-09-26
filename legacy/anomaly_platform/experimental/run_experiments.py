"""
run_experiments.py — 统一实验入口

用法:
  python run_experiments.py --list-suites
  python run_experiments.py --prepare-proxy                 # 预训练代理 DDPM checkpoint
  python run_experiments.py --suite dry_run [--cpu]
  python run_experiments.py --suite main_anomaly_binary     # 正式实验 (跳过已完成)
  python run_experiments.py --suite main_anomaly_binary --resume-failed
  python run_experiments.py --aggregate --suite main_anomaly_binary
  python run_experiments.py --plots --suite main_anomaly_binary

运行目录: runs/<suite>/<experiment>/seed_<s>/
  config.json / status.json / stdout.log / stderr.log / best.pt
  final_metrics.json / metrics.jsonl / ckr.jsonl / split_report.json / resource_usage.json

规则:
  - 已 success 且存在 final_metrics.json 的运行自动跳过, 不覆盖
  - 失败运行记录原因; --resume-failed 时用 best.pt 恢复
  - 汇总只纳入 status=success 的运行, 缺失种子在 summary.md 中明确报告
  - --cpu 通过 CUDA_VISIBLE_DEVICES='' 强制 CPU (小规模 dry run)
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# 仓库根目录 (experimental/run_experiments.py -> 上一级)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(ROOT, 'runs')
PY = sys.executable
SEEDS = [0, 1, 2, 3, 4]
DRY_SEEDS = [0]

DATASET = os.path.join(ROOT, 'dataset')

# ---------------------------------------------------------------------
#  统一实验配置 (全部实验共享, 正式 GPU 实验不自动缩小)
# ---------------------------------------------------------------------
COMMON_ARGS = [
    '--root', DATASET, '--dataset', 'Cora', '--num_clients', '10',
    '--num_epochs', '1', '--hid_dim', '32', '--dropout', '0.2',
    '--lr', '1e-2', '--weight_decay', '5e-4',
    '--num_rounds', '8', '--f1_threshold=-1e6', '--auc_threshold=-1e6',
    '--contrastive_batch_size', '8', '--rwr_subgraph_size', '3',
    '--edge_perturb_ratio', '0.2',
    '--diffusion_steps', '4', '--diffusion_hidden', '32',
    '--generator_steps', '1', '--distill_steps', '2', '--fake_nodes', '48',
    '--generator_warmup_rounds', '0', '--generator_backprop_mode', 'checkpointed',
    '--reliability_holdout_ratio', '0.2', '--reliability_min_support', '3',
    '--dynamic_ckr_alpha', '0.5', '--dynamic_ckr_ema_decay', '0.8',
    '--dynamic_ckr_metric', 'f1', '--static_ckr_scaling', 'max',
    '--rwr_cache', 'enabled', '--rwr_cache_view1_persistent', 'enabled',
    '--rwr_seed', '0',
]

ANOMALY = ['--task_mode', 'anomaly_binary',
           '--normal_classes', '0,2,3,4,5', '--anomaly_classes', '1,6']
MULTICLASS = ['--task_mode', 'multiclass']

PROXY_CKPT = os.path.join(RUNS_ROOT, 'assets', 'proxy_diffusion.pt')

DRY_OVERRIDES = ['--num_clients', '2', '--num_rounds', '1',
                 '--hid_dim', '16', '--diffusion_hidden', '16',
                 '--fake_nodes', '24', '--contrastive_batch_size', '4',
                 '--rwr_subgraph_size', '2']

# ---------------------------------------------------------------------
#  实验定义
# ---------------------------------------------------------------------
def _exp(name, kwargs, seeds=None):
    return {'name': name, 'kwargs': kwargs, 'seeds': seeds}

# 基线阶梯 (anomaly binary)
B_ANOMALY = {
    'B0_local_only': _exp('B0_local_only', ANOMALY + ['--federated_mode', 'local_only']),
    'B1_fedavg': _exp('B1_fedavg', ANOMALY + ['--distill_weighting', 'none',
                     '--no-use_weighted_ce', '--contrastive_mode', 'none']),
    'B2_fedavg_wce': _exp('B2_fedavg_wce', ANOMALY + ['--distill_weighting', 'none',
                          '--contrastive_mode', 'none']),
    'B3_fedavg_wce_contrast': _exp('B3_fedavg_wce_contrast', ANOMALY +
                                   ['--distill_weighting', 'none']),
    'B4_static_ckr_distill': _exp('B4_static_ckr_distill', ANOMALY +
                                  ['--distill_weighting', 'static_ckr']),
    'B5_full': _exp('B5_full', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                          '--ckr_mode', 'hybrid_dynamic']),
}
B_MULTICLASS = {
    'B1_fedavg': _exp('B1_fedavg', MULTICLASS + ['--distill_weighting', 'none',
                     '--no-use_weighted_ce', '--contrastive_mode', 'none']),
    'B5_full': _exp('B5_full', MULTICLASS + ['--distill_weighting', 'dynamic_ckr',
                                             '--ckr_mode', 'hybrid_dynamic']),
}

SUITES = {
    'dry_run': {
        'seeds': DRY_SEEDS, 'overrides': DRY_OVERRIDES,
        'experiments': {
            **{k: _exp(k, v['kwargs']) for k, v in B_ANOMALY.items()},
            **{('B1_fedavg_mc' if k == 'B1_fedavg' else 'B5_full_mc'): _exp(
                'B1_fedavg_mc' if k == 'B1_fedavg' else 'B5_full_mc', v['kwargs'])
               for k, v in B_MULTICLASS.items()},
            'A1_ckr_static': _exp('A1_ckr_static',
                                  ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                             '--ckr_mode', 'static_topology']),
            'A9_backprop_truncated': _exp('A9_backprop_truncated',
                                          ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                     '--ckr_mode', 'hybrid_dynamic',
                                                     '--generator_backprop_mode', 'truncated']),
        },
    },
    'main_anomaly_binary': {'seeds': SEEDS, 'reference': 'B5_full',
                            'experiments': B_ANOMALY},
    'main_multiclass': {'seeds': SEEDS, 'reference': 'B5_full',
                        'experiments': B_MULTICLASS},
    'ckr_ablation': {'seeds': SEEDS, 'reference': 'A1_hybrid_dynamic',
                     'experiments': {
        'A1_static_topology': _exp('A1_static_topology',
                                   ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                              '--ckr_mode', 'static_topology']),
        'A1_performance_only': _exp('A1_performance_only',
                                    ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                               '--ckr_mode', 'performance_only']),
        'A1_hybrid_dynamic': _exp('A1_hybrid_dynamic',
                                  ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                             '--ckr_mode', 'hybrid_dynamic']),
    }},
    'metric_ablation': {'seeds': SEEDS, 'reference': 'A2_f1', 'experiments': {
        'A2_f1': _exp('A2_f1', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                          '--dynamic_ckr_metric', 'f1']),
        'A2_recall': _exp('A2_recall', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                  '--dynamic_ckr_metric', 'recall']),
        'A2_confidence': _exp('A2_confidence', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                          '--dynamic_ckr_metric', 'confidence']),
    }},
    'ema_ablation': {'seeds': SEEDS, 'reference': 'A3_default_ema', 'experiments': {
        'A3_no_ema': _exp('A3_no_ema', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                  '--dynamic_ckr_ema_decay', '0.0']),
        'A3_default_ema': _exp('A3_default_ema', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                            '--dynamic_ckr_ema_decay', '0.8']),
        'A3_strong_ema': _exp('A3_strong_ema', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                          '--dynamic_ckr_ema_decay', '0.95']),
    }},
    'generator_init_ablation': {'seeds': SEEDS, 'reference': 'A8_scratch', 'experiments': {
        'A8_scratch': _exp('A8_scratch', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                    '--generator_init', 'teacher_guided_scratch']),
        'A8_proxy_pretrained': _exp('A8_proxy_pretrained',
                                    ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                               '--generator_init', 'proxy_pretrained',
                                               '--proxy_checkpoint', PROXY_CKPT]),
    }},
    'backprop_ablation': {'seeds': SEEDS, 'reference': 'A9_checkpointed', 'experiments': {
        'A9_full': _exp('A9_full', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                              '--generator_backprop_mode', 'full']),
        'A9_checkpointed': _exp('A9_checkpointed', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                              '--generator_backprop_mode', 'checkpointed']),
        'A9_truncated': _exp('A9_truncated', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                        '--generator_backprop_mode', 'truncated',
                                                        '--generator_truncate_interval', '2']),
    }},
    'rwr_ablation': {'seeds': SEEDS, 'reference': 'A10_cache_on', 'experiments': {
        'A10_cache_off': _exp('A10_cache_off', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                          '--rwr_cache', 'disabled']),
        'A10_cache_on': _exp('A10_cache_on', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                        '--rwr_cache', 'enabled']),
    }},
    'contrastive_ablation': {'seeds': SEEDS, 'reference': 'A5_contrastive_on', 'experiments': {
        'A5_contrastive_off': _exp('A5_contrastive_off',
                                   ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                              '--contrastive_mode', 'none']),
        'A5_contrastive_on': _exp('A5_contrastive_on',
                                  ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                             '--contrastive_mode', 'subgraph_cross_view']),
    }},
    'ce_ablation': {'seeds': SEEDS, 'reference': 'A6_weighted_ce', 'experiments': {
        'A6_plain_ce': _exp('A6_plain_ce', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                      '--no-use_weighted_ce']),
        'A6_weighted_ce': _exp('A6_weighted_ce', ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                            '--use_weighted_ce']),
    }},
    'distill_ablation': {'seeds': SEEDS, 'reference': 'A7_dynamic_ckr', 'experiments': {
        'A7_equal': _exp('A7_equal', ANOMALY + ['--distill_weighting', 'equal']),
        'A7_static_ckr': _exp('A7_static_ckr', ANOMALY + ['--distill_weighting', 'static_ckr']),
        'A7_dynamic_ckr': _exp('A7_dynamic_ckr', ANOMALY + ['--distill_weighting', 'dynamic_ckr']),
    }},
    'source_ablation': {'seeds': [0, 1, 2], 'reference': 'A4_reliability_holdout',
                        'experiments': {
        'A4_reliability_holdout': _exp('A4_reliability_holdout',
                                       ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                                  '--dynamic_ckr_eval_source', 'reliability_holdout']),
        'A4_validation_leak': _exp('A4_validation_leak',
                                   ANOMALY + ['--distill_weighting', 'dynamic_ckr',
                                              '--dynamic_ckr_eval_source', 'validation']),
    }},
}

# ---------------------------------------------------------------------
#  第二阶段 suites
# ---------------------------------------------------------------------
SEEDS_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 动态 CKR 机制验证 (M1-M8, 10 seeds): static/dynamic 对必须同 seed 同划分
def _ckr_pair(name, support_kwargs, dynamic):
    kw = (ANOMALY + ['--distill_weighting', 'dynamic_ckr']
          + support_kwargs
          + (['--ckr_mode', 'hybrid_dynamic'] if dynamic
             else ['--ckr_mode', 'static_topology']))
    return _exp(name, kw)

CKR_MECHANISM = {
    'M1_natural_static': _ckr_pair('M1_natural_static', [], False),
    'M2_natural_dynamic': _ckr_pair('M2_natural_dynamic', [], True),
    'M3_stratified_static': _ckr_pair('M3_stratified_static', [
        '--split_support_mode', 'stratified_local',
        '--min_train_support_per_class', '3', '--min_val_support_per_class', '3',
        '--allow_support_infeasible'], False),
    'M4_stratified_dynamic': _ckr_pair('M4_stratified_dynamic', [
        '--split_support_mode', 'stratified_local',
        '--min_train_support_per_class', '3', '--min_val_support_per_class', '3',
        '--allow_support_infeasible'], True),
    'M5_boost_static': _ckr_pair('M5_boost_static', [
        '--split_support_mode', 'anomaly_holdout_boost',
        '--anomaly_val_ratio', '0.5',
        '--dynamic_ckr_eval_source', 'validation'], False),
    'M6_boost_dynamic': _ckr_pair('M6_boost_dynamic', [
        '--split_support_mode', 'anomaly_holdout_boost',
        '--anomaly_val_ratio', '0.5',
        '--dynamic_ckr_eval_source', 'validation'], True),
    'M7_enriched_static': _ckr_pair('M7_enriched_static', [
        '--split_support_mode', 'anomaly_enriched_partition',
        '--anomaly_partition_target', '75',
        '--min_train_support_per_class', '3'], False),
    'M8_enriched_dynamic': _ckr_pair('M8_enriched_dynamic', [
        '--split_support_mode', 'anomaly_enriched_partition',
        '--anomaly_partition_target', '75',
        '--min_train_support_per_class', '3'], True),
}
# dynamic/static 机制配对 (同划分模式)
MECHANISM_PAIRS = [('M2_natural_dynamic', 'M1_natural_static', 'natural'),
                   ('M4_stratified_dynamic', 'M3_stratified_static', 'stratified_local'),
                   ('M6_boost_dynamic', 'M5_boost_static', 'anomaly_holdout_boost'),
                   ('M8_enriched_dynamic', 'M7_enriched_static', 'anomaly_enriched_partition')]

LONG_HORIZON = {
    **{f'LH_{m}_natural': _exp(f'LH_{m}_natural',
                               (ANOMALY if m != 'B3' else ANOMALY) + [
                                   '--num_rounds', '16',
                                   '--distill_weighting',
                                   ('none' if m == 'B3' else 'dynamic_ckr'),
                                   '--ckr_mode',
                                   ('static_topology' if m == 'B4' else 'hybrid_dynamic')])
       for m in ['B3', 'B4', 'B5']},
    **{f'LH_{m}_enriched': _exp(f'LH_{m}_enriched',
                                (ANOMALY if m != 'B3' else ANOMALY) + [
                                    '--num_rounds', '16',
                                    '--split_support_mode', 'anomaly_enriched_partition',
                                    '--anomaly_partition_target', '75',
                                    '--distill_weighting',
                                    ('none' if m == 'B3' else 'dynamic_ckr'),
                                    '--ckr_mode',
                                    ('static_topology' if m == 'B4' else 'hybrid_dynamic')])
       for m in ['B3', 'B4', 'B5']},
}

BENCH_BASE = ['--feat_dim', '1433', '--batch_size', '48', '--num_classes', '2',
              '--resource_warmup_steps', '1', '--resource_measure_steps', '2']

RWR_EFFECTIVE = {}
_rwr_idx = 0
for epochs in (1, 3, 5):
    for cache_mode in ('off', 'safe_exact', 'epoch_reuse'):
        for anchor in ('random_each_epoch', 'fixed_per_round'):
            if cache_mode == 'safe_exact' and epochs != 3:
                continue  # safe_exact 只在 epochs=3 时覆盖 (控制网格规模)
            _rwr_idx += 1
            name = (f'RWR_e{epochs}_c{cache_mode}_a'
                    f"{'rand' if anchor == 'random_each_epoch' else 'fixed'}")
            RWR_EFFECTIVE[name] = _exp(name, ANOMALY + [
                '--local_epochs', str(epochs),
                '--rwr_cache_mode', cache_mode,
                '--anchor_sampling_mode', anchor,
                '--distill_weighting', 'dynamic_ckr',
            ])

SUITES2 = {
    'main_10seeds': {'seeds': SEEDS_10, 'reference': 'B5_full',
                     'experiments': B_ANOMALY},
    'ckr_support_mechanism': {'seeds': SEEDS_10, 'reference': 'M8_enriched_dynamic',
                              'mechanism_pairs': MECHANISM_PAIRS,
                              'experiments': CKR_MECHANISM},
    'long_horizon_ckr': {'seeds': SEEDS, 'reference': 'LH_B5_natural',
                         'experiments': LONG_HORIZON},
    'checkpoint_memory_scaling': {
        'seeds': [0], 'runner': 'benchmark',
        'experiments': {
            f'CMP_h{hid}_s{steps}_{mode}': _exp(
                f'CMP_h{hid}_s{steps}_{mode}',
                BENCH_BASE + ['--hidden_dim', str(hid),
                              '--diffusion_steps', str(steps),
                              '--generator_backward_mode', mode,
                              '--checkpoint_segments',
                              '2' if mode == 'checkpointed' else '1'])
            for hid in (32, 64, 128, 256)
            for steps in (10, 25, 50)
            for mode in ('full', 'checkpointed')
        }},
    'rwr_cache_effective': {'seeds': [0, 1, 2], 'reference': 'RWR_e3_cepoch_reuse_arand',
                            'experiments': RWR_EFFECTIVE},
    'dataset_transfer_citeseer': {
        'seeds': SEEDS, 'reference': 'B5_full', 'dataset': 'CiteSeer',
        'experiments': {
            **{f'B{i}_mc': _exp(f'B{i}_mc', ([] if i != 1 else ['--no-use_weighted_ce']) + [
                '--dataset', 'CiteSeer', '--task_mode', 'multiclass',
                '--distill_weighting', ('none' if i <= 3 else 'dynamic_ckr'),
                '--contrastive_mode', ('none' if i <= 2 else 'subgraph_cross_view'),
                '--ckr_mode', ('static_topology' if i == 4 else 'hybrid_dynamic')])
               for i in range(1, 6)},
            **{f'B{i}_ab': _exp(f'B{i}_ab', ([] if i != 1 else ['--no-use_weighted_ce']) + [
                '--dataset', 'CiteSeer', '--task_mode', 'anomaly_binary',
                '--normal_classes', '0,1,2,3', '--anomaly_classes', '4,5,6',
                '--distill_weighting', ('none' if i <= 3 else 'dynamic_ckr'),
                '--contrastive_mode', ('none' if i <= 2 else 'subgraph_cross_view'),
                '--ckr_mode', ('static_topology' if i == 4 else 'hybrid_dynamic')])
               for i in range(1, 6)},
        }},
    'pubmed_smoke': {'seeds': [0], 'reference': 'B5_full', 'dataset': 'PubMed',
                     'experiments': {
        'B1_pubmed': _exp('B1_pubmed', ['--dataset', 'PubMed', '--task_mode', 'multiclass',
                                        '--distill_weighting', 'none',
                                        '--no-use_weighted_ce', '--contrastive_mode', 'none']),
        'B5_pubmed': _exp('B5_pubmed', ['--dataset', 'PubMed', '--task_mode', 'multiclass',
                                        '--distill_weighting', 'dynamic_ckr']),
    }},
}

SUITES.update(SUITES2)

# ---------------------------------------------------------------------
#  阶段三 suites (frozen split 严格配对设计)
# ---------------------------------------------------------------------
P3_COMMON = ANOMALY + ['--resplit_stratified']

def _split_builder(p, enriched=False, natural=False):
    kw = (P3_COMMON + ['--partition_seed', str(p), '--allocation_seed', str(p),
                       '--split_seed', str(p),
                       '--build_split_artifact', f'@SPLITS@/Cora/'
                       f'{"enriched" if enriched else "natural"}_p{p}'])
    if enriched:
        kw += ['--split_support_mode', 'anomaly_enriched_partition',
               '--anomaly_partition_target', '75',
               '--min_train_support_per_class', '3']
    return _exp(f'BUILD_{"E" if enriched else "N"}_p{p}', kw, seeds=[0])

def _frozen_run(name, p, mode, enriched=False, seeds=SEEDS):
    """严格配对运行: 共享 frozen split, 唯一差异是 CKR mode。"""
    split_dir = f'@SPLITS@/Cora/{"enriched" if enriched else "natural"}_p{p}'
    kw = (P3_COMMON + ['--partition_seed', str(p), '--allocation_seed', str(p),
                       '--split_seed', str(p), '--frozen_split', split_dir,
                       '--distill_weighting', 'dynamic_ckr'])
    if mode == 'static':
        kw += ['--ckr_mode', 'static_topology']
    else:
        kw += ['--ckr_mode', 'hybrid_dynamic']
    return _exp(name, kw, seeds=seeds)

REPLICATION = {}
for p in range(5):
    REPLICATION[f'BUILD_E_p{p}'] = _split_builder(p, enriched=True)
    REPLICATION[f'STATIC_p{p}'] = _frozen_run(f'STATIC_p{p}', p, 'static',
                                              enriched=True)
    REPLICATION[f'DYNAMIC_p{p}'] = _frozen_run(f'DYNAMIC_p{p}', p, 'dynamic',
                                               enriched=True)

# ---- CKR 信息诊断 ----
def _diag_exp(name, mode, split_tag, p, seeds=SEEDS):
    split_dir = f'@SPLITS@/Cora/{split_tag}_p{p}'
    kw = (P3_COMMON + ['--partition_seed', str(p), '--allocation_seed', str(p),
                       '--split_seed', str(p), '--frozen_split', split_dir])
    mode_kw = {
        'equal': ['--distill_weighting', 'equal'],
        'static_ckr': ['--distill_weighting', 'static_ckr'],
        'dynamic_ckr': ['--distill_weighting', 'dynamic_ckr',
                        '--ckr_mode', 'hybrid_dynamic'],
        'performance_only': ['--distill_weighting', 'dynamic_ckr',
                             '--ckr_mode', 'performance_only'],
        'shuffled': ['--distill_weighting', 'static_ckr',
                     '--ckr_diagnostic_mode', 'shuffled'],
        'inverse': ['--distill_weighting', 'static_ckr',
                    '--ckr_diagnostic_mode', 'inverse'],
        'oracle': ['--distill_weighting', 'dynamic_ckr',
                   '--ckr_diagnostic_mode', 'oracle_validation'],
    }
    return _exp(name, kw + mode_kw[mode], seeds=seeds)

DIAGNOSTICS = {'BUILD_N_p0': _split_builder(0, natural=True),
               'BUILD_E_p0': _split_builder(0, enriched=True)}
for mode in ('equal', 'static_ckr', 'dynamic_ckr', 'performance_only',
             'shuffled', 'inverse', 'oracle'):
    DIAGNOSTICS[f'DIAG_N_{mode}'] = _diag_exp(f'DIAG_N_{mode}', mode, 'natural', 0)
for mode in ('equal', 'static_ckr', 'dynamic_ckr', 'oracle'):
    DIAGNOSTICS[f'DIAG_E_{mode}'] = _diag_exp(f'DIAG_E_{mode}', mode, 'enriched', 0)

# ---- 蒸馏因果链 (D0-D8) ----
CHAIN = {'BUILD_N_p0': _split_builder(0, natural=True)}
for d in range(9):
    kw_base = (P3_COMMON + ['--partition_seed', '0', '--allocation_seed', '0',
                            '--split_seed', '0',
                            '--frozen_split', '@SPLITS@/Cora/natural_p0'])
    cfg = {
        0: ['--distill_weighting', 'none'],
        1: ['--distill_weighting', 'equal'],
        2: ['--distill_weighting', 'static_ckr'],
        3: ['--distill_weighting', 'dynamic_ckr'],
        4: ['--distill_weighting', 'dynamic_ckr',
            '--ckr_diagnostic_mode', 'oracle_validation'],
        5: ['--distill_weighting', 'static_ckr',
            '--generator_update_mode', 'frozen'],
        6: ['--distill_weighting', 'static_ckr',
            '--lambda_disagreement', '0.0'],
        7: ['--distill_weighting', 'static_ckr'],
        8: ['--distill_weighting', 'static_ckr',
            '--fake_graph_topology', 'isolated'],
    }[d]
    CHAIN[f'D{d}'] = _exp(f'D{d}', kw_base + cfg)
# 16 轮关键配置
for d in (0, 3, 7):
    kw_base = (P3_COMMON + ['--partition_seed', '0', '--allocation_seed', '0',
                            '--split_seed', '0', '--num_rounds', '16',
                            '--frozen_split', '@SPLITS@/Cora/natural_p0'])
    cfg = {0: ['--distill_weighting', 'none'],
           3: ['--distill_weighting', 'dynamic_ckr'],
           7: ['--distill_weighting', 'static_ckr']}[d]
    CHAIN[f'D{d}_long'] = _exp(f'D{d}_long', kw_base + cfg)

# ---- 支持度 × 动态收益网格 ----
# 网格组合改变划分 (boost), 因此不使用 frozen split;
# 同组合内 static/dynamic 由相同种子保证同划分 (严格配对)。
GRID = {}
for val_ratio in (0.1, 0.2, 0.3):
    for min_train_sup in (0, 2, 5):
        for mode in ('static', 'dynamic'):
            kw = (P3_COMMON + ['--partition_seed', '0', '--allocation_seed', '0',
                               '--split_seed', '0',
                               '--split_support_mode', 'anomaly_holdout_boost',
                               '--anomaly_val_ratio', str(val_ratio),
                               '--min_train_support_per_class', str(min_train_sup),
                               '--allow_support_infeasible',
                               '--dynamic_ckr_eval_source', 'validation',
                               '--distill_weighting', 'dynamic_ckr',
                               '--ckr_mode',
                               ('static_topology' if mode == 'static'
                                else 'hybrid_dynamic')])
            GRID[f'G_v{val_ratio}_s{min_train_sup}_{mode}'] = _exp(
                f'G_v{val_ratio}_s{min_train_sup}_{mode}', kw, seeds=[0, 1, 2])

# ---- 最差客户端公平性 ----
FAIR = {'BUILD_N_p0': _split_builder(0, natural=True)}
fair_base = (P3_COMMON + ['--partition_seed', '0', '--allocation_seed', '0',
                          '--split_seed', '0',
                          '--frozen_split', '@SPLITS@/Cora/natural_p0'])
fair_cfg = {
    'F0_b3': ['--distill_weighting', 'none'],
    'F1_b5': ['--distill_weighting', 'dynamic_ckr'],
    'F2_b3_qffl': ['--distill_weighting', 'none',
                   '--fairness_mode', 'qffl_aggregation'],
    'F3_b5_qffl': ['--distill_weighting', 'dynamic_ckr',
                   '--fairness_mode', 'qffl_aggregation'],
    'F4_b5_deficit': ['--distill_weighting', 'dynamic_ckr',
                      '--fairness_mode', 'validation_deficit_distillation'],
    'F5_b5_combined': ['--distill_weighting', 'dynamic_ckr',
                       '--fairness_mode', 'combined'],
}
for name, cfg in fair_cfg.items():
    FAIR[name] = _exp(name, fair_base + cfg, seeds=SEEDS)

SUITES3 = {
    'ckr_enriched_independent_replication': {
        'seeds': SEEDS, 'reference': 'DYNAMIC_p0',
        'split_builder': True, 'experiments': REPLICATION},
    'ckr_information_diagnostics': {
        'seeds': SEEDS, 'reference': 'DIAG_N_dynamic_ckr',
        'split_builder': True, 'experiments': DIAGNOSTICS},
    'distillation_causal_chain': {
        'seeds': SEEDS, 'reference': 'D7',
        'split_builder': True, 'experiments': CHAIN},
    'ckr_support_gain_grid': {
        'seeds': [0, 1, 2], 'reference': 'G_v0.2_s2_dynamic',
        'split_builder': True, 'experiments': GRID},
    'worst_client_fairness': {
        'seeds': SEEDS, 'reference': 'F1_b5',
        'split_builder': True, 'experiments': FAIR},
}
SUITES.update(SUITES3)

# ---------------------------------------------------------------------
#  工具
# ---------------------------------------------------------------------
def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return 'unknown'


def cfg_hash(kwargs):
    payload = json.dumps(sorted(kwargs), default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


SPLITS_ROOT = os.path.join(RUNS_ROOT, 'splits')


def _resolve_split_tokens(kwargs):
    """将 '@SPLITS@' 占位符替换为 splits 根目录 (frozen split 共享路径)。"""
    return [os.path.join(SPLITS_ROOT, v[1:]) if v.startswith('@SPLITS@')
            else v for v in kwargs]


def build_cli(kwargs, seed, rundir, runner='train'):
    kwargs = _resolve_split_tokens(kwargs)
    if runner == 'benchmark':
        return [PY, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'benchmark_checkpointing.py')] + kwargs + [
            '--seed', str(seed),
            '--output_json', os.path.join(rundir, 'numerical_equivalence.json')]
    cli = [PY, os.path.join(ROOT, 'train_fedtad.py')] + COMMON_ARGS + kwargs + [
        '--seed', str(seed),
        '--model_seed', str(seed),
        '--checkpoint_dir', rundir,
        '--log_dir', rundir,
        '--final_metrics_json', os.path.join(rundir, 'final_metrics.json'),
        '--metrics_jsonl', os.path.join(rundir, 'metrics.jsonl'),
        '--split_report_json', os.path.join(rundir, 'split_report.json'),
        '--resource_usage_json', os.path.join(rundir, 'resource_usage.json'),
        '--evaluation_report_json', os.path.join(rundir, 'evaluation_report.json'),
        '--ckr_availability_json', os.path.join(rundir, 'ckr_availability.json'),
        '--fairness_metrics_json', os.path.join(rundir, 'fairness_metrics.json'),
        '--cache_metrics_json', os.path.join(rundir, 'cache_metrics.json'),
        '--data_identity_json', os.path.join(rundir, 'data_identity.json'),
        '--initialization_json', os.path.join(rundir, 'initialization.json'),
        '--teacher_mixture_jsonl', os.path.join(rundir, 'teacher_mixture.jsonl'),
        '--distillation_diagnostics_jsonl', os.path.join(
            rundir, 'distillation_diagnostics.jsonl'),
        '--dataset_manifest', os.path.join(rundir, 'dataset_manifest.json'),
    ]
    return cli


def _success_marker(runner, rundir, kwargs=None):
    """runner 对应的成功标志文件 (builder 以 artifact 目录为准)。"""
    if runner == 'benchmark':
        return os.path.join(rundir, 'numerical_equivalence.json')
    if kwargs and '--build_split_artifact' in kwargs:
        idx = kwargs.index('--build_split_artifact')
        return _resolve_split_tokens([kwargs[idx + 1]])[0]
    return os.path.join(rundir, 'final_metrics.json')


TINY_OVERRIDES = ['--num_clients', '2', '--num_rounds', '1',
                  '--hid_dim', '8', '--diffusion_hidden', '8',
                  '--fake_nodes', '16', '--contrastive_batch_size', '4',
                  '--rwr_subgraph_size', '2', '--diffusion_steps', '3',
                  '--num_epochs', '1', '--distill_steps', '1']
BENCH_TINY = ['--hidden_dim', '16', '--diffusion_steps', '4',
              '--feat_dim', '32', '--batch_size', '8',
              '--resource_warmup_steps', '0', '--resource_measure_steps', '1']


def run_one(suite_cfg, exp, seed, opts):
    name = exp['name']
    runner = suite_cfg.get('runner', 'train')
    rundir = os.path.join(RUNS_ROOT, opts.suite, name, f'seed_{seed}')
    os.makedirs(rundir, exist_ok=True)
    status_path = os.path.join(rundir, 'status.json')
    kwargs = exp['kwargs'] + (suite_cfg.get('overrides', []) or [])
    final_path = _success_marker(runner, rundir, kwargs)
    if opts.dry:
        kwargs = kwargs + (BENCH_TINY if runner == 'benchmark' else TINY_OVERRIDES)
    cur_hash = cfg_hash(kwargs)

    # 外部资源阻塞: 套件内已有同一 dataset 的 external_blocked 记录时,
    # 该 dataset 的其余单元标记 skipped_unavailable_resource (不重复制造失败)
    def _kw_dataset(kw):
        for i in range(0, len(kw), 2):
            if kw[i] == '--dataset':
                return kw[i + 1]
        return 'Cora'
    cur_dataset = _kw_dataset(kwargs)
    suite_base = os.path.join(RUNS_ROOT, opts.suite)
    if os.path.isdir(suite_base) and not os.path.exists(status_path):
        blocked_dataset = None
        for n0 in sorted(os.listdir(suite_base)):
            d0 = os.path.join(suite_base, n0)
            if not os.path.isdir(d0):
                continue
            for sd0 in os.listdir(d0):
                sp0 = os.path.join(d0, sd0, 'status.json')
                if os.path.exists(sp0):
                    try:
                        st0 = json.load(open(sp0))
                    except Exception:
                        continue
                    if st0.get('status') == 'external_blocked':
                        cfg0 = os.path.join(d0, sd0, 'config.json')
                        if os.path.exists(cfg0):
                            try:
                                blocked_dataset = _kw_dataset(
                                    json.load(open(cfg0)).get('kwargs', []))
                            except Exception:
                                blocked_dataset = None
                        else:
                            blocked_dataset = None
                        break
            if blocked_dataset:
                break
        if blocked_dataset == cur_dataset:
            json.dump({'status': 'skipped_unavailable_resource',
                       'reason': f'dataset {cur_dataset} 已有 external_blocked 记录',
                       'config_hash': cur_hash}, open(status_path, 'w'), indent=1)
            print(f"[skip-resource] {name}/seed_{seed} "
                  f"({cur_dataset} external_blocked 已记录)")
            return 'skipped_unavailable_resource'

    # 跳过已完成 (且配置哈希一致; 配置哈希改变必须重跑)
    if os.path.exists(final_path) and os.path.exists(status_path):
        try:
            st = json.load(open(status_path))
            if (st.get('status') == 'success'
                    and st.get('config_hash') == cur_hash):
                print(f"[skip] {name}/seed_{seed} 已完成 (hash 一致)")
                return 'skipped'
            if st.get('status') == 'success':
                print(f"[rerun] {name}/seed_{seed} 配置哈希变化 "
                      f"({st.get('config_hash')} -> {cur_hash})")
        except Exception:
            pass

    cli = build_cli(kwargs, seed, rundir, runner=runner)
    if opts.resume_failed and os.path.exists(os.path.join(rundir, 'best.pt')):
        cli += ['--resume_checkpoint', os.path.join(rundir, 'best.pt')]
        print(f"[resume] {name}/seed_{seed} 从 best.pt 恢复")

    cfg = {'suite': opts.suite, 'experiment': name, 'seed': seed,
           'runner': runner,
           'config_hash': cur_hash, 'kwargs': kwargs,
           'command': ' '.join(cli), 'git_commit': git_commit(),
           'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(cfg, open(os.path.join(rundir, 'config.json'), 'w'), indent=1)
    json.dump({'status': 'running', 'started_at': cfg['started_at']},
              open(status_path, 'w'), indent=1)

    env = dict(os.environ)
    if opts.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    t0 = time.time()
    with open(os.path.join(rundir, 'stdout.log'), 'w') as fo, \
            open(os.path.join(rundir, 'stderr.log'), 'w') as fe:
        proc = subprocess.run(cli, cwd=ROOT, env=env, stdout=fo, stderr=fe)
    wall = time.time() - t0

    # ckr jsonl 重命名
    ckr_src = os.path.join(rundir, 'dynamic_ckr_history.jsonl')
    if os.path.exists(ckr_src):
        os.replace(ckr_src, os.path.join(rundir, 'ckr.jsonl'))

    if proc.returncode == 0 and os.path.exists(final_path):
        json.dump({'status': 'success', 'wall_sec': round(wall, 1),
                   'config_hash': cur_hash,
                   'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(status_path, 'w'), indent=1)
        print(f"[ok] {name}/seed_{seed} ({wall:.0f}s)")
        return 'success'
    err_tail = ''
    try:
        err_tail = open(os.path.join(rundir, 'stderr.log')).read()[-2000:]
    except Exception:
        pass
    # 状态分类: 退出码 3 + [BLOCKED] 哨兵 = external_blocked (外部资源阻塞, 非代码失败)
    blocked = (proc.returncode == 3 or '[BLOCKED]' in err_tail
               or 'external resource blocked' in err_tail)
    if blocked:
        status = 'external_blocked'
    elif proc.returncode == 2:
        status = 'failed_config'
    else:
        status = 'failed_code'
    json.dump({'status': status, 'wall_sec': round(wall, 1),
               'returncode': proc.returncode, 'stderr_tail': err_tail,
               'config_hash': cur_hash,
               'resumable': os.path.exists(os.path.join(rundir, 'best.pt'))},
              open(status_path, 'w'), indent=1)
    print(f"[{status.upper()}] {name}/seed_{seed} exit={proc.returncode}\n"
          f"{err_tail[-800:]}")
    return status


# ---------------------------------------------------------------------
#  指标提取
# ---------------------------------------------------------------------
def extract_metrics(final):
    """从 final_metrics.json 提取扁平指标 dict。"""
    m = final.get('metrics', {})
    out = {
        'seed': final.get('seed'),
        'config_hash': final.get('config_hash'),
        'best_round': final.get('best_round'),
        'loaded_from': final.get('loaded_from'),
        'task_mode': final.get('task_mode'),
    }
    pooled = m.get('pooled', {})
    cm = m.get('client_macro', {})
    if final.get('task_mode') == 'multiclass':
        out.update({
            'pooled_accuracy': pooled.get('accuracy'),
            'pooled_macro_f1': pooled.get('macro_f1'),
            'client_macro_accuracy': cm.get('accuracy'),
            'client_macro_f1': cm.get('macro_f1'),
        })
    else:
        out.update({
            'pooled_roc_auc': pooled.get('roc_auc'),
            'pooled_pr_auc': pooled.get('pr_auc'),
            'client_macro_roc_auc': cm.get('roc_auc'),
            'client_macro_pr_auc': cm.get('pr_auc'),
        })
    return out


def collect_runs(suite):
    """收集 suite 下所有 success 运行。返回 [(exp_name, seed, final_metrics, rundir)]"""
    rows = []
    base = os.path.join(RUNS_ROOT, suite)
    if not os.path.exists(base):
        return rows
    runner = SUITES.get(suite, {}).get('runner', 'train')
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        for sd in sorted(os.listdir(d)):
            if not sd.startswith('seed_'):
                continue
            rundir = os.path.join(d, sd)
            sp = os.path.join(rundir, 'status.json')
            kwargs = None
            cfgp = os.path.join(rundir, 'config.json')
            if os.path.exists(cfgp):
                try:
                    kwargs = json.load(open(cfgp)).get('kwargs')
                except Exception:
                    kwargs = None
            marker = _success_marker(runner, '', kwargs)
            fp = os.path.join(rundir, marker)
            if not (os.path.exists(sp) and os.path.exists(fp)):
                continue
            try:
                st = json.load(open(sp))
                if st.get('status') != 'success':
                    continue
                fm = json.load(open(fp)) if os.path.exists(
                    os.path.join(rundir, 'final_metrics.json')) else {}
            except Exception:
                continue
            seed = int(sd.replace('seed_', ''))
            rows.append({'suite': suite, 'name': name, 'seed': seed,
                         'final': fm, 'rundir': rundir})
    return rows


# ---------------------------------------------------------------------
#  聚合
# ---------------------------------------------------------------------
def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})
    print(f"  wrote {path}")


def aggregate_suite(suite):
    from experiment_stats import summarize, paired_diff
    rows = collect_runs(suite)
    suite_cfg = SUITES.get(suite, {})
    print(f"[aggregate] {suite}: {len(rows)} success runs")

    # all_runs.csv
    header = ['suite', 'name', 'seed', 'config_hash', 'task_mode',
              'pooled_pr_auc', 'pooled_roc_auc', 'client_macro_pr_auc',
              'client_macro_roc_auc', 'pooled_accuracy', 'pooled_macro_f1',
              'best_round', 'loaded_from', 'wall_sec', 'gen_peak_mem_mb_max',
              'rwr_hit_rate', 'zero_anomaly_clients']
    all_rows = []
    per_client_rows = []
    for r in rows:
        em = extract_metrics(r['final'])
        res = {}
        if os.path.exists(os.path.join(r['rundir'], 'resource_usage.json')):
            try:
                res = json.load(open(os.path.join(r['rundir'], 'resource_usage.json')))
            except Exception:
                res = {}
        row = {'suite': r['suite'], 'name': r['name'], 'seed': r['seed'],
               'config_hash': r['final'].get('config_hash'),
               'task_mode': r['final'].get('task_mode'),
               **{k: em.get(k) for k in
                  ('pooled_pr_auc', 'pooled_roc_auc', 'client_macro_pr_auc',
                   'client_macro_roc_auc', 'pooled_accuracy', 'pooled_macro_f1',
                   'best_round', 'loaded_from')},
               'wall_sec': round(res.get('total_wall_sec', float('nan')), 1),
               'gen_peak_mem_mb_max': res.get('gen_peak_mem_mb_max'),
               'rwr_hit_rate': res.get('rwr_cache_hit_rate_final'),
               'zero_anomaly_clients': json.dumps(
                   r['final'].get('zero_anomaly_clients', []))}
        all_rows.append(row)
        # per-client
        per_c = r['final'].get('metrics', {}).get('per_client', {})
        for ci, cm_ in per_c.items():
            pc = {'suite': r['suite'], 'name': r['name'], 'seed': r['seed'],
                  'client_id': ci}
            if r['final'].get('task_mode') == 'multiclass':
                pc.update({'accuracy': cm_.get('accuracy'), 'macro_f1': cm_.get('macro_f1'),
                           'per_class_f1': json.dumps(cm_.get('per_class_f1'))})
            else:
                pc.update({'roc_auc': cm_.get('roc_auc'), 'pr_auc': cm_.get('pr_auc'),
                           'recall': cm_.get('recall'), 'precision': cm_.get('precision'),
                           'f1': cm_.get('f1'),
                           'balanced_accuracy': cm_.get('balanced_accuracy')})
            per_client_rows.append(pc)
    _write_csv(os.path.join(RUNS_ROOT, suite, 'all_runs.csv'), header, all_rows)
    _write_csv(os.path.join(RUNS_ROOT, suite, 'per_client_results.csv'),
               list(per_client_rows[0].keys()) if per_client_rows else ['suite'],
               per_client_rows)

    # main_results.csv: 每实验 mean±std (排除 builder 等无 task_mode 的行)
    all_rows = [r for r in all_rows if r.get('task_mode')]
    task_mode = all_rows[0].get('task_mode') if all_rows else 'anomaly_binary'
    metric_cols = (['pooled_pr_auc', 'pooled_roc_auc', 'client_macro_pr_auc',
                    'client_macro_roc_auc'] if task_mode == 'anomaly_binary'
                   else ['pooled_accuracy', 'pooled_macro_f1'])
    mr_header = ['name', 'seeds'] + [f'{c}_mean' for c in metric_cols] + \
                [f'{c}_std' for c in metric_cols] + [f'{c}_n' for c in metric_cols]
    mr_rows = []
    by_name = {}
    for r in all_rows:
        by_name.setdefault(r['name'], []).append(r)
    for name in sorted(by_name):
        vals = by_name[name]
        seeds = sorted(v['seed'] for v in vals)
        row = {'name': name, 'seeds': json.dumps(seeds)}
        for c in metric_cols:
            s = summarize([v[c] for v in vals])
            row[f'{c}_mean'] = round(s['mean'], 3) if s['n'] else ''
            row[f'{c}_std'] = round(s['std'], 3) if s['n'] else ''
            row[f'{c}_n'] = s['n']
        mr_rows.append(row)
    _write_csv(os.path.join(RUNS_ROOT, suite, 'main_results.csv'), mr_header, mr_rows)

    # resource_results.csv
    res_header = ['name', 'seed', 'total_wall_sec', 'round_times_mean_sec',
                  'gen_peak_mem_mb_max', 'rwr_hit_rate']
    res_rows = []
    for r in rows:
        rp = os.path.join(r['rundir'], 'resource_usage.json')
        res = json.load(open(rp)) if os.path.exists(rp) else {}
        rt = res.get('round_times_sec', [])
        res_rows.append({'name': r['name'], 'seed': r['seed'],
                         'total_wall_sec': res.get('total_wall_sec'),
                         'round_times_mean_sec': round(np.mean(rt), 2) if rt else '',
                         'gen_peak_mem_mb_max': res.get('gen_peak_mem_mb_max'),
                         'rwr_hit_rate': res.get('rwr_cache_hit_rate_final')})
    _write_csv(os.path.join(RUNS_ROOT, suite, 'resource_results.csv'), res_header, res_rows)

    # paired_comparisons.csv: reference 配置 vs 其余 (同 suite, 同种子)
    reference = suite_cfg.get('reference', 'B5_full')
    if reference in by_name and len(by_name) > 1:
        ref_by_seed = {v['seed']: v for v in by_name[reference]}
        pc_header = ['reference', 'baseline', 'metric', 'n_pairs', 'mean_diff',
                     'std_diff', 'ci95_low', 'ci95_high', 'wins', 'losses',
                     'ties', 'wilcoxon_p']
        pc_rows = []
        for name in sorted(by_name):
            if name == reference:
                continue
            base_by_seed = {v['seed']: v for v in by_name[name]}
            common_seeds = sorted(set(ref_by_seed) & set(base_by_seed))
            if not common_seeds:
                continue
            # 评估口径守卫: 只有 metric_comparability_group 相同才能配对
            # (B0 local_ensemble 不得与 global_model 方法配对)
            def _scope(name_, seed_):
                er = os.path.join(RUNS_ROOT, suite, name_, f'seed_{seed_}',
                                  'evaluation_report.json')
                if os.path.exists(er):
                    try:
                        return json.load(open(er)).get('evaluation_scope')
                    except Exception:
                        return None
                return None
            scope_ref = _scope(reference, common_seeds[0])
            scope_base = _scope(name, common_seeds[0])
            if scope_ref and scope_base and scope_ref != scope_base:
                print(f"  [pairing-skip] {name} (scope={scope_base}) 与 "
                      f"{reference} (scope={scope_ref}) 口径不同, 不配对")
                continue
            for c in metric_cols:
                fv = [ref_by_seed[s][c] for s in common_seeds]
                bv = [base_by_seed[s][c] for s in common_seeds]
                pd = paired_diff(fv, bv, common_seeds)
                pc_rows.append({'reference': reference, 'baseline': name,
                                'metric': c, 'n_pairs': pd['n_pairs'],
                                'mean_diff': round(pd['mean_diff'], 3) if pd['n_pairs'] else '',
                                'std_diff': round(pd['std_diff'], 3) if pd['n_pairs'] else '',
                                'ci95_low': round(pd['ci95_low'], 3) if pd['n_pairs'] else '',
                                'ci95_high': round(pd['ci95_high'], 3) if pd['n_pairs'] else '',
                                'wins': pd['wins'], 'losses': pd['losses'],
                                'ties': pd['ties'],
                                'wilcoxon_p': round(pd['wilcoxon_p'], 4)
                                if not math.isnan(pd['wilcoxon_p']) else ''})
        _write_csv(os.path.join(RUNS_ROOT, suite, 'paired_comparisons.csv'),
                   pc_header, pc_rows)

    # summary.md
    md = [f"# Suite: {suite}", "", f"success runs: {len(rows)}",
          f"git commit: {git_commit()}", "", "## Main results (mean ± std)", ""]
    for r in mr_rows:
        md.append(f"- **{r['name']}** (seeds {r['seeds']}): " + ', '.join(
            f"{c}={r[f'{c}_mean']} ± {r[f'{c}_std']} (n={r[f'{c}_n']})" for c in metric_cols))
    md.append("")
    missing = []
    for name in sorted(set(e['name'] for e in SUITES[suite]['experiments'].values())):
        got = {v['seed'] for v in by_name.get(name, [])}
        want = set(SUITES[suite].get('seeds', SEEDS))
        if got != want:
            missing.append(f"{name}: missing seeds {sorted(want - got)}")
    md.append("## Missing / failed")
    md.append(', '.join(missing) if missing else 'none')
    with open(os.path.join(RUNS_ROOT, suite, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"  wrote summary.md")
    aggregate_extended(suite, all_rows, by_name, metric_cols, task_mode)
    return all_rows


def _load_sidecar(rundir, fname):
    p = os.path.join(rundir, fname)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def aggregate_extended(suite, all_rows, by_name, metric_cols, task_mode):
    """
    第二阶段扩展聚合:
      - 10-seed 统计 (median / bootstrap CI / Cliff's delta / Holm 校正)
      - 机制配对 (dynamic - static per support mode)
      - ckr_analysis.csv / fairness_analysis.csv / cache_analysis.csv /
        numerical_equivalence.csv
    """
    from experiment_stats import (summarize, paired_diff, paired_bootstrap_ci,
                                       holm_correction, cliffs_delta, sensitivity_analysis)
    rows = collect_runs(suite)
    outdir = os.path.join(RUNS_ROOT, suite)

    # ---- 增强配对统计 (median diff, bootstrap CI, Cliff's delta, Holm) ----
    pc_path = os.path.join(outdir, 'paired_comparisons.csv')
    if os.path.exists(pc_path):
        import csv as _csv
        pc = list(_csv.DictReader(open(pc_path)))
        ref = SUITES.get(suite, {}).get('reference', 'B5_full')
        if pc:
            # 收集 (reference, baseline, metric) 的原始配对数据并增强
            def _by_seed(rows):
                return {r['seed']: r for r in rows}
            names = sorted(by_name)
            ref_rows = _by_seed(by_name.get(ref, []))
            enhanced = []
            holm_input = []
            for r in pc:
                base = r['baseline']
                base_rows = _by_seed(by_name.get(base, []))
                common = sorted(set(ref_rows) & set(base_rows))
                fv = [ref_rows[s][r['metric']] for s in common]
                bv = [base_rows[s][r['metric']] for s in common]
                pd = paired_diff(fv, bv, common)
                boot = paired_bootstrap_ci(fv, bv, common, n_boot=1000)
                cd = cliffs_delta(fv, bv)
                sens = sensitivity_analysis(fv, bv)
                rec = dict(r)
                rec.update({'median_diff': (round(boot['median_diff'], 3)
                                            if boot['n_pairs'] else ''),
                            'bootstrap_ci_low': (round(boot['ci95_low'], 3)
                                                 if boot['n_pairs'] else ''),
                            'bootstrap_ci_high': (round(boot['ci95_high'], 3)
                                                  if boot['n_pairs'] else ''),
                            'cliffs_delta': (round(cd['delta'], 3)
                                             if cd['n'] else ''),
                            'cliffs_interpretation': cd['interpretation'],
                            'sensitivity_range': (round(sens['range'], 3)
                                                  if sens['n'] else ''),
                            'missing_pairs': len(common) - pd['n_pairs']})
                if pd['wilcoxon_p'] != '' and pd['wilcoxon_p'] != 'nan':
                    holm_input.append((f"{r['baseline']}|{r['metric']}",
                                       float(pd['wilcoxon_p'])))
                enhanced.append(rec)
            if holm_input:
                holm_map = {name: p_holm
                            for name, p, p_holm in holm_correction(holm_input)}
                for rec in enhanced:
                    key = f"{rec['baseline']}|{rec['metric']}"
                    rec['wilcoxon_p_holm'] = (round(holm_map[key], 4)
                                              if key in holm_map else '')
            else:
                for rec in enhanced:
                    rec['wilcoxon_p_holm'] = ''
            header = list(enhanced[0].keys())
            _write_csv(os.path.join(outdir, 'paired_comparisons_enhanced.csv'),
                       header, enhanced)
            md_extra = [f"## Enhanced paired stats (bootstrap CI, Cliff's delta, Holm)",
                        '']
            for rec in enhanced:
                md_extra.append(
                    f"- {rec['baseline']} vs {ref} [{rec['metric']}]: "
                    f"mean_diff={rec['mean_diff']} median_diff={rec['median_diff']} "
                    f"boot95=[{rec['bootstrap_ci_low']},{rec['bootstrap_ci_high']}] "
                    f"cliffs={rec['cliffs_delta']}({rec['cliffs_interpretation']}) "
                    f"p={rec['wilcoxon_p']} p_holm={rec['wilcoxon_p_holm']} "
                    f"wins={rec['wins']}/{rec['losses']}")
            with open(os.path.join(outdir, 'summary.md'), 'a') as f:
                f.write('\n'.join(md_extra) + '\n')

    # ---- 机制配对: dynamic - static (ckr_support_mechanism) ----
    mp = SUITES.get(suite, {}).get('mechanism_pairs', [])
    if mp:
        def _by_seed(rows):
            return {r['seed']: r for r in rows}
        mech_rows = []
        for dyn, stat, mode_label in mp:
            d_rows = _by_seed(by_name.get(dyn, []))
            s_rows = _by_seed(by_name.get(stat, []))
            common = sorted(set(d_rows) & set(s_rows))
            for c in metric_cols:
                fv = [d_rows[s][c] for s in common]
                bv = [s_rows[s][c] for s in common]
                pd = paired_diff(fv, bv, common)
                boot = paired_bootstrap_ci(fv, bv, common, n_boot=1000)
                mech_rows.append({
                    'support_mode': mode_label, 'dynamic': dyn, 'static': stat,
                    'metric': c, 'n_pairs': pd['n_pairs'],
                    'mean_diff': round(pd['mean_diff'], 3) if pd['n_pairs'] else '',
                    'median_diff': round(boot['median_diff'], 3) if boot['n_pairs'] else '',
                    'bootstrap_ci_low': round(boot['ci95_low'], 3) if boot['n_pairs'] else '',
                    'bootstrap_ci_high': round(boot['ci95_high'], 3) if boot['n_pairs'] else '',
                    'wins': pd['wins'], 'losses': pd['losses'],
                    'wilcoxon_p': round(pd['wilcoxon_p'], 4)
                    if not math.isnan(pd['wilcoxon_p']) else ''})
        _write_csv(os.path.join(outdir, 'mechanism_pairs.csv'),
                   list(mech_rows[0].keys()) if mech_rows else ['support_mode'],
                   mech_rows)
        # CKR availability 与 dynamic gain 的相关
        avails = {}
        for dyn, stat, mode_label in mp:
            for r in rows:
                if r['name'] != dyn:
                    continue
                ck = _load_sidecar(r['rundir'], 'ckr_availability.json')
                ag = ck.get('rounds', [])
                if ag:
                    avails[(mode_label, r['seed'])] = (sum(x['available_ratio']
                                                           for x in ag) / len(ag))
        corr_rows = []
        for mode_label in sorted({m[2] for m in mp}):
            xs, ys = [], []
            for (ml, seed), avail in avails.items():
                if ml != mode_label:
                    continue
                d_rows = {r['seed']: r for r in by_name.get(
                    next(m[0] for m in mp if m[2] == ml), [])}
                s_rows = {r['seed']: r for r in by_name.get(
                    next(m[1] for m in mp if m[2] == ml), [])}
                if seed in d_rows and seed in s_rows:
                    g = d_rows[seed]['pooled_pr_auc'] - s_rows[seed]['pooled_pr_auc']
                    if g is not None and not (isinstance(g, str)):
                        try:
                            xs.append(avail)
                            ys.append(float(g))
                        except (TypeError, ValueError):
                            pass
            if len(xs) >= 3:
                corr = np.corrcoef(xs, ys)[0, 1]
            else:
                corr = float('nan')
            corr_rows.append({'support_mode': mode_label,
                              'n': len(xs),
                              'corr_available_vs_dynamic_gain': round(corr, 3)})
        _write_csv(os.path.join(outdir, 'ckr_availability_vs_gain.csv'),
                   list(corr_rows[0].keys()) if corr_rows else ['support_mode'],
                   corr_rows)

    # ---- ckr_analysis.csv ----
    ckr_rows = []
    for r in rows:
        ck = _load_sidecar(r['rundir'], 'ckr_availability.json')
        ag = ck.get('rounds', [])
        if not ag:
            continue
        ckr_rows.append({
            'suite': suite, 'name': r['name'], 'seed': r['seed'],
            'n_rounds': len(ag),
            'available_ratio_mean': round(sum(x['available_ratio']
                                              for x in ag) / len(ag), 4),
            'fallback_ratio_mean': round(sum(x['fallback_ratio']
                                             for x in ag) / len(ag), 4),
            'changed_ratio_mean': round(sum(x['changed_ratio']
                                            for x in ag) / len(ag), 4),
            'mean_abs_delta_mean': round(sum(x['mean_abs_delta']
                                             for x in ag) / len(ag), 6),
            'per_class_available_ratio_mean': [
                round(sum(x['per_class_available_ratio'][j] for x in ag) / len(ag), 4)
                for j in range(len(ag[0]['per_class_available_ratio']))],
        })
    if ckr_rows:
        _write_csv(os.path.join(outdir, 'ckr_analysis.csv'), list(ckr_rows[0].keys()),
                   ckr_rows)

    # ---- fairness_analysis.csv ----
    fair_rows = []
    for r in rows:
        f = _load_sidecar(r['rundir'], 'fairness_metrics.json')
        if not f:
            continue
        fair_rows.append({'suite': suite, 'name': r['name'], 'seed': r['seed'],
                          'worst_client_metric': f.get('worst_client_metric'),
                          'bottom_10pct_client_mean': f.get('bottom_10pct_client_mean'),
                          'bottom_20pct_client_mean': f.get('bottom_20pct_client_mean'),
                          'best_client_metric': f.get('best_client_metric'),
                          'client_metric_std': f.get('client_metric_std'),
                          'client_metric_iqr': f.get('client_metric_iqr'),
                          'performance_gap': f.get('performance_gap'),
                          'unavailable_client_count': f.get('unavailable_client_count'),
                          'zero_anomaly_train_client_count': f.get(
                              'zero_anomaly_train_client_count'),
                          'zero_anomaly_test_client_count': f.get(
                              'zero_anomaly_test_client_count')})
    if fair_rows:
        _write_csv(os.path.join(outdir, 'fairness_analysis.csv'),
                   list(fair_rows[0].keys()), fair_rows)

    # ---- cache_analysis.csv ----
    cache_rows = []
    for r in rows:
        c = _load_sidecar(r['rundir'], 'cache_metrics.json')
        res = _load_sidecar(r['rundir'], 'resource_usage.json')
        if not c:
            continue
        rt = res.get('round_times_sec', [])
        cache_rows.append({'suite': suite, 'name': r['name'], 'seed': r['seed'],
                           'mode': c.get('mode'), 'scope': c.get('scope'),
                           'anchor_sampling_mode': c.get('anchor_sampling_mode'),
                           'local_epochs': c.get('local_epochs'),
                           'requests': c.get('stats', {}).get('hits', 0)
                           + c.get('stats', {}).get('misses', 0),
                           'hits': c.get('stats', {}).get('hits'),
                           'misses': c.get('stats', {}).get('misses'),
                           'hit_rate': c.get('stats', {}).get('hit_rate'),
                           'entries': c.get('stats', {}).get('entries'),
                           'round_time_mean_sec': round(np.mean(rt), 3) if rt else ''})
    if cache_rows:
        _write_csv(os.path.join(outdir, 'cache_analysis.csv'),
                   list(cache_rows[0].keys()), cache_rows)

    # ---- 阶段三: blocked paired (ckr_enriched_independent_replication) ----
    if suite == 'ckr_enriched_independent_replication':
        from experiment_stats import blocked_paired_analysis
        bp_rows = []
        for p in range(5):
            dyn = f'DYNAMIC_p{p}'
            sta = f'STATIC_p{p}'
            d_rows = {r['seed']: r for r in by_name.get(dyn, [])}
            s_rows = {r['seed']: r for r in by_name.get(sta, [])}
            common = sorted(set(d_rows) & set(s_rows))
            fv = [d_rows[s]['pooled_pr_auc'] for s in common]
            bv = [s_rows[s]['pooled_pr_auc'] for s in common]
            bp = blocked_paired_analysis(fv, bv, [p] * len(common),
                                         common, n_boot=1000, seed=0)
            for blk in bp['per_block']:
                bp_rows.append({'partition': p, 'n': blk['n'],
                                'mean_diff': round(blk['mean_diff'], 4),
                                'wins': blk['wins']})
        # 总 blocked 分析
        all_fv, all_bv, all_p = [], [], []
        for p in range(5):
            d_rows = {r['seed']: r for r in by_name.get(f'DYNAMIC_p{p}', [])}
            s_rows = {r['seed']: r for r in by_name.get(f'STATIC_p{p}', [])}
            common = sorted(set(d_rows) & set(s_rows))
            for s in common:
                all_fv.append(d_rows[s]['pooled_pr_auc'])
                all_bv.append(s_rows[s]['pooled_pr_auc'])
                all_p.append(p)
        bp = blocked_paired_analysis(all_fv, all_bv, all_p,
                                     list(range(len(all_fv))), n_boot=2000, seed=0)
        ov = bp['overall']
        bp_rows.append({'partition': 'OVERALL', 'n': ov['n_pairs'],
                        'mean_diff': round(ov['mean_diff'], 4),
                        'median_diff': round(ov['median_diff'], 4),
                        'ci95_low': round(ov['ci95_low'], 4),
                        'ci95_high': round(ov['ci95_high'], 4),
                        'partition_cluster_ci_low':
                            round(ov['partition_cluster_ci_low'], 4),
                        'partition_cluster_ci_high':
                            round(ov['partition_cluster_ci_high'], 4),
                        'wilcoxon_p': round(ov['wilcoxon_p'], 4),
                        'sign_test_p': round(ov['sign_test_p'], 4),
                        'cohen_dz': round(ov['cohen_dz'], 4),
                        'prob_improvement': round(ov['prob_improvement'], 4),
                        'prob_gain_gt': ov['prob_gain_gt'],
                        'variance_partition': round(ov['variance_partition'], 6),
                        'variance_residual': round(ov['variance_residual'], 6),
                        'n_partitions': ov['n_partitions']})
        _header = sorted({k for r in bp_rows for k in r.keys()})
        _write_csv(os.path.join(outdir, 'blocked_paired_comparisons.csv'),
                   _header, bp_rows)

    # ---- teacher_mixture_analysis.csv ----
    tm_rows = []
    for r in rows:
        p = os.path.join(r['rundir'], 'teacher_mixture.jsonl')
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p)]
        if not recs:
            continue
        n_c = sum(1 for k in recs[0] if k.startswith('class_') and 'weights' not in k)
        entropies = [rec[f'class_{c}']['entropy'] for rec in recs
                     for c in range(n_c)]
        diff_static = [rec[f'class_{c}']['diff_from_static'] for rec in recs
                       for c in range(n_c)]
        diff_prev = [rec[f'class_{c}']['diff_from_previous']
                     for rec in recs for c in range(n_c)
                     if rec[f'class_{c}']['diff_from_previous'] is not None]
        tm_rows.append({'suite': suite, 'name': r['name'], 'seed': r['seed'],
                        'mean_entropy': round(sum(entropies) / len(entropies), 5),
                        'mean_diff_from_static':
                            round(sum(diff_static) / len(diff_static), 5),
                        'mean_diff_from_previous':
                            round(sum(diff_prev) / len(diff_prev), 5)
                            if diff_prev else ''})
    if tm_rows:
        _write_csv(os.path.join(outdir, 'teacher_mixture_analysis.csv'),
                   list(tm_rows[0].keys()), tm_rows)

    # ---- distillation_chain_analysis.csv ----
    dc_rows = []
    for r in rows:
        p = os.path.join(r['rundir'], 'distillation_diagnostics.jsonl')
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p)]
        if not recs:
            continue
        def _mean(k):
            vals = [x[k] for x in recs if x.get(k) is not None]
            return round(sum(vals) / len(vals), 6) if vals else ''
        dc_rows.append({'suite': suite, 'name': r['name'], 'seed': r['seed'],
                        'distill_grad_norm_mean': _mean('distill_grad_norm'),
                        'global_param_update_norm_mean':
                            _mean('global_param_update_norm'),
                        'global_prediction_change_val_mean':
                            _mean('global_prediction_change_val'),
                        'distill_loss_mean': _mean('distill_loss'),
                        'fake_feature_std_mean': _mean('fake_feature_std'),
                        'fake_graph_mean_degree_mean':
                            _mean('fake_graph_mean_degree')})
    if dc_rows:
        _write_csv(os.path.join(outdir, 'distillation_chain_analysis.csv'),
                   list(dc_rows[0].keys()), dc_rows)

    # ---- fairness_pareto.csv ----
    fp_rows = []
    for r in rows:
        fa = _load_sidecar(r['rundir'], 'fairness_metrics.json')
        em = extract_metrics(r['final']) if r['final'] else {}
        if not fa or not em:
            continue
        fp_rows.append({'suite': suite, 'name': r['name'], 'seed': r['seed'],
                        'pooled_pr_auc': em.get('pooled_pr_auc'),
                        'worst_client_metric': fa.get('worst_client_metric'),
                        'bottom_10pct_client_mean':
                            fa.get('bottom_10pct_client_mean'),
                        'bottom_20pct_client_mean':
                            fa.get('bottom_20pct_client_mean'),
                        'client_metric_std': fa.get('client_metric_std'),
                        'performance_gap': fa.get('performance_gap'),
                        'zero_anomaly_test_client_count':
                            fa.get('zero_anomaly_test_client_count')})
    if fp_rows:
        _write_csv(os.path.join(outdir, 'fairness_pareto.csv'),
                   list(fp_rows[0].keys()), fp_rows)

    # ---- dataset_manifests.csv + blocked_external_resources.csv ----
    man_rows = []
    for r in rows:
        m = _load_sidecar(r['rundir'], 'dataset_manifest.json')
        if m:
            man_rows.append({'suite': suite, 'name': r['name'], 'seed': r['seed'],
                             'dataset': m.get('dataset'), 'source': m.get('source'),
                             'num_nodes': m.get('num_nodes'),
                             'num_edges': m.get('num_edges'),
                             'feature_dim': m.get('feature_dim'),
                             'original_num_classes':
                                 m.get('original_num_classes')})
    if man_rows:
        _write_csv(os.path.join(outdir, 'dataset_manifests.csv'),
                   list(man_rows[0].keys()), man_rows)

    # ---- blocked_external_resources.csv (全库扫描) ----
    ext_rows = []
    for s in sorted(os.listdir(RUNS_ROOT)):
        sb = os.path.join(RUNS_ROOT, s)
        if not os.path.isdir(sb) or s.startswith('_'):
            continue
        for n0 in sorted(os.listdir(sb)):
            d0 = os.path.join(sb, n0)
            if not os.path.isdir(d0):
                continue
            for sd0 in sorted(os.listdir(d0)):
                sp0 = os.path.join(d0, sd0, 'status.json')
                if os.path.exists(sp0):
                    try:
                        st = json.load(open(sp0))
                    except Exception:
                        continue
                    if st.get('status') == 'external_blocked':
                        ext_rows.append({'suite': s, 'experiment': n0,
                                         'seed': sd0,
                                         'returncode': st.get('returncode'),
                                         'stderr_tail':
                                             (st.get('stderr_tail') or '')[-200:]})
    if ext_rows:
        _write_csv(os.path.join(RUNS_ROOT, suite, 'blocked_external_resources.csv'),
                   list(ext_rows[0].keys()), ext_rows)

    # ---- numerical_equivalence.csv (benchmark suite) ----
    if SUITES.get(suite, {}).get('runner') == 'benchmark':
        ne_rows = []
        for r in rows:
            ne = _load_sidecar(r['rundir'], 'numerical_equivalence.json')
            if not ne:
                continue
            ne_rows.append({'name': r['name'], 'seed': r['seed'],
                            'hidden_dim': ne.get('hidden_dim'),
                            'diffusion_steps': ne.get('diffusion_steps'),
                            'mode': ne.get('mode'),
                            'gpu_available': ne.get('gpu_available'),
                            'step_time_sec': ne.get('step_time_sec'),
                            'peak_memory_allocated_mb': ne.get(
                                'peak_memory_allocated_mb'),
                            'peak_memory_reserved_mb': ne.get(
                                'peak_memory_reserved_mb'),
                            'equivalence': ne.get('numerical_equivalence', {}).get(
                                'equivalence'),
                            'output_max_abs_diff': ne.get(
                                'numerical_equivalence', {}).get('output_max_abs_diff'),
                            'gradient_max_abs_diff': ne.get(
                                'numerical_equivalence', {}).get(
                                    'gradient_max_abs_diff')})
        if ne_rows:
            _write_csv(os.path.join(outdir, 'numerical_equivalence.csv'),
                       list(ne_rows[0].keys()), ne_rows)


# ---------------------------------------------------------------------
#  可视化 (英文标签)
# ---------------------------------------------------------------------
def make_plots(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiment_stats import summarize

    rows = collect_runs(suite)
    if not rows:
        print(f"[plots] {suite}: no runs")
        return
    plot_dir = os.path.join(RUNS_ROOT, suite, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    by_name = {}
    for r in rows:
        if not r['final'].get('metrics') and r['final'].get('task_mode') is None:
            continue  # benchmark 运行 (无训练指标)
        by_name.setdefault(r['name'], []).append(r)
    task_mode = next((r['final'].get('task_mode') for r in rows
                      if r['final'].get('task_mode')), None)
    metrics = (['pooled_pr_auc', 'pooled_roc_auc', 'client_macro_pr_auc']
               if task_mode == 'anomaly_binary'
               else ['pooled_accuracy', 'pooled_macro_f1'])

    # 1) main metrics mean±std bar chart
    names = sorted(by_name)
    for m in metrics:
        plt.figure(figsize=(8, 4.5))
        means, stds = [], []
        src = 'client_macro' if m.startswith('client_macro_') else 'pooled'
        key = m.replace('client_macro_', '')
        for n in names:
            s = summarize([r['final']['metrics'][src].get(key) for r in by_name[n]])
            means.append(s['mean']); stds.append(s['std'])
        plt.bar(range(len(names)), means, yerr=stds, capsize=4, color='steelblue')
        plt.xticks(range(len(names)), names, rotation=30, ha='right')
        plt.ylabel(f'{m} (%)')
        plt.title(f'{suite}: {m} (mean ± std over seeds)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'main_{m}.png'), dpi=120)
        plt.close()
    print(f"  plots: main_*.png")

    # 2) CKR curves over rounds + fallback ratio (from ckr.jsonl)
    ck = [(r['name'], r['seed'], r['rundir']) for r in rows
          if os.path.exists(os.path.join(r['rundir'], 'ckr.jsonl'))]
    if ck:
        name0 = ck[0][0]
        recs_by_round = {}
        for name, seed, rundir in ck:
            if name != name0:
                continue
            for line in open(os.path.join(rundir, 'ckr.jsonl')):
                rec = json.loads(line)
                recs_by_round.setdefault(rec['round'], []).append(rec)
        if recs_by_round:
            rounds = sorted(recs_by_round)
            for cls in sorted({r['class_id'] for r in recs_by_round[rounds[0]]}):
                plt.figure(figsize=(7, 4))
                for name, seed, rundir in ck[:5]:
                    by_r = {}
                    for line in open(os.path.join(rundir, 'ckr.jsonl')):
                        rec = json.loads(line)
                        if rec['class_id'] == cls:
                            by_r[rec['round']] = rec.get('ema_ckr', rec.get('ema'))
                    xs = sorted(by_r)
                    plt.plot(xs, [by_r[x] for x in xs], marker='o',
                             label=f'seed {seed}')
                plt.xlabel('communication round'); plt.ylabel('EMA CKR')
                plt.title(f'CKR trajectory class {cls} ({name0})')
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'ckr_class_{cls}.png'), dpi=120)
                plt.close()
            # fallback ratio
            plt.figure(figsize=(7, 4))
            for name, seed, rundir in ck[:5]:
                by_r = {}
                for line in open(os.path.join(rundir, 'ckr.jsonl')):
                    rec = json.loads(line)
                    by_r.setdefault(rec['round'], []).append(1.0 if rec['fallback'] else 0.0)
                xs = sorted(by_r)
                plt.plot(xs, [np.mean(by_r[x]) for x in xs], marker='o',
                         label=f'seed {seed}')
            plt.xlabel('communication round'); plt.ylabel('fallback ratio')
            plt.title(f'CKR fallback ratio per round ({name0})')
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'ckr_fallback_ratio.png'), dpi=120)
            plt.close()
        print("  plots: ckr_class_*.png, ckr_fallback_ratio.png")

    # 3) resource comparison (backprop/rwr suites)
    if suite in ('backprop_ablation', 'rwr_ablation'):
        res_rows = []
        for name, seeds_by in by_name.items():
            for r in seeds_by:
                rp = os.path.join(r['rundir'], 'resource_usage.json')
                res = json.load(open(rp)) if os.path.exists(rp) else {}
                rt = res.get('round_times_sec', [])
                res_rows.append({'name': name, 'seed': r['seed'],
                                 'mem': res.get('gen_peak_mem_mb_max'),
                                 'time': np.mean(rt) if rt else float('nan'),
                                 'hit': res.get('rwr_cache_hit_rate_final')})
        for metric, key, ylab, fname in [
                ('peak memory (MB)', 'mem', 'peak generator memory (MB)',
                 'resource_memory.png'),
                ('mean round time (s)', 'time', 'mean round time (s)',
                 'resource_time.png')]:
            plt.figure(figsize=(7, 4))
            names = sorted({r['name'] for r in res_rows})
            vals = {n: summarize([r[key] for r in res_rows if r['name'] == n])
                    for n in names}
            plt.bar(range(len(names)), [vals[n]['mean'] for n in names],
                    yerr=[vals[n]['std'] for n in names], capsize=4, color='darkorange')
            plt.xticks(range(len(names)), names, rotation=20)
            plt.ylabel(ylab); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, fname), dpi=120)
            plt.close()
        if suite == 'rwr_ablation':
            plt.figure(figsize=(7, 4))
            names = sorted({r['name'] for r in res_rows})
            hits = {n: summarize([r['hit'] for r in res_rows if r['name'] == n])
                    for n in names}
            times = {n: summarize([r['time'] for r in res_rows if r['name'] == n])
                     for n in names}
            plt.scatter([times[n]['mean'] for n in names],
                        [hits[n]['mean'] for n in names], s=80)
            for n in names:
                plt.annotate(n, (times[n]['mean'], hits[n]['mean']), fontsize=9)
            plt.xlabel('mean round time (s)'); plt.ylabel('RWR cache hit rate')
            plt.title('RWR cache: hit rate vs time')
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'rwr_hit_vs_time.png'), dpi=120)
            plt.close()
        print("  plots: resource_*.png, rwr_hit_vs_time.png")

    # 4.5) CKR availability vs dynamic gain (ckr_support_mechanism)
    if suite == 'ckr_support_mechanism' and os.path.exists(
            os.path.join(RUNS_ROOT, suite, 'mechanism_pairs.csv')):
        mp = SUITES[suite].get('mechanism_pairs', [])
        avail_rows = []
        for dyn, stat, mode_label in mp:
            for r in rows:
                if r['name'] != dyn:
                    continue
                ck = {}
                ap = os.path.join(r['rundir'], 'ckr_availability.json')
                if os.path.exists(ap):
                    ck = json.load(open(ap))
                ag = ck.get('rounds', [])
                if not ag:
                    continue
                avail = sum(x['available_ratio'] for x in ag) / len(ag)
                s_rows = {rr['seed']: rr for rr in by_name.get(stat, [])}
                d_rows = {rr['seed']: rr for rr in by_name.get(dyn, [])}
                gain = None
                if r['seed'] in s_rows and r['seed'] in d_rows:
                    try:
                        gain = (float(d_rows[r['seed']]['final']['metrics']
                                      ['pooled']['pr_auc'])
                                - float(s_rows[r['seed']]['final']['metrics']
                                        ['pooled']['pr_auc']))
                    except (TypeError, ValueError, KeyError):
                        gain = None
                if gain is not None:
                    avail_rows.append({'support_mode': mode_label, 'seed': r['seed'],
                                       'available_ratio': avail, 'gain': gain})
        if avail_rows:
            plt.figure(figsize=(7, 5))
            for ml in sorted({a['support_mode'] for a in avail_rows}):
                xs = [a['available_ratio'] for a in avail_rows if a['support_mode'] == ml]
                ys = [a['gain'] for a in avail_rows if a['support_mode'] == ml]
                plt.scatter(xs, ys, label=ml, s=40, alpha=0.7)
            plt.axhline(0, color='grey', lw=0.8)
            plt.xlabel('CKR available ratio (mean over rounds)')
            plt.ylabel('dynamic - static pooled PR-AUC gain')
            plt.title('CKR availability vs dynamic gain (seed-level)')
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'ckr_availability_vs_gain.png'), dpi=120)
            plt.close()
            # 保存绘图源数据
            import csv as _csv
            with open(os.path.join(plot_dir, 'ckr_availability_vs_gain_data.csv'),
                      'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=['support_mode', 'seed',
                                                   'available_ratio', 'gain'])
                w.writeheader()
                w.writerows(avail_rows)
            print("  plots: ckr_availability_vs_gain.png (+data csv)")

    # 4.6) 显存/时间曲线 (checkpoint_memory_scaling)
    if suite == 'checkpoint_memory_scaling':
        ne_rows = []
        for r in rows:
            p = os.path.join(r['rundir'], 'numerical_equivalence.json')
            if os.path.exists(p):
                ne_rows.append(json.load(open(p)))
        if ne_rows:
            for dim in sorted({n['hidden_dim'] for n in ne_rows}):
                plt.figure(figsize=(7, 4.5))
                for mode in ('full', 'checkpointed'):
                    xs = sorted({n['diffusion_steps'] for n in ne_rows
                                 if n['mode'] == mode})
                    mems = [next(n['peak_memory_allocated_mb'] for n in ne_rows
                                 if n['mode'] == mode and n['diffusion_steps'] == s
                                 and n['hidden_dim'] == dim and n.get('gpu_available'))
                            for s in xs]
                    plt.plot(xs, mems, marker='o', label=mode)
                plt.xlabel('diffusion steps'); plt.ylabel('peak memory allocated (MB)')
                plt.title(f'Generator peak memory vs steps (hidden={dim})')
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'memory_vs_steps_h{dim}.png'),
                            dpi=120)
                plt.close()
            for steps in sorted({n['diffusion_steps'] for n in ne_rows}):
                plt.figure(figsize=(7, 4.5))
                for mode in ('full', 'checkpointed'):
                    xs = sorted({n['hidden_dim'] for n in ne_rows
                                 if n['mode'] == mode})
                    mems = [next(n['peak_memory_allocated_mb'] for n in ne_rows
                                 if n['mode'] == mode and n['hidden_dim'] == h
                                 and n['diffusion_steps'] == steps
                                 and n.get('gpu_available'))
                            for h in xs]
                    plt.plot(xs, mems, marker='o', label=mode)
                plt.xlabel('hidden dim'); plt.ylabel('peak memory allocated (MB)')
                plt.title(f'Generator peak memory vs hidden dim (steps={steps})')
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'memory_vs_hidden_s{steps}.png'),
                            dpi=120)
                plt.close()
            plt.figure(figsize=(7, 4.5))
            for mode in ('full', 'checkpointed'):
                xs = sorted({n['hidden_dim'] for n in ne_rows if n['mode'] == mode})
                ts = [next(n['step_time_sec'] for n in ne_rows
                           if n['mode'] == mode and n['hidden_dim'] == h
                           and n['diffusion_steps'] == 50) for h in xs]
                plt.plot(xs, ts, marker='o', label=mode)
            plt.xlabel('hidden dim'); plt.ylabel('step time (s)')
            plt.title('Runtime-memory tradeoff (steps=50)')
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'runtime_tradeoff.png'), dpi=120)
            plt.close()
            print("  plots: memory_vs_*.png, runtime_tradeoff.png")

    # 4.7) RWR hit rate vs speedup
    if suite == 'rwr_cache_effective':
        ca_path = os.path.join(RUNS_ROOT, suite, 'cache_analysis.csv')
        if os.path.exists(ca_path):
            import csv as _csv
            cas = list(_csv.DictReader(open(ca_path)))
            plt.figure(figsize=(7, 4.5))
            xs, ys, labels = [], [], []
            for name in sorted({c['name'] for c in cas}):
                cc = [c for c in cas if c['name'] == name]
                hits = [float(c['hit_rate']) for c in cc if c['hit_rate'] not in ('', 'None')]
                times = [float(c['round_time_mean_sec']) for c in cc
                         if c['round_time_mean_sec']]
                if hits and times:
                    xs.append(sum(hits) / len(hits))
                    ys.append(sum(times) / len(times))
                    labels.append(name)
            plt.scatter(xs, ys, s=60)
            for x, y, l in zip(xs, ys, labels):
                plt.annotate(l, (x, y), fontsize=7)
            plt.xlabel('RWR cache hit rate'); plt.ylabel('mean round time (s)')
            plt.title('RWR cache: hit rate vs round time')
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'rwr_hit_vs_time.png'), dpi=120)
            plt.close()
            print("  plots: rwr_hit_vs_time.png")

    # 4.8) worst-client / bottom-10% (fairness)
    fa_path = os.path.join(RUNS_ROOT, suite, 'fairness_analysis.csv')
    if os.path.exists(fa_path):
        import csv as _csv
        fas = list(_csv.DictReader(open(fa_path)))
        plt.figure(figsize=(9, 4.5))
        for name in sorted({f['name'] for f in fas}):
            ff = [f for f in fas if f['name'] == name]
            worst = [float(f['worst_client_metric']) for f in ff
                     if f['worst_client_metric'] not in ('', 'None')]
            b10 = [float(f['bottom_10pct_client_mean']) for f in ff
                   if f['bottom_10pct_client_mean'] not in ('', 'None')]
            if worst and b10:
                plt.bar(f'{name}\nworst', sum(worst) / len(worst), alpha=0.7)
                plt.bar(f'{name}\nbottom10', sum(b10) / len(b10), alpha=0.7)
        plt.ylabel('metric (%)'); plt.xticks(rotation=45, ha='right')
        plt.title(f'{suite}: worst-client and bottom-10% performance')
        plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'worst_client.png'), dpi=120)
        plt.close()
        print("  plots: worst_client.png")

    # 4.9) 动态-静态分区块差异 (enriched replication)
    bp_path = os.path.join(RUNS_ROOT, suite, 'blocked_paired_comparisons.csv')
    if suite == 'ckr_enriched_independent_replication' and os.path.exists(bp_path):
        import csv as _csv
        bps = list(_csv.DictReader(open(bp_path)))
        blocks = [b for b in bps if b['partition'] != 'OVERALL']
        if blocks:
            plt.figure(figsize=(7, 4.5))
            xs = [b['partition'] for b in blocks]
            ys = [float(b['mean_diff']) for b in blocks]
            plt.bar(xs, ys, color=['steelblue' if v > 0 else 'salmon' for v in ys])
            plt.axhline(0, color='grey', lw=0.8)
            plt.xlabel('partition block (partition_seed)')
            plt.ylabel('dynamic - static pooled PR-AUC (pp)')
            plt.title(f'Dynamic minus static by partition block '
                      f'({len(blocks)} blocks × model seeds)')
            plt.grid(axis='y', alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir,
                                     'dynamic_minus_static_by_block.png'), dpi=120)
            plt.close()
            # 绘图源数据
            with open(os.path.join(plot_dir,
                                   'dynamic_minus_static_by_block_data.csv'),
                      'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=list(blocks[0].keys()))
                w.writeheader()
                w.writerows(blocks)
            print("  plots: dynamic_minus_static_by_block.png (+data)")

    # 4.10) 蒸馏因果链消融
    dc_path = os.path.join(RUNS_ROOT, suite, 'distillation_chain_analysis.csv')
    if suite == 'distillation_causal_chain' and os.path.exists(dc_path):
        import csv as _csv
        dcs = list(_csv.DictReader(open(dc_path)))
        names = sorted({d['name'] for d in dcs if not d['name'].endswith('_long')})
        vals = {}
        for n in names:
            vv = [float(d['distill_grad_norm_mean']) for d in dcs
                  if d['name'] == n and d['distill_grad_norm_mean']]
            if vv:
                vals[n] = sum(vv) / len(vv)
        if vals:
            plt.figure(figsize=(8, 4.5))
            xs = range(len(vals))
            plt.bar(xs, [vals[n] for n in vals], color='mediumseagreen')
            plt.xticks(list(xs), list(vals), rotation=30)
            plt.ylabel('mean distill grad norm')
            plt.title(f'Distillation gradient norm by config ({suite})')
            plt.grid(axis='y', alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'distill_grad_norm.png'), dpi=120)
            plt.close()
            print("  plots: distill_grad_norm.png")

    # 4.11) 公平性 Pareto
    fp_path = os.path.join(RUNS_ROOT, suite, 'fairness_pareto.csv')
    if suite == 'worst_client_fairness' and os.path.exists(fp_path):
        import csv as _csv
        fps = list(_csv.DictReader(open(fp_path)))
        plt.figure(figsize=(7, 5))
        for name in sorted({f['name'] for f in fps}):
            pts = [(float(f['pooled_pr_auc']), float(f['worst_client_metric']))
                   for f in fps if f['name'] == name
                   and f['pooled_pr_auc'] not in ('', 'None')
                   and f['worst_client_metric'] not in ('', 'None')]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                plt.scatter(xs, ys, label=name, s=40, alpha=0.8)
                plt.annotate(name, (sum(xs) / len(xs), sum(ys) / len(ys)),
                             fontsize=8)
        plt.xlabel('pooled PR-AUC (%)')
        plt.ylabel('worst-client PR-AUC (%)')
        plt.title('Pooled vs worst-client Pareto frontier (seed-level)')
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'fairness_pareto.png'), dpi=120)
        plt.close()
        print("  plots: fairness_pareto.png")

    # 4) anomaly per-client recall (main suite)
    if task_mode == 'anomaly_binary' and os.path.exists(
            os.path.join(RUNS_ROOT, suite, 'per_client_results.csv')):
        import csv as _csv
        pcs = list(_csv.DictReader(open(os.path.join(RUNS_ROOT, suite,
                                                     'per_client_results.csv'))))
        if pcs:
            plt.figure(figsize=(9, 4.5))
            for name in sorted({r['name'] for r in pcs})[:6]:
                vals = [float(r['recall']) for r in pcs if r['name'] == name
                        and r['recall'] not in ('', 'nan')]
                s = summarize(vals)
                plt.bar(name, s['mean'], yerr=s['std'], capsize=3, alpha=0.7)
            plt.xticks(rotation=30, ha='right')
            plt.ylabel('client-macro anomaly recall (%)')
            plt.title(f'{suite}: anomaly recall (mean ± std)')
            plt.grid(axis='y', alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'anomaly_recall.png'), dpi=120)
            plt.close()
            print("  plots: anomaly_recall.png")


# ---------------------------------------------------------------------
#  入口
# ---------------------------------------------------------------------
def prepare_proxy():
    os.makedirs(os.path.dirname(PROXY_CKPT), exist_ok=True)
    cli = [PY, os.path.join(ROOT, 'pretrain_diffusion.py'),
           '--proxy_n_samples', '4000', '--proxy_epochs', '60',
           '--proxy_feature_dim', '512', '--target_feature_dim', '1433',
           '--diffusion_steps', '4', '--diffusion_hidden', '32',
           '--proxy_pretrain_mode', 'unconditional', '--num_classes', '7',
           '--proxy_checkpoint', PROXY_CKPT, '--seed', '0']
    print(f"[prepare-proxy] {' '.join(cli)}")
    subprocess.run(cli, cwd=ROOT, check=True)
    print(f"[prepare-proxy] done: {PROXY_CKPT}")


def plan_suite(suite):
    """dry-run 计划输出: suite/实验/种子/总数/复用/新增/输出路径/关键配置差异。"""
    suite_cfg = SUITES[suite]
    total = reuse = new = 0
    lines = [f"SUITE: {suite}", f"  reference: {suite_cfg.get('reference', '-')}",
             f"  runner: {suite_cfg.get('runner', 'train')}",
             f"  seeds: {suite_cfg.get('seeds', SEEDS)}"]
    for name in sorted(suite_cfg['experiments']):
        exp = suite_cfg['experiments'][name]
        seeds = exp['seeds'] or suite_cfg.get('seeds', SEEDS)
        n_ok = 0
        for seed in seeds:
            rundir = os.path.join(RUNS_ROOT, suite, name, f'seed_{seed}')
            sp = os.path.join(rundir, 'status.json')
            kw = None
            cfgp = os.path.join(rundir, 'config.json')
            if os.path.exists(cfgp):
                try:
                    kw = json.load(open(cfgp)).get('kwargs')
                except Exception:
                    kw = None
            fp = _success_marker(suite_cfg.get('runner', 'train'), rundir, kw)
            if os.path.exists(sp) and os.path.exists(fp):
                try:
                    st = json.load(open(sp))
                    if st.get('status') == 'success':
                        n_ok += 1
                except Exception:
                    pass
        total += len(seeds)
        reuse += n_ok
        new += len(seeds) - n_ok
        kw = exp['kwargs']
        diffs = [kw[i] for i in range(0, len(kw), 2)
                 if kw[i] not in ('--task_mode', '--normal_classes', '--anomaly_classes')]
        lines.append(f"  {name}: seeds={seeds} reuse={n_ok} new={len(seeds) - n_ok} "
                     f"out=runs/{suite}/{name}/seed_<s>/ "
                     f"key_cfg={diffs}")
    lines.append(f"TOTAL: {total} runs, reuse {reuse}, new {new}")
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', type=str, default='')
    ap.add_argument('--list-suites', action='store_true')
    ap.add_argument('--prepare-proxy', action='store_true')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--plan', action='store_true', help='dry-run 计划输出 (不运行)')
    ap.add_argument('--resume-failed', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--dry', action='store_true',
                    help='小规模覆盖注入 (2 客户端/1 轮/小模型), 仅用于功能验证')
    ap.add_argument('--limit', type=int, default=0,
                    help='每个实验最多运行的种子数 (0=全部)')
    opts = ap.parse_args()

    if opts.list_suites:
        for s in SUITES:
            exps = sorted(SUITES[s]['experiments'])
            print(f"{s}: {len(exps)} experiments, "
                  f"seeds={SUITES[s].get('seeds', SEEDS)}, "
                  f"runner={SUITES[s].get('runner', 'train')}")
        return
    if opts.prepare_proxy:
        prepare_proxy()
        return
    if not opts.suite:
        ap.error('--suite 必填 (--list-suites 查看)')
    if opts.suite not in SUITES:
        ap.error(f"未知 suite: {opts.suite}")
    if opts.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('[cpu] 强制 CPU 模式')

    suite_cfg = SUITES[opts.suite]
    if opts.plan:
        plan_suite(opts.suite)
        return
    if not opts.aggregate and not opts.plots:
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        for name in sorted(suite_cfg['experiments']):
            exp = suite_cfg['experiments'][name]
            seeds = exp['seeds'] or suite_cfg.get('seeds', SEEDS)
            if opts.limit:
                seeds = seeds[:opts.limit]
            for seed in seeds:
                st = run_one(suite_cfg, exp, seed, opts)
                results[st if st in results else 'failed'] += 1
        print(f"[suite {opts.suite}] done: {results}")
    if opts.aggregate:
        aggregate_suite(opts.suite)
    if opts.plots:
        make_plots(opts.suite)


if __name__ == '__main__':
    main()
