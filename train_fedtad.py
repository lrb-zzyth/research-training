"""
面向子图异质性的联邦图异常检测方法

五步骤流程:
  S1 - 拓扑感知节点嵌入与类别知识可靠性计算
  S2 - 客户端本地联合训练 (加权CE + 子图-子图跨视图对比)
  S3 - 服务器基础加权聚合 (FedAvg)
  S4 - 教师引导扩散式伪图生成与拓扑感知对抗蒸馏
  S5 - 双重机制迭代收敛

任务模式:
  --task_mode multiclass     : 多分类节点分类 (Cora 默认 7 类)
  --task_mode anomaly_binary : 二分类异常检测 (标签映射为 0=正常, 1=异常)

动态 CKR (--ckr_mode hybrid_dynamic):
  每轮客户端本地训练完成后, 在 reliability_idx (从 train 中分层划分,
  默认不使用 val) 上计算每类别 F1/召回率/置信度,
  与静态拓扑先验加权, 再经 EMA 平滑, 按类别归一化后用于
  生成器语义损失/分歧损失与全局蒸馏。
  客户端仅上传每类别分数 + support + available mask, 不上传节点/标签/Data。

每轮顺序 (动态 CKR 视角):
  1 广播全局模型 → 2 fit_idx 本地训练 → 3 训练结束 → 4 reliability_idx 计算指标
  → 5 仅上传聚合指标 → 6 服务器更新 DynamicCKRTracker → 7 FedAvg
  → 8 生成器更新(本轮 CKR) → 9 蒸馏(本轮 CKR) → 10 校正全局模型
  → 11 val_idx 模型选择 → 12 下一轮
"""

import argparse
import os
import sys
import json
import time
import warnings
import math
import random
import hashlib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

from util.task_util import (
    cal_topo_emb,
    multiclass_metrics,
    binary_anomaly_metrics,
    edge_perturbation,
    rwr_subgraph_sampling,
    subgraph_contrastive_loss,
    compute_class_weights,
)
from util.base_util import seed_everything, load_dataset
from util.dynamic_ckr import (
    DynamicCKRTracker,
    compute_per_class_reliability_metrics,
)
from util.data_split import (
    stratified_reliability_split,
    stratified_split,
    stratified_split_with_support,
    anomaly_holdout_boost_split,
    split_report,
)
from util.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_final_checkpoint,
)
from util.rwr_cache import RWRSubgraphCache
from util.kl_utils import categorical_kl_from_logits
from util.projection_utils import (
    project_to_pretrain_radius as _stable_project_to_pretrain_radius,
    diagnostic_l2_norm,
)
from model import GCN, ConditionalDiffusionGenerator

warnings.filterwarnings('ignore')

# =====================================================================
#  参数配置
# =====================================================================

parser = argparse.ArgumentParser()

# 环境
parser.add_argument('--seed', type=int, default=2024,
                    help='兼容入口: 未显式指定时展开为全部四类种子')
parser.add_argument('--partition_seed', type=int, default=-1,
                    help='Louvain 社区划分随机种子 (-1=用 --seed)')
parser.add_argument('--allocation_seed', type=int, default=-1,
                    help='社区分配与 enriched 重分配随机种子 (-1=用 --seed)')
parser.add_argument('--split_seed', type=int, default=-1,
                    help='客户端内 train/val/test 划分随机种子 (-1=用 --seed)')
parser.add_argument('--model_seed', type=int, default=-1,
                    help='模型初始化/本地训练/伪节点随机种子 (-1=用 --seed)')
parser.add_argument('--bootstrap_seed', type=int, default=-1,
                    help='统计重采样随机种子 (-1=用 --seed)')
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--gpu_id', type=str, default='0')

# ---- 数据来源 (离线导入) ----
parser.add_argument('--dataset_source', type=str, default='auto',
                    choices=['auto', 'pyg_cache', 'local_raw', 'local_processed'],
                    help='auto: 本地缓存优先, 缺失时尝试下载, 失败标记 external_blocked; '
                         'pyg_cache: 只读缓存不联网; local_raw/local_processed: 本地数据')
parser.add_argument('--offline', action='store_true', default=False,
                    help='完全离线 (等价 dataset_source=pyg_cache)')
parser.add_argument('--dataset_manifest', type=str, default='',
                    help='数据集 manifest JSON 输出路径')
parser.add_argument('--dataset_checksum', type=str, default='',
                    help='校验原始文件 sha1 (空=不校验)')

# ---- frozen split artifact ----
parser.add_argument('--build_split_artifact', type=str, default='',
                    help='构建 frozen split artifact 到该目录并退出 (不训练)')
parser.add_argument('--frozen_split', type=str, default='',
                    help='从 artifact 加载冻结划分 (不运行 Louvain/不重划分)')
parser.add_argument('--dataset', type=str, default="Cora")
parser.add_argument('--partition', type=str, default="Louvain",
                    choices=["Louvain", "Metis"])
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=64)

# 任务模式
parser.add_argument('--task_mode', type=str, default='anomaly_binary',
                    choices=['multiclass', 'anomaly_binary'],
                    help='anomaly_binary: 二分类异常检测 (正常=0, 异常=1, 默认核心模式); '
                         'multiclass: 多分类 (仅兼容模式)')
parser.add_argument('--normal_classes', type=str, default='0,1,2,3',
                    help='anomaly_binary 模式: 正常类ID, 逗号分隔, 如 "0,1,2,3"')
parser.add_argument('--anomaly_classes', type=str, default='4,5,6',
                    help='anomaly_binary 模式: 异常类ID, 逗号分隔, 如 "4,5,6"')

# 数据划分
parser.add_argument('--resplit_stratified', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='加载后按类别分层重新划分 train/val/test (seed 控制); '
                         'anomaly_binary 在标签映射之后执行 (--resplit_after_label_mapping 为兼容别名)。'
                         '默认 None = 按 task_mode 自动: anomaly_binary=True, multiclass=False。'
                         'multiclass 默认关闭的原因(2026-09-21 实测): 重切分会丢弃缓存 data*.pt 中'
                         '作者发布的原始切分, 使训练/测试节点与原版不同, Cora-10 FedAvg 被抬到 77.21'
                         '(原版/论文 73.6~74.2), 跨种子 std 从 ±0.3 放大到 ±1.5, 数字不可与论文对照')
parser.add_argument('--resplit_after_label_mapping', action=argparse.BooleanOptionalAction,
                    default=True, help=argparse.SUPPRESS)

# ---- 划分支持度模式 (机制实验) ----
parser.add_argument('--split_support_mode', type=str, default='natural',
                    choices=['natural', 'stratified_local', 'anomaly_holdout_boost',
                             'anomaly_enriched_partition'],
                    help='natural: 原划分; stratified_local: 客户端内按类分层+最小 support 约束; '
                         'anomaly_holdout_boost: 提高异常类 val 占比 (仅可靠性估计, 不从 test 移动); '
                         'anomaly_enriched_partition: 受控支持度划分 (机制实验, 移动完整社区, '
                         '标记为 controlled support partition)')
parser.add_argument('--min_train_support_per_class', type=int, default=0)
parser.add_argument('--min_val_support_per_class', type=int, default=0)
parser.add_argument('--anomaly_val_ratio', type=float, default=0.5,
                    help='anomaly_holdout_boost: val 中异常类目标占比')
parser.add_argument('--support_rebalance_seed', type=int, default=0)
parser.add_argument('--allow_support_infeasible', action='store_true', default=False,
                    help='最小 support 不可满足时继续并如实记录 (默认抛错)')
parser.add_argument('--anomaly_partition_target', type=int, default=0,
                    help='anomaly_enriched_partition: 每客户端最少异常节点数 (0=不启用)')

# 联邦模式
parser.add_argument('--federated_mode', type=str, default='fedavg',
                    choices=['local_only', 'fedavg'],
                    help='local_only: 每客户端只用自己的数据训练, 不聚合 (基线 B0); '
                         'fedavg: 标准联邦流程')

# 蒸馏权重来源 (A7 消融)
parser.add_argument('--distill_weighting', type=str, default='static_ckr',
                    choices=['none', 'equal', 'static_ckr', 'dynamic_ckr'],
                    help='none: 跳过生成器与蒸馏 (B1/B2/B3 基线); '
                         'equal: 无 CKR 等权蒸馏; '
                         'static_ckr: 静态拓扑 CKR 加权蒸馏 (B4 完整方法, 默认); '
                         'dynamic_ckr: 动态 CKR (EMA) 加权蒸馏 (实验性扩展, 非核心主线)')

# 加权CE
parser.add_argument('--use_weighted_ce', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='类别加权 CE。默认 None = 按 task_mode 自动: anomaly_binary=True, '
                         'multiclass=False。multiclass 默认关闭的原因: Cora-10 五种子配对实测 '
                         '该组件相对 FedAvg 为 -2.10 (逆频率加权在客户端内部极度非同分布时扭曲优化目标)')
parser.add_argument('--class_weight_method', type=str, default='inverse',
                    choices=['inverse', 'effective_num'])
parser.add_argument('--beta', type=float, default=0.999)

# 对比学习模式
parser.add_argument('--allow_anomaly_majority', action='store_true', default=False,
                    help='允许异常为多数类 (默认异常必须是少数类)')
parser.add_argument('--contrastive_mode', type=str, default='subgraph_cross_view',
                    choices=['subgraph_cross_view', 'none'],
                    help='subgraph_cross_view: 子图-子图跨视图对比(默认); '
                         'none: 不使用对比学习')
parser.add_argument('--edge_perturb_ratio', type=float, default=0.2,
                    help='边扰动总比例: 删除 ratio/2 的无向边 + 新增 ratio/2 的无向边 '
                         '(第二视图增强, 以无向边为单位)')
parser.add_argument('--rwr_restart_prob', type=float, default=0.5)
parser.add_argument('--rwr_subgraph_size', type=int, default=5)
parser.add_argument('--contrastive_batch_size', type=int, default=64)
parser.add_argument('--contrastive_temperature', type=float, default=0.2,
                    help='InfoNCE 温度 τ。默认 0.2 (原为 0.5)。'
                         'Cora-10 五种子实测: τ=0.2 -> 76.53±0.77 vs τ=0.5 -> 75.75±0.87 (+0.78), '
                         '且 τ 越小越好 (τ=1.0 -> 75.00), 5/5 种子一致')
parser.add_argument('--lambda_subgraph', type=float, default=0.1)
parser.add_argument('--contrastive_anchor_scope', type=str, default='all_nodes',
                    choices=['train_nodes', 'all_nodes'])

# ---- RWR 子图采样缓存 ----
parser.add_argument('--rwr_cache', type=str, default='enabled',
                    choices=['enabled', 'disabled'],
                    help='RWR 子图结构缓存 (只缓存采样结构, 不缓存 GNN 输出)')
parser.add_argument('--rwr_cache_max_entries', type=int, default=4096)
parser.add_argument('--rwr_cache_view1_persistent', type=str, default='enabled',
                    choices=['enabled', 'disabled'],
                    help='view_1 原始图采样结构跨轮缓存 (图结构固定)')
parser.add_argument('--rwr_seed', type=int, default=0,
                    help='RWR 采样的确定性随机种子 (缓存 key 的一部分)')
parser.add_argument('--rwr_cache_mode', type=str, default='epoch_reuse',
                    choices=['off', 'safe_exact', 'epoch_reuse'],
                    help='off: 关闭; safe_exact: 所有 key 分量一致才命中 (view_1 不跨轮); '
                         'epoch_reuse: view_1 跨轮持久 + view_2 轮内复用 (默认)')
parser.add_argument('--rwr_cache_scope', type=str, default='round',
                    choices=['round', 'run'],
                    help='view_2 缓存作用域: round=按轮失效; run=不按轮失效 '
                         '(正确性仍由 edge hash 保证, 每轮扰动不同时等价于 miss)')
parser.add_argument('--local_epochs', type=int, default=0,
                    help='本地训练轮数别名 (0=使用 --num_epochs)')
parser.add_argument('--anchor_sampling_mode', type=str, default='random_each_epoch',
                    choices=['random_each_epoch', 'fixed_per_round', 'fixed_global'],
                    help='锚点采样: 每 epoch 随机 / 每轮固定 (轮内跨 epoch 复用) / 全程固定')
parser.add_argument('--anchor_pool_size', type=int, default=0,
                    help='锚点池大小 (0=使用全部候选节点)')

# 扩散生成器
parser.add_argument('--diffusion_steps', type=int, default=20,
                    help='扩散总步数 T。默认 20 (原 10)。原因: T=10/β_end=0.02 时 '
                         'alpha_bar_T=0.904, x_T 仍残留 90%% 原始信号, 不满足专利 S3.1 '
                         '「经过 T 步后退化为纯噪声」; T=20/β_end=0.5 时 alpha_bar_T=0.002 ✓')
parser.add_argument('--diffusion_hidden', type=int, default=256)
parser.add_argument('--diffusion_beta_start', type=float, default=1e-4)
parser.add_argument('--diffusion_beta_end', type=float, default=0.5,
                    help='β 调度终点。默认 0.5 (原 0.02), 与 --diffusion_steps 20 配合使前向过程真正退化')
parser.add_argument('--generator_output_bound', type=str, default='tanh',
                    choices=['tanh', 'clamp', 'none'],
                    help='生成器输出激活。原版 FedTAD 用的是无界 Linear; 本仓库历史默认 tanh '
                         '会把伪特征尺度卡死在 [-1,1] (实测三数据集 σ 恒为 0.655, 与数据无关)')
parser.add_argument('--feature_stats_align', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='[实验性, 默认关] 特征统计特性对齐。客户端本地计算自己特征的 (μ,σ) 并上行'
                         '(2×F 个统计量, 非原始特征, 与专利权1「仅上传分值」同一条思路), '
                         '服务端按节点数加权聚合成全局统计量, 再把生成器输出重标定到该统计特性, '
                         '依据专利 S3.2「生成具备真实统计特性的伪节点特征矩阵」。'
                         '2026-09-21 实测: 能把伪特征 σ 精确对齐到真实值(0.98→0.0916, 真实 0.092), '
                         '但会使生成器 L_sem 与 L_dis 同时塌缩到 0、梯度消失, 准确率反而略降'
                         '(72.99→71.53)。根因: 真实尺度的伪特征让教师轻易分类且互相一致, 分歧信号归零。'
                         '故默认关闭, 保留供进一步研究')

# ---- 联邦扩散预训练 (专利 S3.1/S3.2 + 专利权1) ----
parser.add_argument('--federated_diffusion_pretrain', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='联邦扩散预训练 (默认**开**)。主训练前先做联邦去噪预训练: '
                         '各客户端在**本地**用自己的真实特征做前向加噪并训练噪声预测 '
                         '(专利 S3.1 前向过程 + S3.2 噪声预测损失), '
                         '只上传去噪网络参数, 服务端按节点数加权聚合, **原始特征不出域** (专利权1)。'
                         '⚠️ 依赖退化的噪声调度: 已把 --diffusion_steps 默认改为 20、'
                         '--diffusion_beta_end 默认改为 0.5 (alpha_bar_T=0.002 ≈ 纯噪声)。'
                         '关闭后行为与旧版一致(去噪网络不被训练)')
parser.add_argument('--diffusion_pretrain_rounds', type=int, default=10)

# ---- pretrained-manifold 约束 (radius + anchor) ----
parser.add_argument('--radius_constraint', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='[默认开] 硬流形约束: 对抗期 fake_x = r_c * z / ||z||, '
                         'r_c 来自预训练 radius bank(每类)。径向自由度消失 -> '
                         '幅度作弊被堵死, 切向(语义)梯度保留, 仍可搜 hard examples。')
parser.add_argument('--radius_bank_size', type=int, default=64,
                    help='radius bank 每类采样数 (只存标量, 显存可忽略)')
parser.add_argument('--lambda_diffusion_anchor', type=float, default=1e-2,
                    help='L2-SP 参数锚系数: 把生成器锚定在联邦预训练后的 θ_pre, '
                         '防止 prior 被对抗精调洗掉。0 = 关闭')
parser.add_argument('--use_posterior_variance', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='[默认开] DDPM reverse 用 posterior variance β̃_t 代替 √β_t '
                         '(当前 β 很大, √β_t 注入噪声过强)')
parser.add_argument('--diffusion_residual_skip', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='[诊断 A/B, 默认关] denoiser 直通残差: eps_pred = x_t + body(x_t,t,c)。'
                         '高噪声步 x_t≈ε, 恒等映射不再被迫穿过 hidden_dim 瓶颈 '
                         '(S3 已确认 256 瓶颈是晚 t MSE 0.64 的根因)。'
                         '等价于 --diffusion_skip_mode fixed')
parser.add_argument('--diffusion_skip_mode', type=str, default='none',
                    choices=['none', 'fixed', 'timestep_scalar'],
                    help='denoiser 直通残差模式: none=旧版; fixed=eps=x_t+body; '
                         'timestep_scalar=eps=c(t)·x_t+body (每 timestep 一个可学习 '
                         '正 scalar c(t)=exp(g_t), 初始化 1, 自动学 early-t 放大系数, '
                         '修 fixed identity skip 的结构性 bias)')
parser.add_argument('--pretrain_diagnostic_only', action='store_true',
                    help='[诊断] 联邦预训练结束后只做 pretrain-only 健康检查并退出, '
                         '不进对抗主循环 (分级验证 Stage 1)')
parser.add_argument('--diffusion_pretrain_resume', type=str, default='',
                    help='[诊断] 从保存的 pretrained_generator.pt 继续联邦预训练 '
                         '(配合 --pretrain_diagnostic_only 用)')
parser.add_argument('--diffusion_pretrain_resume_round', type=int, default=0,
                    help='[诊断] 续跑前已完成的预训练轮数 (仅用于纵向 checkpoint 轮次标注)')
parser.add_argument('--diffusion_pretrained_checkpoint', type=str, default='',
                    help='[Stage 2 诊断] 加载已完成 Stage 1 的 pretrained generator '
                         '权重初始化 Stage 2: 只加载权重(校验配置一致性), '
                         '不恢复预训练 Adam 状态, 重建对抗期 optimizer。'
                         '配合 --no-federated_diffusion_pretrain 使用')
parser.add_argument('--diffusion_freeze_skip_scale', action='store_true',
                    help='[Stage 2 诊断, 默认 False 保持旧实验兼容] 主循环前把 '
                         'timestep_scalar 的 c(t) 冻结在 pretrained 值 '
                         '(防止对抗 KL 把它变成新的幅度旋钮)')
parser.add_argument('--local_optimizer_lifecycle', type=str, default='persistent',
                    choices=['persistent', 'reset_each_round'],
                    help='本地 optimizer 生命周期 (Bug 5): persistent=跨轮保留 '
                         'Adam 动量并随 checkpoint 保存/恢复 (默认, 旧行为); '
                         'reset_each_round=每轮广播后重建本地 optimizer '
                         '(供 causal diagnostic)')
parser.add_argument('--allow_legacy_resume_without_local_optimizer_state',
                    action='store_true',
                    help='[默认 False] 允许加载缺少 local_optimizer_states 的 '
                         '旧 checkpoint (persistent 模式下轨迹与连续训练不一致, '
                         '仅供兼容旧产物)')
parser.add_argument('--formal_multiclass_protocol_guard', action='store_true',
                    help='[正式主实验] 启动时打印并校验正式 multiclass 协议'
                         '(weighted CE=ON/contrastive=ON/holdout=0/'
                         'accuracy 选轮/reset optimizer), 违反即停止')
parser.add_argument('--paired_rng_diag', action='store_true',
                    help='[诊断] 打印 P0-P4 RNG fingerprint + 分类器参数 hash + '
                         'client0 锚点/增强边 hash (paired-arm R0 一致性核验)')
parser.add_argument('--deterministic_torch', action='store_true',
                    help='[诊断] 开启 torch 确定性模式 (deterministic algorithms + '
                         'cudnn.deterministic + cudnn.benchmark=False), '
                         '用于定位 CUDA reduction 的跨进程数值不确定源')
parser.add_argument('--server_start_round', type=int, default=0,
                    help='[默认 0] Stage 2 服务器阶段 (生成器对抗更新 + 全局蒸馏) '
                         '的启动轮次。round < server_start_round 时只执行客户端训练 '
                         '+ FedAvg (SKIPPED 标记); >= 时执行完整现有 Stage 2。'
                         'Stage 2 初始化 (pretrained 加载/bank/冻结/θ_pre) 不受影响, '
                         '仍在主训练前完成。')
parser.add_argument('--tuning_mode', action='store_true',
                    help='[正式调参] test isolation: 调参模式下完全不计算/打印 test, '
                         '只按 validation 选轮; objective/pruner 无法读取 test')
parser.add_argument('--final_test_only', action='store_true',
                    help='[正式决赛] 每轮只评估 validation，训练结束加载 val-best '
                         'checkpoint 后只计算一次 test；与 --tuning_mode 互斥。')
parser.add_argument('--formal_campaign_guard', action='store_true',
                    help='[正式战役] 启动时校验完整正式协议 (见 validate_formal_campaign), '
                         '违反即 ValueError, 不自动修参数')
parser.add_argument('--diffusion_pretrain_epochs', type=int, default=30)
parser.add_argument('--diffusion_pretrain_batch', type=int, default=256)
parser.add_argument('--diffusion_pretrain_lr', type=float, default=1e-3)

# ---- 生成器初始化与反向传播 ----
parser.add_argument('--generator_init', type=str, default='scratch',
                    choices=['scratch', 'teacher_guided_scratch', 'proxy_pretrained'],
                    help='scratch: 从高斯噪声随机初始化 (默认, 等价 public_pretraining=false); '
                         'teacher_guided_scratch: 教师引导+随机初始化 (兼容别名, 与 scratch 相同); '
                         'proxy_pretrained: 先加载公开代理数据 DDPM 预训练权重 (实验性扩展)')
parser.add_argument('--proxy_checkpoint', type=str, default='',
                    help='proxy_pretrained 模式使用的代理 checkpoint 路径')
parser.add_argument('--generator_backprop_mode', type=str, default='checkpointed',
                    choices=['full', 'checkpointed', 'truncated'],
                    help='full: 完整计算图; checkpointed: 激活重计算(默认, 梯度完整); '
                         'truncated: 每 N 步 detach 的截断近似')
parser.add_argument('--generator_backward_mode', type=str, default='',
                    choices=['', 'full', 'checkpointed'],
                    help='--generator_backprop_mode 的兼容别名')
parser.add_argument('--generator_truncate_interval', type=int, default=2,
                    help='truncated 模式每隔 N 步对采样状态 detach')
parser.add_argument('--generator_sampling_steps', type=int, default=0,
                    help='生成器采样步数 (0 = 使用 --diffusion_steps)')
parser.add_argument('--checkpoint_segments', type=int, default=1,
                    help='checkpointed 模式按 N 步分组做激活重计算 (1=逐步)')

# ---- CKR 诊断模式 (阶段三) ----
parser.add_argument('--ckr_diagnostic_mode', type=str, default='none',
                    choices=['none', 'shuffled', 'inverse', 'oracle_validation'],
                    help='none: 正常权重; '
                         'shuffled: 每类打乱客户端权重 (保留边际分布, 非均匀权重对照); '
                         'inverse: 权重反转 (负对照, 仅诊断, 不得作为主方法); '
                         'oracle_validation: 用客户端 validation 类别性能作权重 '
                         '(诊断上界, 非隐私友好, 严禁使用 test 标签)')
# ---- 蒸馏因果链 (阶段三) ----
parser.add_argument('--generator_update_mode', type=str, default='train',
                    choices=['train', 'frozen'],
                    help='frozen: 生成器不更新, 只用于采样 (D5: 检验生成器训练贡献)')
parser.add_argument('--fake_graph_topology', type=str, default='knn',
                    choices=['knn', 'isolated'],
                    help='isolated: 伪图只含自环 (D8: 检验 KNN 代理拓扑贡献)')
parser.add_argument('--knn_k', type=int, default=5,
                    help='KNN 伪图近邻数 k (默认 5)')

# ---- canonical 参数别名 (平台 canonical 配置名 == CLI 名, 见 docs/TRAINING_PARAMETERS.md) ----
parser.add_argument('--federated_rounds', type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument('--learning_rate', type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument('--fake_node_count', type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument('--distillation_steps', type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument('--task_id', type=str, default='', help=argparse.SUPPRESS)

# ---- 平台结构化事件 / 参数热更新 (round 边界生效) ----
parser.add_argument('--emit_events', action='store_true', default=False,
                    help='向 stdout 输出 [EVENT] 结构化 JSON 事件 (平台实时可视化)')
parser.add_argument('--events_jsonl', type=str, default='',
                    help='结构化事件 JSONL 落盘路径 (与 stdout 事件一致)')
parser.add_argument('--param_update_dir', type=str, default='',
                    help='参数热更新目录: 每轮开始读取 pending_updates.json 并应用'
                         '(仅白名单参数, 轮次边界生效); 应用记录写入 applied_updates.jsonl')
parser.add_argument('--teacher_mixture_jsonl', type=str, default='',
                    help='每轮教师混合诊断 JSONL 输出路径')
parser.add_argument('--distillation_diagnostics_jsonl', type=str, default='',
                    help='每轮蒸馏诊断 JSONL 输出路径')

# ---- 公平性机制 (阶段三, 默认 none 保持原行为) ----
parser.add_argument('--fairness_mode', type=str, default='none',
                    choices=['none', 'qffl_aggregation',
                             'validation_deficit_distillation', 'combined'],
                    help='none: 保持当前行为; '
                         'qffl_aggregation: q-FedAvg 风格聚合权重 (基于本地 validation loss); '
                         'validation_deficit_distillation: 按 validation metric 缺口调整蒸馏权重; '
                         'combined: 两者同时启用 (仅实验配置)')
parser.add_argument('--fairness_q', type=float, default=2.0)
parser.add_argument('--fairness_alpha', type=float, default=0.5)
parser.add_argument('--fairness_weight_min', type=float, default=0.1)
parser.add_argument('--fairness_weight_max', type=float, default=3.0)
parser.add_argument('--fairness_metric', type=str, default='',
                    help='公平性指标: 默认 anomaly=val roc_auc (单类回退 accuracy), '
                         'multiclass=accuracy; 只使用 validation, 严禁 test')
parser.add_argument('--fairness_warmup_rounds', type=int, default=3)

# 生成器训练
parser.add_argument('--generator_steps', type=int, default=1)
parser.add_argument('--generator_lr', type=float, default=1e-3)
parser.add_argument('--lambda_sem', type=float, default=1.0)
parser.add_argument('--lambda_disagreement', type=float, default=0.1)
parser.add_argument('--lambda_diversity', type=float, default=0.1)
parser.add_argument('--lambda_feature_norm', type=float, default=0.0,
                    help='伪特征 L2 范数惩罚权重。默认 0.0 (原为 1e-3)。'
                         '原值下该损失项只占 L_G 的约 0.04%%(L_norm≈0.43 × 1e-3 vs L_G≈0.43), '
                         '实测形同虚设且对伪特征尺度(σ=0.71 vs 真实 0.11)毫无约束, 故置零')
parser.add_argument('--fake_nodes', type=int, default=100)
parser.add_argument('--fake_class_strategy', type=str, default='balanced',
                    choices=['balanced', 'prior', 'reliability'])
parser.add_argument('--generator_warmup_rounds', type=int, default=3)

# 全局蒸馏
parser.add_argument('--distill_steps', type=int, default=5,
                    help='每轮蒸馏内迭代次数。原版 FedTAD 为 glb_epochs(5)×it_d(5)=25, '
                         '故与论文对照时应设 25')
parser.add_argument('--distill_lr', type=float, default=1e-3)
parser.add_argument('--distill_temperature', type=float, default=1.0)
parser.add_argument('--distill_loss_type', type=str, default='kl',
                    choices=['kl', 'l1'],
                    help='蒸馏/分歧损失形式。默认 kl = 本方法(**专利权6 / 专利书 S4.2**)的设计: '
                         '「采用 Kullback-Leibler 散度衡量全局模型与各客户端本地模型在伪图节点上的'
                         '预测分布差异, 并以类别知识可靠性为动态权重加权」。'
                         'l1 = 原版 FedTAD 的做法(mean|global_pred − local_pred.detach()| 作用在原始输出上, '
                         'references/FedTAD/train_fedtad.py:342-345), 仅供消融对照, 不是本方法的设计。'
                         '注意: 二者梯度性质不同, 不是等价变形, 不要为了"对齐原版"而误改默认值')

# ---- CKR 模式 ----
parser.add_argument('--ckr_mode', type=str, default='static_topology',
                    choices=['static_topology', 'dynamic_only', 'performance_only',
                             'hybrid_dynamic'],
                    help='static_topology: 静态拓扑 CKR (核心主线, 默认); '
                         'dynamic_only / performance_only: 仅动态指标 (实验性扩展); '
                         'hybrid_dynamic: 静态先验+动态指标+EMA (实验性扩展)')
parser.add_argument('--dynamic_ckr_eval_source', type=str, default='reliability_holdout',
                    choices=['reliability_holdout', 'validation'],
                    help='动态 CKR 指标评估集: 默认 reliability_holdout '
                         '(从 train 划分, 不污染 val/test); 显式选择 validation 时 '
                         'val 不再是干净的模型选择集')
parser.add_argument('--reliability_holdout_ratio', type=float, default=0.2)
parser.add_argument('--reliability_split_seed', type=int, default=42)
parser.add_argument('--reliability_min_support', type=int, default=3)
parser.add_argument('--dynamic_ckr_metric', type=str, default='f1',
                    choices=['f1', 'recall', 'confidence'])
parser.add_argument('--dynamic_ckr_alpha', type=float, default=0.5)
parser.add_argument('--dynamic_ckr_ema_decay', type=float, default=0.8)
parser.add_argument('--static_ckr_scaling', type=str, default='max',
                    choices=['max', 'minmax', 'rank'])

# ---- Checkpoint ----
parser.add_argument('--checkpoint_dir', type=str, default='',
                    help='最佳模型 checkpoint 保存目录 (非空即启用)')
parser.add_argument('--resume_checkpoint', type=str, default='',
                    help='从 checkpoint 恢复训练 (模型/优化器/CKR EMA/RNG 连续)')
parser.add_argument('--save_last_checkpoint', action='store_true', default=False,
                    help='每轮保存 last.pt')
parser.add_argument('--selection_metric', type=str, default='',
                    choices=['', 'macro_f1', 'pooled_pr_auc', 'accuracy',
                             'weighted_client_auc'],
                    help='最佳模型选择指标; 默认 multiclass=macro_f1, '
                         'anomaly_binary=pooled_pr_auc')

# 日志
parser.add_argument('--log_dir', type=str, default='./logs',
                    help='动态 CKR JSONL 历史日志目录')
parser.add_argument('--metrics_jsonl', type=str, default='',
                    help='每轮训练行为诊断 JSONL 输出路径 (空=不写)')
parser.add_argument('--final_metrics_json', type=str, default='',
                    help='最终 test 指标 JSON 输出路径 (空=不写)')
parser.add_argument('--data_identity_json', type=str, default='',
                    help='data_identity.json 输出路径')
parser.add_argument('--initialization_json', type=str, default='',
                    help='initialization.json (初始化身份哈希) 输出路径')
parser.add_argument('--split_report_json', type=str, default='',
                    help='划分报告 JSON 输出路径 (空=不写)')
parser.add_argument('--resource_usage_json', type=str, default='',
                    help='资源开销 JSON 输出路径 (空=不写)')
parser.add_argument('--evaluation_report_json', type=str, default='',
                    help='统一评估报告 JSON (evaluation_scope/comparability) 输出路径')
parser.add_argument('--ckr_availability_json', type=str, default='',
                    help='CKR 每轮可观测性聚合 JSON 输出路径')
parser.add_argument('--fairness_metrics_json', type=str, default='',
                    help='公平性/低资源分析 JSON 输出路径')
parser.add_argument('--cache_metrics_json', type=str, default='',
                    help='RWR 缓存指标 JSON 输出路径')

# 双重终止
parser.add_argument('--f1_threshold', type=float, default=0.1,
                    help='第一终止(全局): 主指标连续N轮提升小于该值')
parser.add_argument('--auc_threshold', type=float, default=1.0,
                    help='第二终止(低资源): 客户端AUC连续N轮增幅小于该值')

args = (parser.parse_args() if __name__ == '__main__'
        else parser.parse_args([]))

# ---- task_mode 自适应的默认值 (2026-09-21) ----
# 这两个开关在 multiclass 口径下经 5 种子配对实测为负贡献/破坏可对照性,
# 但 anomaly_binary 主线仍需要它们的历史默认值, 故用 None 哨兵按 task_mode 分派,
# 保证异常检测实验的行为逐位不变。
if args.resplit_stratified is None:
    args.resplit_stratified = (args.task_mode == 'anomaly_binary')
if args.use_weighted_ce is None:
    args.use_weighted_ce = (args.task_mode == 'anomaly_binary')

# 兼容别名
args.resplit_after_label_mapping = args.resplit_stratified
if args.generator_backward_mode:
    args.generator_backprop_mode = args.generator_backward_mode
if args.local_epochs:
    args.num_epochs = args.local_epochs
# ---- canonical 别名解析: 平台 canonical 配置名 -> CLI 属性 ----
if args.federated_rounds is not None:
    args.num_rounds = args.federated_rounds
if args.learning_rate is not None:
    args.lr = args.learning_rate
if args.fake_node_count is not None:
    args.fake_nodes = args.fake_node_count
if args.distillation_steps is not None:
    args.distill_steps = args.distillation_steps
# ---- 四类随机种子拆分: --seed 兼容入口展开 ----
for _seed_attr in ('partition_seed', 'allocation_seed', 'split_seed',
                   'model_seed', 'bootstrap_seed'):
    if getattr(args, _seed_attr) < 0:
        setattr(args, _seed_attr, args.seed)
# FGLDataset 分区支持度模式 (enriched 使用独立缓存目录)
args.partition_support_mode = ('anomaly_enriched'
                               if args.split_support_mode == 'anomaly_enriched_partition'
                               else 'natural')
args.partition_anomaly_classes = ([int(x) for x in args.anomaly_classes.split(',') if x.strip()]
                                  if args.anomaly_classes else None)
args.partition_anomaly_target = args.anomaly_partition_target
args.partition_rebalance_seed = args.allocation_seed
# partition_mode: natural(缓存) / build(frozen artifact 构建) / frozen(加载)
if args.build_split_artifact:
    args.partition_mode = 'build'
elif args.frozen_split:
    args.partition_mode = 'frozen'
else:
    args.partition_mode = 'natural'

# =====================================================================
#  工具函数
# =====================================================================

def _write_dataset_manifest(dataset, args):
    """写 dataset_manifest.json (离线导入校验)。"""
    import hashlib as _hashlib
    G = dataset.global_data
    manifest = {
        'dataset': args.dataset,
        'source': args.dataset_source,
        'offline': bool(args.offline),
        'root': args.root,
        'num_nodes': int(G.x.shape[0]),
        'num_edges': int(G.edge_index.shape[1]),
        'feature_dim': int(G.x.shape[1]),
        'original_num_classes': int(dataset.num_classes),
        'processed_num_classes': int(getattr(dataset, 'out_dim', dataset.num_classes)),
        'task_mode': args.task_mode,
        'label_mapping': (f"normal={args.normal_classes},anomaly={args.anomaly_classes}"
                          if args.task_mode == 'anomaly_binary' else 'none'),
        'preprocessing_version': 'fedtad-phase3-v1',
    }
    # 源文件大小与校验和
    import glob as _glob
    files = {}
    for f in _glob.glob(os.path.join(args.root, '**', '*'), recursive=True):
        if os.path.isfile(f):
            try:
                files[f] = os.path.getsize(f)
            except OSError:
                pass
    manifest['local_files'] = files
    if args.dataset_checksum:
        manifest['checksum_sha1'] = args.dataset_checksum
        manifest['checksum_verified'] = False  # 由外部校验器设置
    if args.dataset_manifest:
        os.makedirs(os.path.dirname(args.dataset_manifest) or '.', exist_ok=True)
        with open(args.dataset_manifest, 'w') as f:
            json.dump(manifest, f, indent=1, default=str)
        print(f"[Dataset Manifest] 写入 {args.dataset_manifest}")


def validate_manifest(manifest_path, expected_dataset):
    """
    校验 dataset manifest: 数据集名称必须匹配, 否则拒绝 (防 Cora 冒充 CiteSeer)。
    Returns: (ok, reason)
    """
    try:
        with open(manifest_path) as f:
            m = json.load(f)
    except Exception as e:
        return False, f"manifest 读取失败: {e}"
    if m.get('dataset') != expected_dataset:
        return False, (f"manifest dataset={m.get('dataset')} != 期望 {expected_dataset} "
                       f"(禁止用其他数据集冒充)")
    if not m.get('num_nodes') or not m.get('feature_dim'):
        return False, "manifest 缺少 num_nodes/feature_dim"
    return True, 'ok'


def config_hash(args):
    """对全部超参数生成稳定哈希, 写入结果文件。"""
    import hashlib, json as _json
    payload = _json.dumps(vars(args), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def run_integrity_assertions(subgraphs, num_classes, args):
    """
    正式实验前正确性检查 (泄漏防护):
      1. fit/reliability/val/test 两两不重叠
      2. fit+reliability == train, reliability ⊆ train
      3. 训练只使用 fit (结构上由代码路径保证, 此处断言划分)
      4. CKR 动态指标只使用 reliability (评估集在代码路径中固定为 reliability_idx)
    """
    for ci, sg in enumerate(subgraphs):
        masks = {name: sg[f'{name}_idx']
                 for name in ('fit', 'reliability', 'val', 'test')}
        names = list(masks)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = int((masks[a] & masks[b]).sum())
                assert overlap == 0, \
                    f"integrity: client {ci} {a}_idx 与 {b}_idx 重叠 {overlap} 个节点!"
        union = int((masks['fit'] | masks['reliability']).sum())
        train_n = int(sg.train_idx.sum())
        assert union == train_n, \
            f"integrity: client {ci} fit+reliability({union}) != train({train_n})"
        assert int((masks['reliability'] & ~sg.train_idx).sum()) == 0, \
            f"integrity: client {ci} reliability 超出 train!"
        if args.task_mode == 'anomaly_binary' and args.resplit_stratified:
            assert hasattr(sg, 'original_y'), \
                f"integrity: client {ci} 缺少 original_y, 标签映射未执行"
    print("  [Integrity] fit/reliability/val/test 两两不重叠, "
          "fit+reliability==train, reliability⊆train ✓")


def _write_teacher_mixture(path, round_id, weights, tracker, num_classes):
    """每轮教师混合诊断: 权重矩阵、熵、top 客户端、与静态/上轮差异。"""
    import math as _m
    K = weights.shape[0]
    w = weights.detach().cpu()
    prev = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                lines = f.readlines()
            if lines:
                prev = json.loads(lines[-1])
        except Exception:
            prev = {}
    static_norm = normalize_ckr_safe(tracker.static_ckr_scaled).detach().cpu()
    record = {'round': round_id}
    for c in range(num_classes):
        col = w[:, c]
        eps = 1e-12
        entropy = -float((col * torch.log(col + eps)).sum())
        top_k = int(col.argmax())
        diff_static = float((col - static_norm[:, c]).abs().mean())
        diff_prev = None
        if prev and f'class_{c}_weights' in prev:
            p = torch.tensor(prev[f'class_{c}_weights'])
            diff_prev = float((col - p).abs().mean())
        record[f'class_{c}'] = {
            'entropy': entropy,
            'top_client': top_k,
            'diff_from_static': diff_static,
            'diff_from_previous': diff_prev,
        }
        record[f'class_{c}_weights'] = [round(float(v), 6) for v in col.tolist()]
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def _client_val_signals(local_models, subgraphs, task_mode, device):
    """每客户端 validation 标量信号 (loss + metric), 只用于公平性/qffl。"""
    import torch.nn.functional as F
    signals = {}
    for ci, sg in enumerate(subgraphs):
        model = local_models[ci]
        model.eval()
        with torch.no_grad():
            logits = model.forward(sg)
            val_logits = logits[sg.val_idx]
            val_y = sg.y[sg.val_idx]
            loss = float(F.cross_entropy(val_logits, val_y).item())
            em = evaluate_client(model, sg, task_mode, eval_mask=sg.val_idx)
            if task_mode == 'anomaly_binary':
                metric = em.get('roc_auc', float('nan'))
                if math.isnan(metric):
                    metric = em.get('accuracy', float('nan'))
            else:
                metric = em.get('accuracy', float('nan'))
        signals[ci] = {'val_loss': loss, 'val_metric': metric}
    return signals


def _sanitize_json(obj):
    """将 numpy/torch 标量转为 python 原生类型 (结果文件可 JSON 序列化)。"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return _sanitize_json(obj.cpu().tolist())
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    return obj


# =====================================================================
#  平台集成: 结构化事件 + 参数热更新 (轮次边界生效)
# =====================================================================

def _emit_event(args, event_type, round_id=None, stage=None, **fields):
    """发出结构化事件: stdout 输出 [EVENT] {json} (平台解析), 并可选落盘 JSONL。

    事件只用于平台实时可视化与审计, 不参与任何训练计算。
    """
    import time as _time
    ev = {'event': event_type,
          'task_id': args.task_id if hasattr(args, 'task_id') else '',
          'timestamp': round(_time.time(), 3)}
    if round_id is not None:
        ev['round'] = round_id
    if stage is not None:
        ev['stage'] = stage
    ev.update(fields)
    line = '[EVENT] ' + json.dumps(ev, ensure_ascii=False)
    if args.emit_events:
        print(line, flush=True)
    if args.events_jsonl:
        os.makedirs(os.path.dirname(args.events_jsonl) or '.', exist_ok=True)
        with open(args.events_jsonl, 'a') as f:
            f.write(json.dumps(ev, ensure_ascii=False) + '\n')


# 热更新白名单: canonical 参数名 -> (args 属性名, 类型, [min, max])
# 这些参数在轮次边界应用是安全的: 均在每轮循环内读取, 不改变模型结构/数据划分。
HOT_UPDATABLE_PARAMS = {
    'learning_rate': ('lr', float, [1e-6, 10.0]),
    'lr': ('lr', float, [1e-6, 10.0]),
    'generator_lr': ('generator_lr', float, [1e-6, 10.0]),
    'distill_lr': ('distill_lr', float, [1e-6, 10.0]),
    'lambda_subgraph': ('lambda_subgraph', float, [0.0, 100.0]),
    'lambda_sem': ('lambda_sem', float, [0.0, 100.0]),
    'lambda_diversity': ('lambda_diversity', float, [0.0, 100.0]),
    'lambda_disagreement': ('lambda_disagreement', float, [0.0, 100.0]),
    'lambda_feature_norm': ('lambda_feature_norm', float, [0.0, 100.0]),
    'contrastive_temperature': ('contrastive_temperature', float, [1e-3, 10.0]),
    'generator_steps': ('generator_steps', int, [1, 100]),
    'distillation_steps': ('distill_steps', int, [1, 1000]),
    'distill_steps': ('distill_steps', int, [1, 1000]),
    'edge_perturb_ratio': ('edge_perturb_ratio', float, [0.0, 1.0]),
    'knn_k': ('knn_k', int, [1, 1000]),
}


def _apply_pending_param_updates(args, round_id,
                                 local_optimizers, gen_optimizer, global_optimizer):
    """每轮开始: 读取 pending_updates.json, 在轮次边界应用白名单参数。

    返回已应用更新列表; 每个更新同时写入 applied_updates.jsonl 并发事件。
    非白名单参数直接忽略 (平台侧已在 API 层拒绝)。
    """
    if not args.param_update_dir:
        return []
    pending_path = os.path.join(args.param_update_dir, 'pending_updates.json')
    applied_path = os.path.join(args.param_update_dir, 'applied_updates.jsonl')
    if not os.path.exists(pending_path):
        return []
    try:
        with open(pending_path) as f:
            pending = json.load(f)
    except Exception as e:
        print(f"  [HotUpdate] 解析失败: {e}", flush=True)
        return []
    if not isinstance(pending, list) or not pending:
        return []

    applied = []
    for upd in pending:
        if not isinstance(upd, dict):
            continue
        param = upd.get('parameter')
        value = upd.get('value')
        if param not in HOT_UPDATABLE_PARAMS:
            print(f"  [HotUpdate] 忽略非白名单参数: {param}", flush=True)
            continue
        attr, vtype, [vmin, vmax] = HOT_UPDATABLE_PARAMS[param]
        try:
            new_value = vtype(value)
        except (TypeError, ValueError):
            print(f"  [HotUpdate] 参数 {param} 类型非法: {value!r}", flush=True)
            continue
        new_value = max(vmin, min(vmax, new_value))
        old_value = getattr(args, attr)
        if old_value == new_value:
            continue
        setattr(args, attr, new_value)
        # 优化器学习率真实更新 (下一轮 optimizer step 生效)
        if param == 'learning_rate':
            for opt in local_optimizers:
                for pg in opt.param_groups:
                    pg['lr'] = new_value
        elif param == 'generator_lr':
            for pg in gen_optimizer.param_groups:
                pg['lr'] = new_value
        elif param == 'distill_lr':
            for pg in global_optimizer.param_groups:
                pg['lr'] = new_value
        record = {
            'parameter': param, 'old_value': old_value, 'new_value': new_value,
            'effective_round': round_id, 'request_id': upd.get('request_id', ''),
            'timestamp': round(time.time(), 3),
        }
        applied.append(record)
        _emit_event(args, 'parameter_update_applied',
                    round_id=round_id, stage='round_started',
                    **{k: record[k] for k in
                       ('parameter', 'old_value', 'new_value', 'effective_round')})
        print(f"  [HotUpdate] {param}: {old_value} -> {new_value} "
              f"(round {round_id} 生效)", flush=True)

    if applied:
        os.makedirs(args.param_update_dir, exist_ok=True)
        with open(applied_path, 'a') as f:
            for rec in applied:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # 清空已处理请求
        with open(pending_path, 'w') as f:
            json.dump([], f)
    return applied


def _count_connected_components(edge_index, num_nodes):
    """伪图连通分量数量 (小图并查集, 仅用于事件报告)。"""
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ei = edge_index.cpu()
    for i in range(ei.shape[1]):
        a, b = int(ei[0, i]), int(ei[1, i])
        if a >= num_nodes or b >= num_nodes:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = set()
    for x in range(num_nodes):
        roots.add(find(x))
    return len(roots)


def collect_final_metrics(model, subgraphs, task_mode, split='test',
                          evaluation_scope='global_model'):
    """
    统一评估: pooled / client_macro / client_weighted + 评估口径字段。
    test 指标不进入 checkpoint 保存条件; 本函数只在 final test 时调用。

    model: 单个全局模型 (evaluation_scope='global_model')
          或每客户端本地模型列表 (evaluation_scope='local_ensemble', 即 B0)。
    """
    import sklearn.metrics as skm
    is_ensemble = evaluation_scope == 'local_ensemble'
    models = model if is_ensemble else [model] * len(subgraphs)

    # per-client 指标 + 测试样本数
    per_client = {}
    test_counts = {}
    for ci, sg in enumerate(subgraphs):
        mask = sg.val_idx if split == 'val' else sg.test_idx
        em = evaluate_client(models[ci], sg, task_mode, eval_mask=mask)
        per_client[ci] = _sanitize_json(em)
        test_counts[ci] = int(mask.sum())

    num_test_samples = sum(test_counts.values())

    if task_mode == 'multiclass':
        n_c = len(per_client[0]['per_class_f1'])
        conf_sum = [[0] * n_c for _ in range(n_c)]
        total_correct = 0
        total_n = 0
        for ci, em in per_client.items():
            conf = em['confusion_matrix']
            for t in range(n_c):
                for p in range(n_c):
                    conf_sum[t][p] += int(conf[t][p])
            total_correct += sum(conf[t][t] for t in range(n_c))
            total_n += sum(sum(row) for row in conf)
        pooled_acc = total_correct / total_n * 100.0 if total_n > 0 else float('nan')
        pooled_f1 = []
        for c in range(n_c):
            tp = conf_sum[c][c]
            fp = sum(conf_sum[t][c] for t in range(n_c)) - tp
            fn = sum(conf_sum[c][p] for p in range(n_c)) - tp
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            pooled_f1.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
        pooled_macro_f1 = sum(pooled_f1) / n_c if n_c > 0 else float('nan')
        client_macro_f1 = sum(em['macro_f1'] for em in per_client.values()) / len(per_client)
        client_macro_acc = sum(em['accuracy'] for em in per_client.values()) / len(per_client)
        wsum = sum(test_counts.values())
        client_weighted_f1 = (sum(em['macro_f1'] * test_counts[ci]
                                  for ci, em in per_client.items()) / wsum
                              if wsum > 0 else float('nan'))
        client_weighted_acc = (sum(em['accuracy'] * test_counts[ci]
                                   for ci, em in per_client.items()) / wsum
                               if wsum > 0 else float('nan'))
        # ---- FedTAD 原版口径: 按**客户端总节点数**加权 (非测试集大小) ----
        # 原版 references/FedTAD/train_fedtad.py:227
        #   global_acc_test += x.shape[0]/global_data.x.shape[0] * acc_test
        node_counts = {ci: int(sg.x.shape[0]) for ci, sg in enumerate(subgraphs)}
        ntot = sum(node_counts.values())
        fedtad_acc = (sum(em['accuracy'] * node_counts[ci]
                          for ci, em in per_client.items()) / ntot
                      if ntot > 0 else float('nan'))
        fedtad_f1 = (sum(em['macro_f1'] * node_counts[ci]
                         for ci, em in per_client.items()) / ntot
                     if ntot > 0 else float('nan'))
        return {
            'pooled': {'accuracy': pooled_acc, 'macro_f1': pooled_macro_f1,
                       'per_class_f1': pooled_f1,
                       'confusion_matrix': conf_sum},
            'client_macro': {'accuracy': client_macro_acc, 'macro_f1': client_macro_f1},
            'client_weighted': {'accuracy': client_weighted_acc,
                                'macro_f1': client_weighted_f1},
            'fedtad_official': {'accuracy': fedtad_acc, 'macro_f1': fedtad_f1},
            'per_client': per_client,
            'valid_clients': {'all': len(per_client)},
            'num_test_samples': num_test_samples,
            'evaluation_scope': evaluation_scope,
            'aggregation_levels': ['pooled', 'client_macro', 'client_weighted',
                                   'fedtad_official'],
            'metric_comparability_group': f'{task_mode}|{evaluation_scope}|{split}',
        }

    # ---- anomaly_binary ----
    all_labels, all_probs = [], []
    client_aucs, client_pr_aucs = [], []
    valid_counts = []
    for ci, sg in enumerate(subgraphs):
        mask = sg.val_idx if split == 'val' else sg.test_idx
        models[ci].eval()
        with torch.no_grad():
            logits = models[ci].forward(sg)
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_np = sg.y.cpu().numpy()
        m = mask.cpu().numpy()
        all_labels.extend(y_np[m].tolist())
        all_probs.extend(prob[m].tolist())
        uniq = set(int(x) for x in y_np[m])
        if m.sum() > 0 and len(uniq) >= 2:
            client_aucs.append(skm.roc_auc_score(y_np[m], prob[m]) * 100.0)
            client_pr_aucs.append(skm.average_precision_score(y_np[m], prob[m]) * 100.0)
            valid_counts.append(int(m.sum()))
        else:
            client_aucs.append(float('nan'))
            client_pr_aucs.append(float('nan'))

    pooled_auc = pooled_pr_auc = float('nan')
    if len(set(all_labels)) >= 2:
        try:
            pooled_auc = skm.roc_auc_score(all_labels, all_probs) * 100.0
            pooled_pr_auc = skm.average_precision_score(all_labels, all_probs) * 100.0
        except Exception:
            pass
    valid_auc = [v for v in client_aucs if not math.isnan(v)]
    valid_pr = [v for v in client_pr_aucs if not math.isnan(v)]
    wsum = sum(valid_counts)
    weighted_auc = (sum(a * c for a, c in zip(valid_auc, valid_counts)) / wsum
                    if wsum > 0 else float('nan'))
    weighted_pr = (sum(p * c for p, c in zip(valid_pr, valid_counts)) / wsum
                   if wsum > 0 else float('nan'))
    return {
        'pooled': {'roc_auc': pooled_auc, 'pr_auc': pooled_pr_auc},
        'client_macro': {'roc_auc': sum(valid_auc) / len(valid_auc) if valid_auc else float('nan'),
                         'pr_auc': sum(valid_pr) / len(valid_pr) if valid_pr else float('nan')},
        'client_weighted': {'roc_auc': weighted_auc, 'pr_auc': weighted_pr},
        'per_client': per_client,
        'valid_clients': {'roc_auc': len(valid_auc), 'pr_auc': len(valid_pr),
                          'total': len(per_client)},
        'num_test_samples': num_test_samples,
        'evaluation_scope': evaluation_scope,
        'aggregation_levels': ['pooled', 'client_macro', 'client_weighted'],
        'metric_comparability_group': f'{task_mode}|{evaluation_scope}|{split}',
    }


def _build_fairness_metrics(final_test, subgraphs, num_classes, task_mode, tracker):
    """
    公平性/低资源分析: 最差/底部 10%/20% 客户端、gap、IQR、零异常计数、资源分组。
    仅描述客户端间性能分布, 不称为"公平性证明"。
    """
    import statistics as st
    per_client = final_test.get('per_client', {})
    if not per_client:
        return {}
    if task_mode == 'multiclass':
        vals = {ci: em['macro_f1'] for ci, em in per_client.items()}
    else:
        vals = {ci: em['pr_auc'] for ci, em in per_client.items()}
    valid = {ci: v for ci, v in vals.items() if not math.isnan(v)}
    if not valid:
        return {}
    vs = sorted(valid.values())
    n = len(vs)
    worst = vs[0]
    best = vs[-1]
    bottom10 = sum(vs[:max(1, n // 10)]) / max(1, n // 10)
    bottom20 = sum(vs[:max(1, n // 5)]) / max(1, n // 5)
    mean = sum(vs) / n
    std = st.stdev(vs) if n > 1 else 0.0
    qs = [vs[int(n * 0.25)], vs[int(n * 0.75)]]
    iqr = qs[1] - qs[0]

    zero_anom_train = zero_anom_test = 0
    for sg in subgraphs:
        if (sg.y[sg.fit_idx] == 1).sum().item() == 0:
            zero_anom_train += 1
        if (sg.y[sg.test_idx] == 1).sum().item() == 0:
            zero_anom_test += 1

    # 资源分组 (按训练节点数 tercile)
    train_sizes = {ci: int(subgraphs[ci].fit_idx.sum()) for ci in range(len(subgraphs))}
    order = sorted(train_sizes, key=train_sizes.get)
    third = max(1, len(order) // 3)
    groups = {'low_resource': order[:third], 'mid_resource': order[third:2 * third],
              'high_resource': order[2 * third:]}
    group_means = {g: (sum(valid[ci] for ci in cis if ci in valid) /
                       max(1, sum(1 for ci in cis if ci in valid)))
                   for g, cis in groups.items()}

    # 按异常训练样本数分组
    anom_train = {ci: int((subgraphs[ci].y[subgraphs[ci].fit_idx] == 1).sum())
                  for ci in range(len(subgraphs))}
    anom_groups = {'anomaly_low': [], 'anomaly_mid': [], 'anomaly_high': []}
    ao = sorted(anom_train, key=anom_train.get)
    for ci in ao[:third]:
        anom_groups['anomaly_low'].append(ci)
    for ci in ao[third:2 * third]:
        anom_groups['anomaly_mid'].append(ci)
    for ci in ao[2 * third:]:
        anom_groups['anomaly_high'].append(ci)
    anom_group_means = {g: (sum(valid[ci] for ci in cis if ci in valid) /
                            max(1, sum(1 for ci in cis if ci in valid)))
                        for g, cis in anom_groups.items()}

    # 按 CKR available ratio 分组 (只对动态 CKR 有意义)
    ckr_groups = {}
    if tracker is not None and tracker.last_available_mask is not None:
        avail_ratio = {ci: float(tracker.last_available_mask[ci].float().mean().item())
                       for ci in range(len(subgraphs))}
        ckr_groups = {'ckr_available_ratio_by_client': avail_ratio}

    return {
        'worst_client_metric': worst,
        'bottom_10pct_client_mean': bottom10,
        'bottom_20pct_client_mean': bottom20,
        'best_client_metric': best,
        'client_metric_mean': mean,
        'client_metric_std': std,
        'client_metric_iqr': iqr,
        'performance_gap': best - worst,
        'unavailable_client_count': len(vals) - n,
        'zero_anomaly_train_client_count': zero_anom_train,
        'zero_anomaly_test_client_count': zero_anom_test,
        'grouping_by_train_size': group_means,
        'grouping_by_anomaly_train': anom_group_means,
        **ckr_groups,
    }

def reseed_client_rng(args, round_id, ci):
    """
    客户端本地随机流与服务器路径解耦 (paired-arm 一致性修复):

    每个 (round, client) 使用确定性种子重播种 torch (cpu/cuda) 与 python random。
    不做 save/restore —— 后续客户端/轮次都会重新播种, 服务器侧 (如 Stage2 采样、
    nn.Embedding 初始化等) 的 RNG 消耗无法再污染客户端训练轨迹。
    效果: C/B 同一 (round, client) 的 dropout / edge perturbation / 锚点采样
    随机流逐位一致, 与服务器代码消耗 RNG 的数量完全无关。
    """
    local_seed = args.seed + round_id * 1000 + ci * 37
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(local_seed)
    random.seed(local_seed)


def make_server_rng(device, base_seed, round_id, stream_id):
    """Stage2 专用 RNG，与客户端全局 RNG 完全解耦。

    每个 (round, stream) 独立确定性播种。这样 generator_steps 等服务器路径
    的 RNG 消耗不会改变下一轮客户端随机流，也不会改变 final-sampling stream。
    """
    seed = (int(base_seed) * 1_000_003 + int(round_id) * 10_007
            + int(stream_id) * 97) % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(seed)


def _model_param_hash(model):
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def _tensor_hash(t):
    if t is None:
        return 'None'
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()[:12]


def _rng_fingerprint():
    """torch(cpu/cuda) + python random 全局 RNG 状态的稳定 hash (仅诊断)。"""
    h = hashlib.md5()
    h.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    h.update(pickle.dumps(random.getstate()))
    return h.hexdigest()[:12]


def set_requires_grad(module, enabled):
    for p in module.parameters():
        p.requires_grad_(enabled)


def formal_finite_guard(args, round_id, stage, tensors=None, modules=None,
                        check_grads=False):
    """正式战役 fail-fast 数值守门。

    health checker 不能只依赖 stdout 正则；正式协议下在训练进程内部直接检查
    raw/projected feature、loss、参数以及 backward 后梯度，发现 NaN/Inf 立即失败。
    """
    if not getattr(args, 'formal_campaign_guard', False):
        return
    for name, t in (tensors or {}).items():
        if t is not None and not bool(torch.isfinite(t).all()):
            raise FloatingPointError(
                f"[FORMAL-NONFINITE] round={round_id} stage={stage} tensor={name}")
    for name, module in (modules or {}).items():
        for pn, p in module.named_parameters():
            target = p.grad if check_grads else p.data
            if target is not None and not bool(torch.isfinite(target).all()):
                kind = 'grad' if check_grads else 'param'
                raise FloatingPointError(
                    f"[FORMAL-NONFINITE] round={round_id} stage={stage} "
                    f"{kind}={name}.{pn}")


def configure_generator_trainability(generator, trainable,
                                     freeze_skip_scale=False):
    """
    设置生成器参数 trainable 状态 (Bug 1 修复)。

    freeze_skip_scale=True 时, timestep_scalar 的 skip scale c(t) 无论
    trainable 如何切换, 恒保持 requires_grad=False —— 真正冻结, 不依赖
    "optimizer 里没有它" 的间接保证。
    """
    skip_p = None
    if (freeze_skip_scale
            and getattr(generator, 'skip_mode', 'none') == 'timestep_scalar'):
        skip_p = generator.skip_log_scale.weight
    for p in generator.parameters():
        if p is skip_p:
            p.requires_grad_(False)
        else:
            p.requires_grad_(trainable)


def validate_formal_protocol(args):
    """
    正式 multiclass 主实验协议 guard (--formal_multiclass_protocol_guard 时生效):
    启动时打印协议字段, 任一不满足立即报错退出。不静默修参数。
    """
    fields = [
        ('task_mode', 'multiclass', args.task_mode),
        ('selection_metric', 'accuracy', args.selection_metric),
        ('reliability_holdout_ratio', 0, args.reliability_holdout_ratio),
        ('use_weighted_ce', True, bool(args.use_weighted_ce)),
        ('contrastive_mode', 'subgraph_cross_view', args.contrastive_mode),
        ('local_optimizer_lifecycle', 'reset_each_round',
         args.local_optimizer_lifecycle),
    ]
    print("\n[Formal Multiclass Protocol]")
    for name, expect, got in fields:
        ok = (got == expect)
        print(f"  {name:<26} = {got}  (期望 {expect}) {'✓' if ok else '✗'}")
    print(f"  {'test_used_for_selection':<26} = False  (协议声明)")
    bad = [name for name, expect, got in fields if got != expect]
    if bad:
        raise SystemExit(
            "[Formal Multiclass Protocol] 违反: " + ", ".join(bad)
            + " —— 停止, 不自动修参数")


def validate_formal_campaign(args):
    """正式战役 guard: 全部 protocol 字段必须满足, 违反即 ValueError。"""
    import os as _os
    checks = [
        ('task_mode', args.task_mode, 'multiclass'),
        ('use_weighted_ce', bool(args.use_weighted_ce), True),
        ('reliability_holdout_ratio', args.reliability_holdout_ratio, 0),
        ('selection_metric', args.selection_metric, 'accuracy'),
        ('local_optimizer_lifecycle', args.local_optimizer_lifecycle,
         'reset_each_round'),
        ('radius_constraint', bool(args.radius_constraint), True),
        ('feature_stats_align', bool(args.feature_stats_align), False),
        ('diffusion_freeze_skip_scale',
         bool(getattr(args, 'diffusion_freeze_skip_scale', False)), True),
        ('federated_diffusion_pretrain',
         bool(getattr(args, 'federated_diffusion_pretrain', False)), False),
        ('contrastive_mode', getattr(args, 'contrastive_mode', None),
         'subgraph_cross_view'),
        ('distill_loss_type', getattr(args, 'distill_loss_type', None), 'kl'),
        ('diffusion_skip_mode', getattr(args, 'diffusion_skip_mode', None),
         'timestep_scalar'),
        ('diffusion_steps', int(getattr(args, 'diffusion_steps', -1)), 20),
        ('diffusion_beta_end', float(getattr(args, 'diffusion_beta_end', -1)), 0.5),
        ('use_posterior_variance',
         bool(getattr(args, 'use_posterior_variance', False)), True),
        ('lambda_diffusion_anchor',
         float(getattr(args, 'lambda_diffusion_anchor', -1)), 1e-2),
        ('generator_sampling_steps',
         int(getattr(args, 'generator_sampling_steps', -1)), 0),
        ('generator_backprop_mode',
         getattr(args, 'generator_backprop_mode', None), 'checkpointed'),
        ('checkpoint_segments', int(getattr(args, 'checkpoint_segments', -1)), 1),
        ('fake_graph_topology', getattr(args, 'fake_graph_topology', None), 'knn'),
        ('test_isolation', bool(getattr(args, 'tuning_mode', False)
                                or getattr(args, 'final_test_only', False)), True),
    ]
    bad = [n for n, got, exp in checks if got != exp]
    ck = getattr(args, 'diffusion_pretrained_checkpoint', '')
    if not ck or not _os.path.exists(ck):
        bad.append(f'diffusion_pretrained_checkpoint({ck} 不存在)')
    if bad:
        raise ValueError(
            "[Formal Campaign Guard] 违反正式协议: " + ", ".join(bad)
            + " —— 停止, 不自动修参数")


def validate_run_args(args):
    """互斥/非法 CLI 组合 fail-fast (Bug 3/4/9)。不静默猜测用户意图。"""
    if (getattr(args, 'tuning_mode', False)
            and getattr(args, 'final_test_only', False)):
        raise ValueError(
            "--tuning_mode 与 --final_test_only 互斥: 调参期 test 必须完全隔离；"
            "正式决赛才允许训练结束后计算一次 test。")
    if (getattr(args, 'resume_checkpoint', '')
            and getattr(args, 'diffusion_pretrained_checkpoint', '')):
        raise ValueError(
            "--resume_checkpoint 与 --diffusion_pretrained_checkpoint 互斥: "
            "resume Stage2 和 fresh Stage2 pretrained 初始化不能同时使用。")
    if (getattr(args, 'diffusion_pretrained_checkpoint', '')
            and getattr(args, 'federated_diffusion_pretrain', False)):
        raise ValueError(
            "Fresh Stage2 from pretrained checkpoint 要求 "
            "--no-federated_diffusion_pretrain (禁止加载 R20 后又二次联邦预训练)")
    if (getattr(args, 'radius_constraint', False)
            and getattr(args, 'feature_stats_align', False)):
        raise ValueError(
            "--radius_constraint 与 --feature_stats_align 不能同时开启: "
            "事后 stats 对齐会破坏 radius manifold 不变式 "
            "(feature_stats_align 仅供独立消融使用)")


def resolve_resume_freeze_policy(saved_args, cli_freeze):
    """
    Bug A: resume 时 freeze 策略必须与 checkpoint 一致, 否则 fail-fast。

    - 新 checkpoint (args 含 diffusion_freeze_skip_scale): 严格比对, 不一致 raise;
    - legacy checkpoint (无该字段): 明确打印 warning, 按当前 CLI 策略执行
      (未验证一致, 不伪装成已验证)。
    返回实际采用的 freeze policy。
    """
    if isinstance(saved_args, dict) and 'diffusion_freeze_skip_scale' in saved_args:
        saved_freeze = bool(saved_args['diffusion_freeze_skip_scale'])
        if saved_freeze != bool(cli_freeze):
            raise ValueError(
                "Resume freeze 策略不一致: "
                f"checkpoint diffusion_freeze_skip_scale={saved_freeze}, "
                f"当前 CLI diffusion_freeze_skip_scale={bool(cli_freeze)}. "
                "Resume requires the same Stage-2 skip-freeze policy "
                "as the checkpoint.")
        return saved_freeze
    print("  [Resume] legacy checkpoint: 无 diffusion_freeze_skip_scale 字段, "
          "无法核验 freeze 策略; 按当前 CLI 策略 "
          f"({bool(cli_freeze)}) 执行 (未验证一致)")
    return bool(cli_freeze)


def validate_generator_cfg(saved_cfg, current_cfg):
    """Bug 10: resume 时 generator 配置逐项核对; 缺失字段视为 legacy 并跳过。"""
    for k in ('feat_dim', 'num_classes', 'hidden_dim', 'num_steps',
              'beta_start', 'beta_end', 'skip_mode', 'use_posterior_variance'):
        sv = saved_cfg.get(k)
        if sv is None:
            print(f"  [Resume] legacy generator_cfg: 缺字段 {k}, 依赖 "
                  f"strict state_dict 校验")
            continue
        if sv != current_cfg.get(k):
            raise ValueError(
                f"[Resume] generator_cfg 不兼容: {k} 期望 "
                f"{current_cfg.get(k)}, checkpoint 为 {sv}")


def init_stage2_from_pretrained(generator, ckpt_path, args, device,
                                num_classes):
    """
    Fresh Stage 2 初始化 (Mode A, Bug 2/10):
      加载 pretrained DDPM 权重 → 校验配置一致性 (state_dict buffers, 严格) →
      应用 freeze 策略 → θ_pre 快照 → radius bank → 重建对抗期 optimizer。
    不恢复 Stage 1 Adam 状态。返回 (pretrain_ref, radius_bank, gen_optimizer)。
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cur_sd = generator.state_dict()
    for k in ('betas', 'alphas', 'alpha_bars', 'posterior_variance'):
        if k not in ck or not torch.allclose(ck[k].float().to(device),
                                             cur_sd[k].float()):
            raise ValueError(f"[Stage2] checkpoint 调度 buffer 不一致: {k}")
    if (getattr(generator, 'skip_mode', 'none') != 'timestep_scalar'
            or 'skip_log_scale.weight' not in ck):
        raise ValueError("[Stage2] checkpoint 必须为 timestep_scalar 且含 "
                         "skip_log_scale (当前配置不符)")
    if (ck['net.0.weight'].shape != cur_sd['net.0.weight'].shape
            or ck['time_embed.0.weight'].shape
            != cur_sd['time_embed.0.weight'].shape):
        raise ValueError("[Stage2] checkpoint denoiser 形状与当前配置不一致")
    print("  [Stage2] legacy pretrained checkpoint: validated from "
          "state_dict buffers (无 metadata, 逐项核验通过)")
    generator.load_state_dict(ck)   # strict=True, key 不匹配直接报错
    configure_generator_trainability(
        generator, True, freeze_skip_scale=args.diffusion_freeze_skip_scale)
    if args.diffusion_freeze_skip_scale:
        print("  [Stage2] skip scale c(t) 已冻结于 pretrained 值 "
              "(residual denoiser 仍可训练)")
    pretrain_ref = {name: p.detach().clone()
                    for name, p in generator.named_parameters()}
    radius_bank = None
    if args.radius_constraint:
        radius_bank = build_pretrain_radius_bank(
            generator, num_classes, device, args.radius_bank_size,
            rng_seed=args.seed + 7777)   # 专用 RNG, 不消耗全局流
        fin = bool(torch.isfinite(radius_bank).all()
                   and (radius_bank > 0).all())
        qs = torch.quantile(radius_bank,
                            torch.tensor([0.25, 0.5, 0.75], device=device),
                            dim=1)
        print(f"  [Stage2 Radius Bank] 每类 p25={qs[0].tolist()} "
              f"median={qs[1].tolist()} p75={qs[2].tolist()} | "
              f"finite/positive={fin} "
              f"(注意: bank 仍约 4.6x 真实半径, 如实记录)")
        if not fin:
            raise RuntimeError("[Stage2] radius bank 含 NaN/Inf/非正, 停止")
    gen_optimizer = Adam([p for p in generator.parameters()
                          if p.requires_grad],
                         lr=args.generator_lr, weight_decay=0.0)
    print("  [Stage2] 重建对抗期 generator optimizer "
          f"({sum(1 for p in generator.parameters() if p.requires_grad)}"
          f"/{sum(1 for _ in generator.parameters())} 参数组可训练, "
          f"预训练 Adam 状态不恢复)")
    return pretrain_ref, radius_bank, gen_optimizer


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _nan_diag(modules, tag):
    """NaN 诊断: FEDTAD_NAN_DIAG=1 时打印第一个含 NaN 的参数/梯度。正常路径零开销。"""
    if os.environ.get('FEDTAD_NAN_DIAG') != '1':
        return False
    found = False
    for mod in modules:
        for p in mod.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"[NAN-DIAG] {tag}: grad NaN in {type(mod).__name__} "
                      f"({p.shape})")
                found = True
            if torch.isnan(p).any():
                print(f"[NAN-DIAG] {tag}: weight NaN in {type(mod).__name__} "
                      f"({p.shape})")
                found = True
    if not found:
        print(f"[NAN-DIAG] {tag}: no NaN")
    return found


def build_knn_graph(fake_x, k):
    B = fake_x.shape[0]
    actual_k = min(k, B - 1) if B > 1 else 1
    with torch.no_grad():
        x_norm = F.normalize(fake_x.detach(), p=2, dim=1)
        sim = torch.mm(x_norm, x_norm.t())
        sim.fill_diagonal_(-float('inf'))
        _, topk_idx = torch.topk(sim, k=actual_k, dim=1)
        rows = torch.arange(B, device=fake_x.device).unsqueeze(1).expand(-1, actual_k)
        edge_index = torch.stack([rows.reshape(-1), topk_idx.reshape(-1)], dim=0)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        from torch_geometric.utils import add_self_loops
        edge_index, _ = add_self_loops(edge_index)
        edge_index = torch.unique(edge_index, dim=1)
    return Data(x=fake_x, edge_index=edge_index)


def sample_fake_labels(num_nodes, num_classes, strategy='balanced',
                       ckr_weights=None, device='cpu', generator=None):
    if strategy == 'balanced':
        per_class = [num_nodes // num_classes] * num_classes
        for i in range(num_nodes % num_classes):
            per_class[i] += 1
    elif ckr_weights is not None:
        probs = torch.tensor(ckr_weights, device=device).float()
        probs = probs / (probs.sum() + 1e-8)
        labels = torch.multinomial(probs, num_nodes, replacement=True,
                                   generator=generator)
        per_class = [int((labels == c).sum()) for c in range(num_classes)]
        return labels, {c: per_class[c] for c in range(num_classes)}
    else:
        per_class = [num_nodes // num_classes] * num_classes
        for i in range(num_nodes % num_classes):
            per_class[i] += 1
    labels = torch.zeros(num_nodes, device=device, dtype=torch.long)
    ptr = 0
    for c, cnt in enumerate(per_class):
        labels[ptr:ptr + cnt] = c
        ptr += cnt
    return labels, {c: per_class[c] for c in range(num_classes)}


def normalize_ckr_safe(ckr):
    eps = 1e-8
    denom = ckr.sum(dim=0, keepdim=True)
    zero_mask = denom.squeeze() < eps
    normalized = ckr / (denom + eps)
    if zero_mask.any():
        zero_classes = torch.where(zero_mask)[0].tolist()
        print(f"  [CKR] ⚠ 类别 {zero_classes} 总可靠性为 0, 退化为均匀权重")
        for c in zero_classes:
            normalized[:, c] = 1.0 / ckr.shape[0]
    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
        print("  [CKR] ⚠ NaN/Inf 使用均匀回退")
        normalized = torch.ones_like(ckr) / ckr.shape[0]
    return normalized


def resolve_selection_metric(task_mode, user_choice=''):
    """默认 selection_metric: multiclass=macro_f1, anomaly_binary=pooled_pr_auc。"""
    if user_choice:
        return user_choice
    return 'macro_f1' if task_mode == 'multiclass' else 'pooled_pr_auc'


def _split_ratios(dataset_name):
    """与 base_util.load_dataset 一致的 train/val 比例 (用于映射后重划分)。"""
    if dataset_name in ["Cora", "CiteSeer", "PubMed", "Computers", "Photo",
                        "CS", "Physics", "NELL"]:
        return 0.2, 0.4
    if dataset_name in ["ogbn-arxiv", "Flickr"]:
        return 0.6, 0.2
    if dataset_name == "Reddit":
        return 0.8, 0.1
    if dataset_name == "ogbn-products":
        return 0.1, 0.05
    return 0.2, 0.4


# =====================================================================
#  标签映射 (anomaly_binary 模式)
# =====================================================================

def apply_label_mapping(subgraphs, dataset, normal_classes, anomaly_classes):
    """
    将原始多分类标签映射为二分类异常检测标签。
    映射发生在模型/CKR/权重初始化之前。

    返回:
        mapped_subgraphs: list of Data (y 已被映射)
        num_classes: 2
        mapping_info: dict
    """
    nc = [int(x) for x in normal_classes.split(',') if x.strip()]
    ac = [int(x) for x in anomaly_classes.split(',') if x.strip()]

    all_original = set()
    for sg in subgraphs:
        all_original.update(sg.y.unique().tolist())
    all_original = sorted(all_original)

    overlap = set(nc) & set(ac)
    assert not overlap, f"normal_classes 和 anomaly_classes 重叠: {overlap}"
    missing = set(all_original) - set(nc) - set(ac)
    assert not missing, f"以下类别未分配: {missing}"

    mapping = {}
    for c in nc:
        mapping[c] = 0  # normal
    for c in ac:
        mapping[c] = 1  # anomaly

    for sg in subgraphs:
        old_y = sg.y.clone()
        new_y = old_y.clone()
        for old_c, new_c in mapping.items():
            new_y[old_y == old_c] = new_c
        sg.y = new_y
        sg.original_y = old_y

    counts_before = {c: 0 for c in all_original}
    for sg in subgraphs:
        for c in all_original:
            counts_before[c] += (sg.original_y == c).sum().item()

    counts_after = {0: 0, 1: 0}
    for sg in subgraphs:
        for c in [0, 1]:
            counts_after[c] += (sg.y == c).sum().item()

    anomaly_ratio = counts_after[1] / (counts_after[0] + counts_after[1] + 1e-8)

    print(f"  [Anomaly Mapping]")
    print(f"    normal  classes: {nc}")
    print(f"    anomaly classes: {ac}")
    print(f"    normal  count:   {counts_after[0]}")
    print(f"    anomaly count:   {counts_after[1]}")
    print(f"    anomaly ratio:   {anomaly_ratio:.4f}")

    # 按原始类别频率排序的推荐映射
    freq_sorted = sorted(counts_before.items(), key=lambda x: x[1])
    print(f"    [推荐] 按频率升序: {freq_sorted}")

    return subgraphs, 2, {
        'mapping': mapping,
        'normal_classes': nc,
        'anomaly_classes': ac,
        'counts_before': counts_before,
        'counts_after': counts_after,
        'anomaly_ratio': anomaly_ratio,
    }


def apply_label_mapping_to_data(data, normal_classes, anomaly_classes):
    """对单个 PyG Data 对象应用标签映射。"""
    nc = [int(x) for x in normal_classes.split(',') if x.strip()]
    ac = [int(x) for x in anomaly_classes.split(',') if x.strip()]
    mapping = {}
    for c in nc:
        mapping[c] = 0
    for c in ac:
        mapping[c] = 1
    old_y = data.y.clone()
    new_y = old_y.clone()
    for old_c, new_c in mapping.items():
        new_y[old_y == old_c] = new_c
    data.y = new_y
    data.original_y = old_y
    return data


# =====================================================================
#  S1: CKR 计算
# =====================================================================

def compute_ckr(subgraphs, num_classes, args, device, data_identity=None):
    # CKR 完全由当前子图/标签/训练 mask 决定。仅用 dataset/seed 命名不足以防止
    # “同名缓存但 data*.pt 或 split 已变化”的静默复用；正式路径绑定 data identity。
    cache_suffix = '_resplit' if getattr(args, 'resplit_after_label_mapping', False) else ''
    identity_suffix = (f"_d{str(data_identity)[:12]}"
                       if data_identity else '')
    ckr_path = (f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}_"
                f"{args.task_mode}{cache_suffix}_s{args.seed}{identity_suffix}.pt")
    if os.path.exists(ckr_path):
        ckr = torch.load(ckr_path, map_location='cpu').to(device)
        print(f"  [CKR] 加载缓存: {ckr_path}")
        return ckr
    os.makedirs("./ckr", exist_ok=True)
    ckr = torch.zeros((args.num_clients, num_classes)).to(device)
    for client_id in range(args.num_clients):
        data = subgraphs[client_id]
        topo_emb = cal_topo_emb(
            edge_index=data.edge_index,
            num_nodes=data.x.shape[0],
            max_walk_length=5,
        ).to(device)
        h = torch.cat((data.x.to(device), topo_emb), dim=1)
        train_nodes = torch.where(data.train_idx)[0]
        for train_i in train_nodes:
            neighbors = data.edge_index[1, :][data.edge_index[0, :] == train_i]
            if neighbors.shape[0] == 0:
                continue
            sims = F.cosine_similarity(h[train_i].unsqueeze(0), h[neighbors], dim=1)
            avg_sim = sims.mean()
            label = data.y[train_i].item()
            ckr[client_id, label] += avg_sim.item()
    torch.save(ckr, ckr_path)
    print(f"  [CKR] 计算完毕: {ckr_path}")
    return ckr


def federated_diffusion_pretrain(generator, subgraphs, args, device):
    """
    联邦扩散预训练 (专利 S3.1 / S3.2 + 专利权1).

    问题背景: 原实现在联邦无数据设定下, 去噪网络**从未被训练过** ——
    没有前向加噪、没有噪声预测损失, 生成器只被 L_sem/L_dis/L_div 训练,
    等于"套着扩散外壳的 MLP": 承担扩散的全部代价, 却没有学到真实特征分布。
    实测后果: 伪特征 σ 恒为 0.655 与数据无关(真实 σ 为 0.112/0.092/0.018),
    教师看到 OOD 输入 -> 蒸馏梯度爆炸(PubMed |grad| 0.29->280) -> 方法完全失效。

    本函数按专利 S3.1/S3.2 补齐标准 DDPM 训练目标:
        前向:  q(x_t|x_0) = N(x_t; √(ᾱ_t)·x_0, (1−ᾱ_t)·I)
        损失:  L = E[‖ε − ε_θ(x_t, t, c)‖²]
    同时满足联邦约束(专利权1「客户端本地计算、仅上传」):
        每个客户端**只在本地**用自己的真实特征训练去噪网络,
        只上传去噪网络参数; 服务端按节点数加权聚合。**原始特征不出域**。

    返回: 聚合后的去噪网络 (原地写回 generator)
    """
    import copy
    from torch.optim import Adam as _Adam

    alpha_bar_T = generator.alpha_bars[-1].item()
    if alpha_bar_T > 0.05:
        print(f"  ⚠️ [联邦扩散预训练] 噪声调度未退化: alpha_bar_T={alpha_bar_T:.4f} "
              f"(专利 S3.1 要求 → 纯噪声)。"
              f"建议 --diffusion_steps 20 --diffusion_beta_end 0.5 (alpha_bar_T=0.002)。"
              f"当前调度下去噪任务无法学到从纯噪声还原, 预训练效果将大打折扣。")
    else:
        print(f"  [联邦扩散预训练] 调度检查通过: alpha_bar_T={alpha_bar_T:.5f} ≈ 纯噪声 (符合专利 S3.1)")

    # 续跑模式: 从保存的 denoiser 状态继续 (纵向诊断用)
    if getattr(args, 'diffusion_pretrain_resume', ''):
        ck = torch.load(args.diffusion_pretrain_resume, map_location=device,
                        weights_only=False)
        generator.load_state_dict(ck)
        print(f"  [联邦扩散预训练] 从 {args.diffusion_pretrain_resume} 恢复 denoiser, "
              f"续跑 {args.diffusion_pretrain_rounds} 轮 (此前已完成 "
              f"{args.diffusion_pretrain_resume_round} 轮)")

    # 每个客户端本地参与训练的样本 = 该客户端的训练节点 (fit + reliability 之外不碰)
    client_data = []
    for ci, sg in enumerate(subgraphs):
        tr = torch.where(sg.train_idx)[0]
        if tr.numel() == 0:
            continue
        client_data.append((ci, sg.x[tr].to(device).float(), sg.y[tr].to(device)))
    if not client_data:
        print("  [联邦扩散预训练] 无可用训练节点, 跳过")
        return generator

    print(f"  [联邦扩散预训练] {args.diffusion_pretrain_rounds} 联邦轮 × "
          f"{args.diffusion_pretrain_epochs} 本地 epoch, batch={args.diffusion_pretrain_batch}, "
          f"lr={args.diffusion_pretrain_lr}")
    print(f"  [联邦扩散预训练] 上行内容: 去噪网络参数 only（原始特征不出域, 符合专利权1）")

    # ---- S2 诊断: 每客户端固定 diag subset (x0/t/ε 固定, 仅本地计算, 只上行标量) ----
    # 隐私边界: 服务器只收 (n_diag, MSE) 标量, 不收任何诊断特征/原始数据。
    T_steps = generator.num_steps
    bucket_defs = {'early': (0, T_steps // 3),
                   'mid': (T_steps // 3, 2 * T_steps // 3),
                   'late': (2 * T_steps // 3, T_steps)}
    diag = []
    for ci, x, y in client_data:
        n = x.shape[0]
        n_diag = min(128, n)
        # randperm 只支持 CPU generator; randint/randn 要求与输出同设备 -> 两个同种子 generator
        g_cpu = torch.Generator(device='cpu').manual_seed(
            args.seed + ci * 1000 + 7)
        g_dev = torch.Generator(device=x.device).manual_seed(
            args.seed + ci * 1000 + 7)
        idx = torch.randperm(n, generator=g_cpu)[:n_diag]
        x0d = x[idx]
        yd = y[idx]
        td = torch.randint(0, T_steps, (n_diag,), generator=g_dev,
                           device=x.device)
        epsd = torch.randn(x0d.shape, generator=g_dev, device=x.device,
                           dtype=x0d.dtype)
        ab = generator.alpha_bars[td].unsqueeze(-1)
        xtd = torch.sqrt(ab) * x0d + torch.sqrt(1.0 - ab) * epsd
        diag.append((ci, x0d, yd, td, epsd, xtd))

    def _diag_eval(model, dg):
        """在固定 (x_t, t, ε) 上评估 denoise MSE + timestep 分桶。无梯度。"""
        _ci, _x0d, _yd, _td, _epsd, _xtd = dg
        with torch.no_grad():
            pred = model(_xtd, _td, _yd)
            mse = float(((pred - _epsd) ** 2).mean().item())
            bm = {}
            for k, (lo, hi) in bucket_defs.items():
                mask = (_td >= lo) & (_td < hi)
                bm[k] = (float(((pred[mask] - _epsd[mask]) ** 2).mean().item())
                         if mask.any() else float('nan'))
        return mse, bm

    def _weighted(vals, ns):
        tot = sum(ns)
        return sum(v * n for v, n in zip(vals, ns)) / max(tot, 1)

    s2_records = []
    longitudinal = []
    total_steps = 0

    # ---- 纵向 raw-sampling probe: 固定 labels + 固定 initial noise (跨轮可比) ----
    num_classes_probe = int(max(int(sg.y.max().item()) for sg in subgraphs)) + 1
    probe_per_class = 32
    probe_labels = torch.arange(num_classes_probe, device=device).repeat_interleave(
        probe_per_class)
    probe_g = torch.Generator(device=device).manual_seed(args.seed + 4242)
    probe_noise = torch.randn(probe_labels.shape[0], generator.feat_dim,
                              generator=probe_g, device=device)
    probe_rng = torch.Generator(device=device)   # 逐步反向噪声流, 每 checkpoint 重置
    # 真实特征统计 (客户端本地统计, 只上行标量)
    real_std_w = real_med_w = real_mean_w = real_p95_w = real_max_w = 0.0
    tot_real_n = 0
    for _ci, _x, _y in client_data:
        n_real = _x.shape[0]
        r_real = torch.linalg.vector_norm(_x, ord=2, dim=1)
        real_std_w += float(_x.std().item()) * n_real
        real_med_w += float(r_real.median().item()) * n_real
        real_mean_w += float(r_real.mean().item()) * n_real
        real_p95_w += float(torch.quantile(r_real, 0.95).item()) * n_real
        real_max_w += float(r_real.max().item()) * n_real
        tot_real_n += n_real
    real_std = real_std_w / max(tot_real_n, 1)
    real_med = real_med_w / max(tot_real_n, 1)
    real_mean = real_mean_w / max(tot_real_n, 1)
    real_p95 = real_p95_w / max(tot_real_n, 1)
    real_max = real_max_w / max(tot_real_n, 1)
    zero_base = _weighted([float((d[4] ** 2).mean().item()) for d in diag],
                          [d[1].shape[0] for d in diag])
    abT = generator.alpha_bars[-1].item()
    print(f"  [schedule] T={T_steps} beta_start={args.diffusion_beta_start:.0e} "
          f"beta_end={args.diffusion_beta_end} alpha_bar_T={abT:.5f} "
          f"1/sqrt(alpha_bar_T)={1.0 / math.sqrt(abT):.3f}")
    print(f"  [real stats] std={real_std:.4f} median_r={real_med:.4f} "
          f"mean_r={real_mean:.4f} p95_r={real_p95:.4f} max_r={real_max:.4f} "
          f"(n={tot_real_n})")
    print(f"  [probe] {probe_labels.shape[0]} 固定节点 (32/类), 固定 initial "
          f"noise, checkpoint: R0/R1/R3/R5 + 每 5 轮")
    # R0 (初始化) raw probe: 训练前基线 (residual skip 下未训练网络是否已不爆炸)
    generator.eval()
    probe_rng.manual_seed(args.seed + 9173)   # reverse 噪声流, 与 initial noise (seed+4242) 独立
    with torch.no_grad():
        raw_probe0 = generator.sample(
            labels=probe_labels, device=device,
            apply_output_bound=False, initial_noise=probe_noise,
            noise_generator=probe_rng)
    r0_probe = torch.linalg.vector_norm(raw_probe0, ord=2, dim=1)
    longitudinal.append({
        'round': 0, 'approx_local_steps': 0,
        'raw_std': float(raw_probe0.std().item()),
        'raw_median_r': float(r0_probe.median().item()),
        'raw_mean_r': float(r0_probe.mean().item()),
        'raw_p95_r': float(torch.quantile(r0_probe, 0.95).item()),
        'raw_max_r': float(r0_probe.max().item()),
        'radius_vs_real': round(float(r0_probe.median().item())
                                / max(real_med, 1e-12), 1),
        'std_vs_real': round(float(raw_probe0.std().item())
                             / max(real_std, 1e-12), 1)})
    print(f"    [LONG] r0 (初始化): raw σ={longitudinal[0]['raw_std']:.2f} "
          f"median_r={longitudinal[0]['raw_median_r']:.1f} "
          f"r/real={longitudinal[0]['radius_vs_real']}x")

    bs = args.diffusion_pretrain_batch
    for rnd in range(args.diffusion_pretrain_rounds):
        states, weights = [], []
        base_state = {k: v.detach().clone() for k, v in generator.state_dict().items()}
        before, after = [], []
        local_train_mse = 0.0
        local_train_n = 0
        steps_this_round = 0
        for (ci, x, y), dg in zip(client_data, diag):
            local_gen = copy.deepcopy(generator)
            local_gen.load_state_dict(base_state)
            local_gen.train()
            opt = _Adam(local_gen.parameters(), lr=args.diffusion_pretrain_lr)
            n = x.shape[0]
            # A. global_before (下发模型在本地固定 diag 上)
            b_mse, _ = _diag_eval(local_gen, dg)
            before.append(b_mse)
            # 本地训练 (按实际样本数加权统计, 修复旧口径 ÷epochs 的虚高)
            for _ep in range(args.diffusion_pretrain_epochs):
                perm = torch.randperm(n, device=device)
                for s in range(0, n, bs):
                    idx = perm[s:s + bs]
                    if idx.numel() < 2:
                        continue
                    opt.zero_grad()
                    loss = local_gen.denoise_loss(x[idx], y[idx])
                    loss.backward()
                    opt.step()
                    local_train_mse += loss.item() * idx.numel()
                    local_train_n += idx.numel()
                    steps_this_round += 1
            # B. local_after (同一 diag 上)
            a_mse, _ = _diag_eval(local_gen, dg)
            after.append(a_mse)
            states.append({k: v.detach().clone()
                           for k, v in local_gen.state_dict().items()})
            weights.append(float(n))
            del local_gen, opt
        # 服务端加权聚合 (FedAvg over denoiser state_dict; buffer 均相同, 平均不改变)
        tot_w = sum(weights)
        new_state = {}
        for k in states[0]:
            acc = None
            for st, w in zip(states, weights):
                term = st[k].float() * (w / tot_w)
                acc = term if acc is None else acc + term
            new_state[k] = acc.to(generator.state_dict()[k].dtype)
        generator.load_state_dict(new_state)
        # C. aggregated_global: 每客户端在本地 diag 上评估新全局模型 (只上报标量)
        generator.eval()
        agg_vals, agg_ns, agg_buckets = [], [], {k: [] for k in bucket_defs}
        for dg in diag:
            m, bm = _diag_eval(generator, dg)
            agg_vals.append(m)
            agg_ns.append(dg[1].shape[0])
            for k in bucket_defs:
                agg_buckets[k].append(bm[k])
        w_before = _weighted(before, [d[1].shape[0] for d in diag])
        w_after = _weighted(after, [d[1].shape[0] for d in diag])
        w_agg = _weighted(agg_vals, agg_ns)
        local_gain = w_before - w_after
        agg_gain = w_before - w_agg
        agg_gap = w_agg - w_after
        retention = (agg_gain / (local_gain + 1e-12)
                     if local_gain > 1e-4 else float('nan'))
        rec = {'round': rnd,
               'weighted_global_before': round(w_before, 5),
               'weighted_local_after': round(w_after, 5),
               'weighted_aggregated_global': round(w_agg, 5),
               'local_gain': round(local_gain, 5),
               'aggregation_gain': round(agg_gain, 5),
               'aggregation_gap': round(agg_gap, 5),
               'aggregation_retention': (round(retention, 4)
                                         if not math.isnan(retention) else None),
               'local_train_mse': round(local_train_mse / max(local_train_n, 1), 5),
               'agg_bucket_mse': {k: round(_weighted(
                   [v for v in agg_buckets[k] if not math.isnan(v)],
                   [ns for v, ns in zip(agg_buckets[k], agg_ns)
                    if not math.isnan(v)]), 5)
                   if any(not math.isnan(v) for v in agg_buckets[k]) else None
                   for k in bucket_defs},
               'per_client': {
                   str(ci): {'n_diag': dg[1].shape[0],
                             'before': round(before[i], 5),
                             'after': round(after[i], 5),
                             'improvement': round(before[i] - after[i], 5),
                             'improvement_pct': round(
                                 (before[i] - after[i]) / max(before[i], 1e-12)
                                 * 100, 2)}
                   for i, (ci, dg) in enumerate(
                       zip([c for c, _, _ in client_data], diag))}}
        s2_records.append(rec)
        print(f"  [扩散预训练] round {rnd}: 样本加权平均去噪损失="
              f"{rec['local_train_mse']:.5f}")
        print(f"    [S2] before={w_before:.5f} local_after={w_after:.5f} "
              f"agg_global={w_agg:.5f} | local_gain={local_gain:.5f} "
              f"agg_gain={agg_gain:.5f} gap={agg_gap:.5f} "
              f"retention={rec['aggregation_retention']}")
        print(f"    [S2] agg buckets: early={rec['agg_bucket_mse']['early']} "
              f"mid={rec['agg_bucket_mse']['mid']} "
              f"late={rec['agg_bucket_mse']['late']}")
        pc = rec['per_client']
        cids = sorted(pc, key=lambda k: pc[k]['improvement'])
        best_c = pc[cids[-1]] if cids else None
        worst_c = pc[cids[0]] if cids else None
        print(f"    [S2] 客户端明细 ({len(pc)} 个, 全量见 diffusion_s2_diag.json): "
              f"最优 c{cids[-1]} {best_c['before']:.3f}→{best_c['after']:.3f} "
              f"(+{best_c['improvement']:.4f}), "
              f"最差 c{cids[0]} {worst_c['before']:.3f}→{worst_c['after']:.3f} "
              f"(+{worst_c['improvement']:.4f})")
        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(os.path.join(args.checkpoint_dir,
                                   'diffusion_s2_diag.json'), 'w') as f:
                json.dump(s2_records, f, indent=1)
        # ---- 纵向 raw-sampling checkpoint (R1/R3 + 每 5 轮 + gate 集 + 末轮): 固定 labels + 固定噪声 ----
        total_steps += steps_this_round
        rnd_label = args.diffusion_pretrain_resume_round + rnd + 1
        if ((rnd_label) in (8, 10, 12, 15, 18) or (rnd + 1) in (1, 3)
                or (rnd + 1) % 5 == 0
                or rnd == args.diffusion_pretrain_rounds - 1):
            generator.eval()
            probe_rng.manual_seed(args.seed + 9173)   # 重置: 同噪声流跨 checkpoint 可比
            # (与 initial noise 的 seed+4242 独立, Bug 7)
            with torch.no_grad():
                raw_probe = generator.sample(
                    labels=probe_labels, device=device,
                    apply_output_bound=False, initial_noise=probe_noise,
                    noise_generator=probe_rng)
            r_probe = torch.linalg.vector_norm(raw_probe, ord=2, dim=1)
            probe = {
                'round': rnd_label,
                'approx_local_steps': total_steps // len(client_data),
                'diag_mse': rec['weighted_aggregated_global'],
                'mse_zero_ratio': round(rec['weighted_aggregated_global']
                                        / max(zero_base, 1e-12), 4),
                'early_mse': rec['agg_bucket_mse']['early'],
                'mid_mse': rec['agg_bucket_mse']['mid'],
                'late_mse': rec['agg_bucket_mse']['late'],
                'raw_std': float(raw_probe.std().item()),
                'raw_median_r': float(r_probe.median().item()),
                'raw_mean_r': float(r_probe.mean().item()),
                'raw_p95_r': float(torch.quantile(r_probe, 0.95).item()),
                'raw_max_r': float(r_probe.max().item()),
                'radius_vs_real': round(float(r_probe.median().item())
                                        / max(real_med, 1e-12), 1),
                'std_vs_real': round(float(raw_probe.std().item())
                                     / max(real_std, 1e-12), 1),
            }
            if rnd_label in (1, 5):
                probe['reverse_traj'] = reverse_norm_trajectory(
                    generator, probe_labels, probe_noise, probe_rng, device)
            if getattr(generator, 'skip_mode', 'none') == 'timestep_scalar':
                with torch.no_grad():
                    ct = torch.exp(generator.skip_log_scale.weight).squeeze(-1)
                T_ct = ct.numel()
                probe['c_stats'] = {
                    'min': float(ct.min().item()),
                    'max': float(ct.max().item()),
                    'mean_early': float(ct[:T_ct // 3].mean().item()),
                    'mean_mid': float(ct[T_ct // 3:2 * T_ct // 3].mean().item()),
                    'mean_late': float(ct[2 * T_ct // 3:].mean().item()),
                }
                if ct.max().item() > 2.0:
                    print(f"  ⚠ [skip-gate] max c(t)={ct.max().item():.3f} > 2, "
                          f"scalar gate 可能开始走新的幅度路径")
                c_extra = (f" c_early={probe['c_stats']['mean_early']:.3f} "
                           f"c_late={probe['c_stats']['mean_late']:.3f}")
            else:
                c_extra = ""
            longitudinal.append(probe)
            print(f"    [LONG] r{rnd_label}: diag_mse={probe['diag_mse']:.4f} "
                  f"(x{probe['mse_zero_ratio']}) raw σ={probe['raw_std']:.2f} "
                  f"median_r={probe['raw_median_r']:.1f} "
                  f"p95_r={probe['raw_p95_r']:.1f} "
                  f"r/real={probe['radius_vs_real']}x "
                  f"σ/real={probe['std_vs_real']}x{c_extra}")
            if args.checkpoint_dir:
                with open(os.path.join(args.checkpoint_dir,
                                       'diffusion_longitudinal_diag.json'),
                          'w') as f:
                    json.dump(longitudinal, f, indent=1)
                # 每个 gate 单独保存 denoiser (R18/R20 等均保留, 不互相覆盖)
                torch.save(generator.state_dict(),
                           os.path.join(args.checkpoint_dir,
                                        f'pretrained_generator_r{rnd_label}.pt'))
    print(f"  [联邦扩散预训练] 完成. 去噪器已学到真实特征分布; "
          f"后续反向采样将以真实分布为先验")
    return generator


# =========================================================================
#  pretrained-manifold 约束: radius bank + 参数锚 + 诊断
#  (堵住"放大伪特征幅度以增大 KL 分歧"的作弊路径)
# =========================================================================

@torch.no_grad()
def build_pretrain_radius_bank(generator, num_classes, device, bank_size=64,
                               rng_seed=None):
    """
    预训练完成后, 按类别从 pretrained DDPM 的 **raw** 采样建 radius bank。
    r_c 分布来自联邦扩散预训练本身 (专利 S3.2 的流形约束实现),
    不从对抗期的 fake batch 估计, 也不需要客户端 μ/σ 上传。
    每类只存 bank_size 个标量, 显存可忽略。返回 [num_classes, bank_size]。

    rng_seed 非 None 时用专用 generator (不消耗全局 RNG), 保证 Stage2 入口
    不影响 classifier 轨迹的 RNG 流 (paired-arm 身份一致性)。
    """
    banks = []
    for c in range(num_classes):
        labels = torch.full((bank_size,), c, dtype=torch.long, device=device)
        if rng_seed is None:
            raw_x = generator.sample(labels=labels, device=device,
                                     apply_output_bound=False)
        else:
            g0 = torch.Generator(device=device).manual_seed(rng_seed + c)
            g1 = torch.Generator(device=device).manual_seed(rng_seed + 1000 + c)
            init_noise = torch.randn(bank_size, generator.feat_dim,
                                     generator=g0, device=device)
            raw_x = generator.sample(labels=labels, device=device,
                                     apply_output_bound=False,
                                     initial_noise=init_noise,
                                     noise_generator=g1)
        radius = torch.linalg.vector_norm(raw_x, ord=2, dim=1)
        lo = torch.quantile(radius, 0.05)
        hi = torch.quantile(radius, 0.95)
        banks.append(radius.clamp(lo, hi))
    return torch.stack(banks).detach()


def project_to_pretrain_radius(raw_x, labels, radius_bank, eps=1e-12):
    """Numerically stable hard manifold projection.

    Algebraically identical to x = r_c * z / ||z||, but implemented through
    O(1)-scaled rows so a large-yet-finite float32 raw trajectory cannot make
    the squared norm overflow to inf and silently collapse the projected feature.
    """
    return _stable_project_to_pretrain_radius(
        raw_x, labels, radius_bank, eps=eps)


def diffusion_prior_anchor_loss(model, ref, eps=1e-8):
    """
    相对 L2-SP: 把生成器参数锚定在联邦预训练结束时的 θ_pre,
    防止对抗精调把预训练学到的 score field / 数据 prior 洗掉。
    L_anchor = mean_k [ ||θ_k - θ_pre,k||² / mean(θ_pre,k²) ] (逐参数相对化)
    """
    terms = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue   # 冻结参数 (如 c(t)) 不进 anchor, 避免恒 0 项
        p0 = ref[name]
        denom = p0.pow(2).mean().detach().clamp_min(eps)
        terms.append((p - p0).pow(2).mean() / denom)
    return torch.stack(terms).mean()


@torch.no_grad()
def compute_theta_drift(model, ref, eps=1e-8):
    """theta_drift = ||θ - θ_pre|| / ||θ_pre|| —— 对抗精调离预训练 prior 多远。"""
    num = 0.0
    den = 0.0
    for name, p in model.named_parameters():
        p0 = ref[name]
        num += (p - p0).pow(2).sum().item()
        den += p0.pow(2).sum().item()
    return (num / (den + eps)) ** 0.5


@torch.no_grad()
def reverse_norm_trajectory(generator, labels, initial_noise, noise_generator,
                            device):
    """
    逐 reverse step 记录 ||x_t|| 分布 (mean/median/p95) —— 定位发散从哪几步开始。
    仅诊断用; 需在调用前重置 noise_generator 种子以保持跨 checkpoint 可比。
    """
    traj = []
    x_t = initial_noise.clone()
    num_steps = generator.num_steps
    for t in reversed(range(num_steps)):
        r = torch.linalg.vector_norm(x_t, ord=2, dim=1)
        traj.append({'t': int(t),
                     'mean': float(r.mean().item()),
                     'median': float(r.median().item()),
                     'p95': float(torch.quantile(r, 0.95).item())})
        x_t = generator._reverse_step(x_t, t, labels, num_steps,
                                      noise_generator=noise_generator)
    return traj


@torch.no_grad()
def pretrain_only_diagnostics(generator, local_models, global_model,
                              num_classes, device, normalized_ckr,
                              fake_nodes=128, knn_k=8,
                              initial_noise=None, noise_generator=None):
    """
    联邦预训练刚结束(对抗更新之前)的健康检查:
    raw 采样分布统计 + 该分布下 teacher/global 的 L_sem / L_dis。
    用于区分两种根因:
      情况A: 预训练后分布健康, 对抗精调把它冲掉  -> radius+anchor 对症
      情况B: 预训练本身没学到位(raw 已严重偏离真实分布) -> 先查调度/采样
    initial_noise/noise_generator: 可选, 传入固定噪声时不消耗全局 RNG
    (Stage2 paired-arm 身份一致性)。
    返回 dict (调用方落盘)。
    """
    labels = torch.arange(num_classes, device=device).repeat_interleave(
        max(1, fake_nodes // num_classes))
    labels = labels[:fake_nodes]
    raw_x = generator.sample(labels=labels, device=device,
                             apply_output_bound=False,
                             initial_noise=initial_noise,
                             noise_generator=noise_generator)
    fake_graph = build_knn_graph(raw_x, k=min(knn_k, raw_x.shape[0] - 1))
    L_sem = compute_generator_semantic_loss(
        fake_graph, labels, local_models, normalized_ckr, num_classes, device)
    L_dis = compute_generator_disagreement_loss(
        fake_graph, labels, local_models, global_model, normalized_ckr,
        num_classes, device, loss_type='kl')
    r = torch.linalg.vector_norm(raw_x, ord=2, dim=1)
    return {
        'raw_mu': float(raw_x.mean().item()),
        'raw_std': float(raw_x.std().item()),
        'raw_radius_mean': float(r.mean().item()),
        'raw_radius_std': float(r.std().item()),
        'L_sem': float(L_sem.item()),
        'L_dis': float(L_dis.item()),
    }


def compute_global_feature_stats(subgraphs, device):
    """
    特征统计特性对齐 (专利 S3.2「生成具备真实统计特性的伪节点特征矩阵」)。

    客户端在本地计算自己特征的一阶/二阶统计量 (μ_k, σ_k²) 并上行 ——
    **只上行 2×F 个统计量, 不上行任何原始特征值**, 与专利权1
    「CKR 客户端本地计算, 仅上传分值」是同一条设计思路 (数据不出域)。

    服务端按节点数加权聚合成全局 (μ, σ), 再把生成器的输出仿射对齐到该统计特性。

    Returns: (g_mu [F], g_std [F])
    """
    tot_n, s1, s2 = 0, None, None
    for sg in subgraphs:
        x = sg.x.to(device).float()
        n = x.shape[0]
        mu = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        if s1 is None:
            s1, s2 = torch.zeros_like(mu), torch.zeros_like(mu)
        s1 += n * mu
        s2 += n * (var + mu ** 2)
        tot_n += n
    g_mu = s1 / max(tot_n, 1)
    g_var = torch.clamp(s2 / max(tot_n, 1) - g_mu ** 2, min=0.0)
    return g_mu, torch.sqrt(g_var)


def match_feature_stats(fake_x, g_mu, g_std, eps=1e-6):
    """
    把生成器输出重标定到真实特征的统计特性 (专利 S3.2「生成具备真实统计特性的
    伪节点特征矩阵」)。

    两个作用:
      1. 让伪特征具备真实特征的尺度, 教师模型不再看到 OOD 输入;
      2. 同时**消除生成器"放大输出幅度以增大分歧损失"的失控动机** ——
         尺度被外部强制固定, 生成器无法再靠膨胀幅度降低 L_G。
         (实测: 无此对齐时 PubMed 伪特征 σ=0.84、真实 σ=0.018, 差 48 倍,
          蒸馏梯度爆到 280、生成器 L_sem 发散到 37.6)

    ⚠️ 实现细节: 采用**全局标量**重标定而非逐维 z-score。
       逐维 z-score 会在该维批内标准差趋近 0 时(生成器输出近似常数)除以 eps,
       **把梯度彻底杀死** —— 实测 L_sem/L_dis/|grad| 全部塌缩到 0, 生成器退化为常数输出。
       全局标量只调整整体幅度、保留节点间相对结构, 梯度保持线性可传。
       缩放系数 detach, 防止生成器反过来把系数当成可优化的作弊通道。
    """
    f_mean = fake_x.mean()
    f_std = fake_x.std().clamp(min=eps)
    scale = (g_std.mean() / f_std).detach()
    return (fake_x - f_mean.detach()) * scale + g_mu.mean()


# =====================================================================
#  S2: 子图-子图跨视图对比学习 (带 RWR 结构缓存)
# =====================================================================

def subgraph_contrastive_step(model, data, aug_edge_index, anchor_nodes,
                               subgraph_size, tau, device,
                               rwr_cache=None, client_id=0, round_idx=0,
                               view1_persistent=True, rwr_seed=0,
                               cache_scope='round'):
    """
    子图-子图跨视图对比学习。
    严格流程:
      1. 原始图 = view_1, 边扰动图 = view_2
      2. 对锚点 i, 在两个视图中分别 RWR 采样局部子图
      3. 两个局部子图独立 clone, 用 center_indices[i] 定位锚点
      4. 锚点局部特征置零 (不修改原始 data.x)
      5. 分别输入共享 GCN.encode() → 隐藏表示 → mean readout
      6. 同一锚点跨视图为正样本, 不同锚点为负样本
      7. InfoNCE 损失 (不访问 data.y)

    rwr_cache 非 None 时:
      - view_1 (原始图) 结构跨 epoch/轮缓存 (view1_persistent=True)
      - view_2 (扰动图) 每轮失效 (key 含 round_idx 与 view_2 edge hash)
      只缓存 (sampled_global_node_ids, center_local_idx), 不缓存任何模型输出。
    """
    num_nodes = data.x.shape[0]
    anchor_list = anchor_nodes.tolist() if torch.is_tensor(anchor_nodes) else anchor_nodes
    batch_size = len(anchor_list)
    if batch_size == 0:
        return torch.tensor(0.0, device=device)

    # round_idx=None 表示 scope=run (不按轮失效; 正确性由 edge hash 保证)
    round_key = round_idx if cache_scope == 'round' else None

    # ---- view_1: 原始图 (结构固定, 可跨轮缓存) ----
    sg1_list, c1_idx = [], []
    for anchor in anchor_list:
        hit = None
        if rwr_cache is not None:
            hit = rwr_cache.get(data.edge_index, anchor, subgraph_size,
                                args.rwr_restart_prob, rwr_seed,
                                client_id=client_id, view=1, round_idx=round_key)
        if hit is not None:
            sg1_list.append(hit[0]); c1_idx.append(hit[1])
        else:
            sg1, c1 = rwr_subgraph_sampling(
                data.edge_index, num_nodes, [anchor],
                subgraph_size, restart_prob=args.rwr_restart_prob, seed=rwr_seed)
            sg1_list.append(sg1[0]); c1_idx.append(c1[0])
            if rwr_cache is not None:
                rwr_cache.put(data.edge_index, anchor, subgraph_size,
                              args.rwr_restart_prob, rwr_seed,
                              (sg1[0], c1[0]), client_id=client_id,
                              view=1, round_idx=round_key)

    # ---- view_2: 边扰动图 (key 含 round/edge hash, 新视图自动失效) ----
    sg2_list, c2_idx = [], []
    for anchor in anchor_list:
        hit = None
        if rwr_cache is not None:
            hit = rwr_cache.get(aug_edge_index, anchor, subgraph_size,
                                args.rwr_restart_prob, rwr_seed,
                                client_id=client_id, view=2, round_idx=round_key)
        if hit is not None:
            sg2_list.append(hit[0]); c2_idx.append(hit[1])
        else:
            sg2, c2 = rwr_subgraph_sampling(
                aug_edge_index, num_nodes, [anchor],
                subgraph_size, restart_prob=args.rwr_restart_prob, seed=rwr_seed)
            sg2_list.append(sg2[0]); c2_idx.append(c2[0])
            if rwr_cache is not None:
                rwr_cache.put(aug_edge_index, anchor, subgraph_size,
                              args.rwr_restart_prob, rwr_seed,
                              (sg2[0], c2[0]), client_id=client_id,
                              view=2, round_idx=round_key)

    # 批量编码: 所有锚点子图拼成一张 flat 图一次前向 (每个子图是独立连通分量,
    # GCN 消息传递不跨分量; 分段 mean readout 等价于逐子图 mean)。仅 RNG 消费
    # 顺序不同 (dropout 掩码序列), 训练语义不变。
    z1 = _batched_subgraph_encode(
        model, data.x, data.edge_index, sg1_list, c1_idx, anchor_list, device)
    z2 = _batched_subgraph_encode(
        model, data.x, aug_edge_index, sg2_list, c2_idx, anchor_list, device)
    # InfoNCE: 正样本 = 同一锚点跨视图, 负样本 = 不同锚点
    loss = subgraph_contrastive_loss(z1, z2, tau=tau)
    return loss


def _batched_subgraph_encode(model, x_full, edge_index, sg_list, center_idx_list,
                             anchor_list, device):
    """把一批局部子图拼成单张 flat 图, 一次 GCN encode + 分段均值 readout。"""
    batch_size = len(sg_list)
    sizes = [len(s) for s in sg_list]
    flat_nodes = torch.cat([torch.tensor(s, device=device) for s in sg_list])
    offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
    offsets[1:] = torch.tensor(sizes[:-1], device=device).cumsum(0)
    seg = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        torch.tensor(sizes, device=device))

    # 锚点定位 + 特征置零 (不修改原始 x_full)
    x_flat = x_full[flat_nodes].clone()
    center_pos = offsets + torch.tensor(center_idx_list, device=device)
    for i, (pos, anchor_global) in enumerate(zip(center_pos.tolist(),
                                                 anchor_list)):
        assert sg_list[i][center_idx_list[i]] == anchor_global, \
            f"center_idx={center_idx_list[i]} points to {sg_list[i][center_idx_list[i]]}, " \
            f"expected anchor={anchor_global}"
    x_flat[center_pos, :] = 0.0

    # 每子图局部边 + 段偏移, 拼接为 flat 边索引
    edge_parts = []
    for i in range(batch_size):
        ei = _extract_subgraph_edges(edge_index, sg_list[i], device)
        edge_parts.append(ei + offsets[i])
    ei_flat = torch.cat(edge_parts, dim=1)

    h = model.encode(x_flat, ei_flat)
    z = torch.zeros(batch_size, h.shape[1], device=device)
    z.scatter_add_(0, seg.unsqueeze(1).expand(-1, h.shape[1]), h)
    z = z / torch.bincount(seg, minlength=batch_size).float().unsqueeze(1)
    return z


def _extract_subgraph_edges(edge_index, node_set, device):
    # 向量化实现 (等价于原 dict + .item() 循环): 子图节点数 S 很小,
    # S x E 的整型比较矩阵直接在 GPU 上完成全局->局部索引映射。
    node_set_t = torch.tensor(node_set, device=device)
    rows, cols = edge_index
    mask = torch.isin(rows, node_set_t) & torch.isin(cols, node_set_t)
    sub_rows = rows[mask]
    sub_cols = cols[mask]
    if sub_rows.numel() == 0:
        return torch.tensor([[0], [0]], device=device, dtype=torch.long)
    rr = (node_set_t.unsqueeze(1) == sub_rows.unsqueeze(0)).long().argmax(dim=0)
    cc = (node_set_t.unsqueeze(1) == sub_cols.unsqueeze(0)).long().argmax(dim=0)
    return torch.stack([rr, cc], dim=0)


# =====================================================================
#  S4: 生成器与蒸馏损失函数 (服务端)
#  注意: 这些函数不接收客户端原始 Data (隐私边界)
# =====================================================================

def compute_generator_semantic_loss(fake_graph, fake_labels, local_models,
                                     normalized_ckr, num_classes, device):
    """
    生成器语义损失: 教师模型在伪图上预测与伪标签一致。
    参数冻结, 但保留对 fake_x 的梯度。
    生成器目标: 最小化。
    """
    loss = torch.tensor(0.0, device=device)
    each_class = {c: (fake_labels == c) for c in range(num_classes)}

    # 教师前向只依赖 (model_k, fake_graph), 与类别 c 无关 -> 每个教师只算一次
    all_logits = [model_k(fake_graph) for model_k in local_models]  # 保留梯度

    for c in range(num_classes):
        idx_c = each_class[c]
        if idx_c.sum() == 0:
            continue
        for k_idx, logits in enumerate(all_logits):
            w = normalized_ckr[k_idx, c]
            if w < 1e-8:
                continue
            loss += w * F.cross_entropy(logits[idx_c], fake_labels[idx_c])
    return loss


def compute_generator_disagreement_loss(fake_graph, fake_labels,
                                          local_models, global_model,
                                          normalized_ckr, num_classes, device,
                                          loss_type='l1'):
    """
    生成器分歧损失: 可靠教师与全局模型的预测分歧。
    参数冻结, 保留对 fake_x 的梯度。
    生成器目标: 最大化 (总损失中使用负号)。

    loss_type='kl' (默认, 本方法/专利权6+S4.2 的设计):
        按 FedTAD Eq.(10) 对预测分布做 CKR 加权 KL(global‖local)。
    loss_type='l1' (原版 FedTAD 做法, 仅供消融):
        Σ_c Σ_k ckr[k,c] · mean|global_pred[c] − local_pred[c].detach()|, 作用在原始输出上
        (references/FedTAD/train_fedtad.py:342-345)
    """
    loss = torch.tensor(0.0, device=device)
    each_class = {c: (fake_labels == c) for c in range(num_classes)}
    global_logits = global_model(fake_graph)  # 保留梯度
    # 教师前向与类别 c 无关 -> 每个教师只算一次
    teacher_logits = [model_k(fake_graph) for model_k in local_models]

    for c in range(num_classes):
        idx_c = each_class[c]
        if idx_c.sum() == 0:
            continue
        for k_idx, t_logits in enumerate(teacher_logits):
            w = normalized_ckr[k_idx, c]
            if w < 1e-8:
                continue
            if loss_type == 'l1':
                loss += w * torch.abs(
                    global_logits[idx_c] - t_logits[idx_c].detach()).mean()
            else:
                # FedTAD Eq.(10) / patent S4.2: KL(global || local).
                # Keep both model forward graphs w.r.t. fake_x so generator
                # receives the correct adversarial input gradient; model
                # parameters themselves are frozen by the caller.
                loss += w * categorical_kl_from_logits(
                    global_logits[idx_c], t_logits[idx_c],
                    temperature=1.0, reduction='batchmean')
    return loss


def compute_student_distillation_loss(fake_graph, fake_labels,
                                       local_models, global_model,
                                       normalized_ckr, num_classes, device,
                                       temperature=1.0, loss_type='kl'):
    """
    全局模型蒸馏损失: 学生(全局)向教师(本地)对齐。
    教师概率 detach, fake_graph.x detach。
    仅更新全局模型。

    loss_type='kl' (默认, 本方法/专利权6+S4.2 的设计):
        以 CKR 为权重的 KL(global‖local), 对齐全局模型与本地模型的**预测分布**。
    loss_type='l1' (原版 FedTAD 做法, 仅供消融):
        Σ_c Σ_k ckr[k,c] · mean|global_pred[c] − local_pred[c].detach()|, 作用在原始输出上。
    """
    # 确保 x 已 detach
    if fake_graph.x.requires_grad:
        fake_graph = Data(x=fake_graph.x.detach(), edge_index=fake_graph.edge_index)

    loss = torch.tensor(0.0, device=device)
    each_class = {c: (fake_labels == c) for c in range(num_classes)}
    global_logits = global_model(fake_graph)

    # 教师前向与类别 c 无关 -> 每个教师只算一次; 教师侧 no_grad。
    # KL 方向按 FedTAD Eq.(10) / patent S4.2 固定为 global || local。
    with torch.no_grad():
        teacher_logits = [model_k(fake_graph) for model_k in local_models]

    for c in range(num_classes):
        idx_c = each_class[c]
        if idx_c.sum() == 0:
            continue
        if loss_type == 'l1':
            for k_idx, t_logits in enumerate(teacher_logits):
                w = normalized_ckr[k_idx, c]
                if w < 1e-8:
                    continue
                loss += w * torch.abs(
                    global_logits[idx_c] - t_logits[idx_c]).mean()
        else:
            for k_idx, t_logits in enumerate(teacher_logits):
                w = normalized_ckr[k_idx, c]
                if w < 1e-8:
                    continue
                kl = categorical_kl_from_logits(
                    global_logits[idx_c], t_logits[idx_c],
                    temperature=temperature, reduction='batchmean')
                loss += w * kl

    if loss_type == 'kl' and temperature != 1.0:
        loss = loss * (temperature ** 2)
    return loss


# =====================================================================
#  多分类 / 二分类 客户端与全局评估
# =====================================================================

def evaluate_client(model, data, task_mode, eval_mask=None):
    """
    返回按任务模式对应的指标 dict。
    eval_mask 指定评估子集 (val_idx/test_idx); None 表示全图。
    注意: GCN 前向仍需全图 (消息传递), 仅指标计算限于 eval_mask。
    """
    model.eval()
    with torch.no_grad():
        logits = model.forward(data)
        if eval_mask is not None:
            logits = logits[eval_mask]
            y = data.y[eval_mask]
        else:
            y = data.y
        if task_mode == 'multiclass':
            return multiclass_metrics(logits, y)
        else:
            return binary_anomaly_metrics(logits, y)


def evaluate_global(global_model, subgraphs, task_mode, split='val', device=None):
    """
    全局模型在指定划分 (val/test) 上的聚合评估。

    multiclass: 按客户端节点数加权的 accuracy / macro_f1
    anomaly_binary: pooled / macro / weighted / worst client AUC 与 PR-AUC
    """
    total_nodes = sum(sg.x.shape[0] for sg in subgraphs)

    if task_mode == 'multiclass':
        acc = 0.0
        mf1 = 0.0
        for ci, sg in enumerate(subgraphs):
            mask = sg.val_idx if split == 'val' else sg.test_idx
            em = evaluate_client(global_model, sg, task_mode, eval_mask=mask)
            w = sg.x.shape[0] / total_nodes
            acc += w * em['accuracy']
            mf1 += w * em['macro_f1']
        return {'accuracy': acc, 'macro_f1': mf1,
                'pooled_pr_auc': float('nan'), 'weighted_client_auc': float('nan')}

    # ---- anomaly_binary ----
    import sklearn.metrics as skm
    all_labels, all_probs = [], []
    client_aucs, client_pr_aucs, client_weights = [], [], []
    for ci, sg in enumerate(subgraphs):
        mask = sg.val_idx if split == 'val' else sg.test_idx
        global_model.eval()
        with torch.no_grad():
            logits = global_model.forward(sg)
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_np = sg.y.cpu().numpy()
        m = mask.cpu().numpy()
        all_labels.extend(y_np[m].tolist())
        all_probs.extend(prob[m].tolist())
        client_weights.append(int(m.sum()))
        unique = set(int(x) for x in y_np[m])
        if m.sum() > 0 and len(unique) >= 2:
            ca = skm.roc_auc_score(y_np[m], prob[m]) * 100.0
            pa = skm.average_precision_score(y_np[m], prob[m]) * 100.0
        else:
            ca, pa = float('nan'), float('nan')
        client_aucs.append(ca)
        client_pr_aucs.append(pa)

    pooled_auc = pooled_pr_auc = float('nan')
    if len(set(all_labels)) >= 2:
        try:
            pooled_auc = skm.roc_auc_score(all_labels, all_probs) * 100.0
            pooled_pr_auc = skm.average_precision_score(all_labels, all_probs) * 100.0
        except Exception:
            pass

    valid_aucs = [v for v in client_aucs if not math.isnan(v)]
    valid_pr_aucs = [v for v in client_pr_aucs if not math.isnan(v)]
    macro_client_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')
    macro_client_pr_auc = sum(valid_pr_aucs) / len(valid_pr_aucs) if valid_pr_aucs else float('nan')

    weighted_client_auc = float('nan')
    valid_idx = [i for i, v in enumerate(client_aucs) if not math.isnan(v)]
    if valid_idx:
        w_sum = sum(client_weights[i] for i in valid_idx)
        if w_sum > 0:
            weighted_client_auc = sum(client_aucs[i] * client_weights[i]
                                      for i in valid_idx) / w_sum
    worst_client_auc = min(valid_aucs) if valid_aucs else float('nan')

    return {
        'pooled_auc': pooled_auc,
        'pooled_pr_auc': pooled_pr_auc,
        'macro_client_auc': macro_client_auc,
        'macro_client_pr_auc': macro_client_pr_auc,
        'weighted_client_auc': weighted_client_auc,
        'worst_client_auc': worst_client_auc,
        'accuracy': float('nan'),
        'macro_f1': float('nan'),
    }


# =====================================================================
#  主流程
# =====================================================================

def main():
    if getattr(args, 'formal_campaign_guard', False):
        validate_formal_campaign(args)
    if getattr(args, 'formal_multiclass_protocol_guard', False):
        validate_formal_protocol(args)
    validate_run_args(args)   # Bug 3/4/9: 互斥 CLI 组合 fail-fast
    if getattr(args, 'deterministic_torch', False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("  [Deterministic] torch 确定性模式开启")
    seed_everything(seed=args.model_seed)
    t_total0 = time.time()

    print("=" * 60)
    print(f"任务模式: {args.task_mode} | CKR mode: {args.ckr_mode}")
    print(f"[Seeds] partition={args.partition_seed} allocation={args.allocation_seed} "
          f"split={args.split_seed} model={args.model_seed} "
          f"bootstrap={args.bootstrap_seed}")
    if args.task_mode == 'anomaly_binary':
        print(f"  正常类别: {args.normal_classes}")
        print(f"  异常类别: {args.anomaly_classes}")
    print("=" * 60)

    # ---- Core method 声明 (如实反映运行时配置) ----
    print("\nCore method:")
    print(f"  weighted CE: {'enabled' if args.use_weighted_ce else 'disabled'}")
    print(f"  subgraph cross-view contrastive learning: "
          f"{'enabled (' + args.contrastive_mode + ')' if args.contrastive_mode != 'none' else 'disabled'}")
    print(f"  static topology CKR: {args.ckr_mode}")
    print(f"  FedAvg: {args.federated_mode}")
    print(f"  teacher-guided pseudo-graph distillation: "
          f"{'enabled (' + args.distill_weighting + ')' if args.distill_weighting != 'none' else 'disabled'}")
    print("  --")
    print(f"  当前任务模式: {args.task_mode} (标签映射: 正常={args.normal_classes}, "
          f"异常={args.anomaly_classes})")
    print(f"  客户端数量: {args.num_clients} | 联邦轮次: {args.num_rounds} | "
          f"本地 epochs: {args.num_epochs if args.local_epochs <= 0 else args.local_epochs}")
    print(f"  CKR 模式: {args.ckr_mode} | 对比学习模式: {args.contrastive_mode} | "
          f"锚点作用域: {args.contrastive_anchor_scope}")
    print(f"  伪节点类别策略: {args.fake_class_strategy} | 伪节点数: {args.fake_nodes} | "
          f"KNN k: {args.knn_k}")
    print(f"  dynamic CKR: {'enabled' if args.distill_weighting == 'dynamic_ckr' or args.ckr_mode != 'static_topology' else 'disabled'}")
    print(f"  fairness 扩展: {'enabled (' + args.fairness_mode + ')' if args.fairness_mode != 'none' else 'disabled'}")
    print(f"  public pretraining: {'enabled' if args.generator_init == 'proxy_pretrained' else 'disabled'} "
          f"(generator_init={args.generator_init})")
    print("=" * 60)

    # ---- 平台集成: 任务开始事件 (结构化配置快照) ----
    _emit_event(
        args, 'task_started', stage='initializing',
        task_mode=args.task_mode,
        normal_classes=args.normal_classes,
        anomaly_classes=args.anomaly_classes,
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        ckr_mode=args.ckr_mode,
        distill_weighting=args.distill_weighting,
        contrastive_mode=args.contrastive_mode,
        contrastive_anchor_scope=args.contrastive_anchor_scope,
        fake_class_strategy=args.fake_class_strategy,
        fake_node_count=args.fake_nodes,
        knn_k=args.knn_k,
        fairness_mode=args.fairness_mode,
        generator_init=args.generator_init,
        model_seed=args.model_seed,
        partition_seed=args.partition_seed,
        split_seed=args.split_seed,
    )

    try:
        dataset = load_dataset(args)
    except Exception as e:
        if type(e).__name__ == 'DatasetBlockedError':
            print(f"[BLOCKED] external resource: {e}")
            sys.exit(3)
        raise
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device('cpu')
    print(f"[Device] {device}")
    loss_fn = nn.CrossEntropyLoss()

    # ---- 数据集 manifest (离线导入校验) ----
    if args.dataset_manifest or args.dataset_checksum:
        _write_dataset_manifest(dataset, args)

    # ---- 加载子图 (frozen split 优先) ----
    frozen_artifact = None
    if args.frozen_split:
        from util.split_artifact import load_artifact, build_subgraphs_from_artifact
        frozen_artifact = load_artifact(args.frozen_split)
        sc = frozen_artifact['split_config']
        assert sc['dataset'] == args.dataset, \
            f"frozen split dataset 不匹配: {sc['dataset']} != {args.dataset}"
        subgraphs = build_subgraphs_from_artifact(
            dataset, frozen_artifact, dataset.num_classes, device)
        print(f"[Frozen Split] 加载 {args.frozen_split} "
              f"(hash={frozen_artifact['data_identity_hash']})")
    else:
        subgraphs = [dataset.subgraphs[ci].to(device)
                     for ci in range(args.num_clients)]

    # ---- 标签映射 (anomaly_binary) ----
    num_classes = dataset.num_classes
    mapping_info = None
    if args.task_mode == 'anomaly_binary':
        assert args.normal_classes and args.anomaly_classes, \
            "anomaly_binary 模式需要 --normal_classes 和 --anomaly_classes"
        if frozen_artifact is not None:
            # frozen: 使用 artifact 记录的映射
            mp = frozen_artifact['mapping']
            for sg in subgraphs:
                apply_label_mapping_to_data(
                    sg, ','.join(str(c) for c in mp['normal_classes']),
                    ','.join(str(c) for c in mp['anomaly_classes']))
            num_classes = 2
            n_norm = sum(int((sg.y == 0).sum()) for sg in subgraphs)
            n_anom = sum(int((sg.y == 1).sum()) for sg in subgraphs)
            mapping_info = {'mapping': mp['mapping'],
                            'normal_classes': mp['normal_classes'],
                            'anomaly_classes': mp['anomaly_classes'],
                            'counts_after': {0: n_norm, 1: n_anom},
                            'anomaly_ratio': n_anom / (n_norm + n_anom + 1e-8)}
            print(f"[Frozen Label Mapping] normal={mp['normal_classes']} "
                  f"anomaly={mp['anomaly_classes']} "
                  f"(normal={n_norm}, anomaly={n_anom})")
        else:
            subgraphs, num_classes, mapping_info = apply_label_mapping(
                subgraphs, dataset, args.normal_classes, args.anomaly_classes)
            print(f"[Label Mapping] 原始 {len(mapping_info['counts_before'])} 类 "
                  f"→ 2 类 (正常={mapping_info['normal_classes']}, "
                  f"异常={mapping_info['anomaly_classes']})")
            print(f"  映射后分布: 正常={mapping_info['counts_after'][0]}, "
                  f"异常={mapping_info['counts_after'][1]}")
        # 异常少数类检查
        anomaly_ratio = mapping_info.get("anomaly_ratio", 1.0)
        if anomaly_ratio >= 0.5:
            msg = (f"Anomaly ratio = {anomaly_ratio:.4f} >= 0.5, "
                   f"异常不是少数类. 使用 --allow_anomaly_majority 可绕过.")
            if not getattr(args, "allow_anomaly_majority", False):
                raise ValueError(msg)
            else:
                print("  [Anomaly] --allow_anomaly_majority 已启用, 继续训练")
    else:
        print(f"[Task] 多分类, num_classes={num_classes}")

    # ---- 按类别分层重划分 (seed 控制; frozen split 跳过) ----
    support_infeasible = []
    boost_moved_total = {}
    if frozen_artifact is not None:
        print("  [Resplit] frozen split: 使用 artifact 固定划分")
    elif args.resplit_stratified:
        train_r, val_r = _split_ratios(args.dataset)
        print(f"\n[Resplit stratified] train={train_r} val={val_r} "
              f"mode={args.split_support_mode} (split_seed={args.split_seed}+client_id, "
              f"按{'二分类' if args.task_mode == 'anomaly_binary' else '原'}标签分层)")
        for ci in range(args.num_clients):
            sg = subgraphs[ci]
            before = {c: int((sg.y[sg.train_idx] == c).sum()) for c in range(num_classes)}
            if args.split_support_mode == 'anomaly_holdout_boost':
                assert args.task_mode == 'anomaly_binary', \
                    "anomaly_holdout_boost 仅支持 anomaly_binary"
                (train_mask, val_mask, test_mask, sinfo,
                 moved, infeas) = anomaly_holdout_boost_split(
                    sg, train_r, val_r, args.anomaly_val_ratio,
                    seed=args.split_seed + ci,
                    min_train_support=args.min_train_support_per_class,
                    allow_infeasible=args.allow_support_infeasible)
                support_infeasible.extend(infeas)
                for k, v in moved.items():
                    boost_moved_total.setdefault(k, {'moved_from_train': 0,
                                                     'boosted_val': 0})
                    boost_moved_total[k]['moved_from_train'] += v['moved_from_train']
                    boost_moved_total[k]['boosted_val'] += v['boosted_val']
            elif args.split_support_mode == 'stratified_local':
                (train_mask, val_mask, test_mask, sinfo,
                 infeas) = stratified_split_with_support(
                    sg, num_classes, train_r, val_r,
                    min_train_support=args.min_train_support_per_class,
                    min_val_support=args.min_val_support_per_class,
                    seed=args.split_seed + ci,
                    allow_infeasible=args.allow_support_infeasible)
                support_infeasible.extend(infeas)
            else:
                # natural / anomaly_enriched_partition (划分在 load 时已增强)
                train_mask, val_mask, test_mask, sinfo = stratified_split(
                    sg, num_classes, train_r, val_r, seed=args.split_seed + ci)
            sg.train_idx = train_mask
            sg.val_idx = val_mask
            sg.test_idx = test_mask
            after = {c: int((sg.y[sg.train_idx] == c).sum()) for c in range(num_classes)}
            print(f"    client {ci}: 重划分前 train={before}, 重划分后 train={after}")
            assert not (sg.train_idx & sg.val_idx).any() and \
                   not (sg.train_idx & sg.test_idx).any() and \
                   not (sg.val_idx & sg.test_idx).any(), \
                f"client {ci} 重划分后划分重叠!"
        if support_infeasible:
            print(f"  [Support Infeasible] {len(support_infeasible)} 条不可满足记录:")
            for line in support_infeasible[:20]:
                print(f"    - {line}")
        if boost_moved_total:
            print(f"  [Anomaly Holdout Boost] 从 train 移至 val 的异常节点: "
                  f"{boost_moved_total}")
        print(f"  [Resplit] val/test 与 train 无重叠, 划分完成")
    else:
        print("  [Resplit] 已禁用, 使用磁盘缓存的固定划分")

    # ---- data_identity_hash + frozen artifact 构建 ----
    from util.split_artifact import save_split_artifact, data_identity_hash
    if frozen_artifact is not None:
        # frozen: 数据身份由 artifact 记录的 support 模式与参数定义
        args.split_support_mode = frozen_artifact['split_config']['split_support_mode']
        support_params = frozen_artifact['split_config']['support_params']
    else:
        support_params = {
            'holdout_ratio': args.reliability_holdout_ratio,
            'min_support': args.reliability_min_support,
            'min_train_support': args.min_train_support_per_class,
            'min_val_support': args.min_val_support_per_class,
            'anomaly_val_ratio': args.anomaly_val_ratio,
            'allow_support_infeasible': args.allow_support_infeasible,
            'anomaly_partition_target': args.anomaly_partition_target,
        }
    node_dict = (frozen_artifact['node_dict'] if frozen_artifact is not None
                 else getattr(dataset, 'node_dict',
                              {ci: [] for ci in range(args.num_clients)}))
    data_id_hash = data_identity_hash(
        args.dataset, args.task_mode, num_classes, subgraphs,
        mapping_info, node_dict, args.split_support_mode, support_params)
    print(f"[Data Identity] hash={data_id_hash}")

    if args.build_split_artifact:
        save_split_artifact(args.build_split_artifact, args, dataset, subgraphs,
                            num_classes, mapping_info, node_dict, support_params)
        print(f"[Split Artifact] 保存 {args.build_split_artifact} 并退出")
        sys.exit(0)
    if frozen_artifact is not None:
        assert frozen_artifact['data_identity_hash'] == data_id_hash, \
            (f"frozen split 身份哈希不匹配: "
             f"{frozen_artifact['data_identity_hash']} != {data_id_hash}")
        print("  [Frozen Split] data_identity_hash 校验通过")

    # ---- 初始化模型 (映射与划分完成之后) ----
    feat_dim = subgraphs[0].x.shape[1]
    local_models = [
        GCN(feat_dim=feat_dim, hid_dim=args.hid_dim,
            out_dim=num_classes, dropout=args.dropout).to(device)
        for _ in range(args.num_clients)
    ]
    local_optimizers = [
        Adam(local_models[ci].parameters(), lr=args.lr,
             weight_decay=args.weight_decay)
        for ci in range(args.num_clients)
    ]

    global_model = GCN(feat_dim=feat_dim, hid_dim=args.hid_dim,
                       out_dim=num_classes, dropout=args.dropout).to(device)

    # ---- 生成器 ----
    distill_feat_dim = feat_dim
    diffusion_skip_mode = args.diffusion_skip_mode
    if diffusion_skip_mode == 'none' and args.diffusion_residual_skip:
        diffusion_skip_mode = 'fixed'   # 兼容旧 flag
    generator = ConditionalDiffusionGenerator(
        feat_dim=distill_feat_dim,
        num_classes=num_classes,
        hidden_dim=args.diffusion_hidden,
        num_steps=args.diffusion_steps,
        beta_start=args.diffusion_beta_start,
        beta_end=args.diffusion_beta_end,
        output_bound=args.generator_output_bound,
        skip_mode=diffusion_skip_mode,
        use_posterior_variance=args.use_posterior_variance,
    ).to(device)
    # 生成器不适用普通 weight decay (θ→0 会把预训练 prior 也拉向 0);
    # prior 保持由 --lambda_diffusion_anchor 的 L2-SP (θ→θ_pre) 显式负责。
    gen_optimizer = Adam(generator.parameters(), lr=args.generator_lr,
                         weight_decay=0.0)
    if args.paired_rng_diag:
        print(f"  [PairedRNG] P0 (模型初始化后): rng={_rng_fingerprint()} "
              f"classifier={_model_param_hash(global_model)}")
    global_optimizer = Adam(global_model.parameters(), lr=args.distill_lr,
                            weight_decay=args.weight_decay)

    # ---- 生成器初始化: 可选公开代理数据预训练 ----
    if args.generator_init == 'proxy_pretrained':
        assert args.proxy_checkpoint and os.path.exists(args.proxy_checkpoint), \
            f"--generator_init proxy_pretrained 需要存在的 --proxy_checkpoint: " \
            f"{args.proxy_checkpoint}"
        try:
            from pretrain_diffusion import load_proxy_pretrained_generator
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "--generator_init proxy_pretrained 需要 pretrain_diffusion.py；"
                "当前包缺少该可选模块。正式 Stage1→Stage2 主线不依赖它。") from e
        ckpt_meta, skipped = load_proxy_pretrained_generator(
            args.proxy_checkpoint, generator, device)
        print(f"[Generator Init] 加载代理 DDPM 预训练权重: "
              f"{args.proxy_checkpoint} (mode={ckpt_meta['meta']['mode']}, "
              f"跳过参数: {skipped or '无'})")
    else:
        print("[Generator Init] scratch: 随机初始化, "
              "教师引导扩散式生成器 (public pretraining 关闭)")

    # ---- 初始化身份哈希 (严格配对验证: 同 model_seed -> 同初始化) ----
    from util.split_artifact import initialization_hash
    init_hash = initialization_hash(
        args.model_seed, feat_dim, num_classes, args.hid_dim, args.dropout,
        args.num_clients)
    print(f"[Initialization] hash={init_hash} (model_seed={args.model_seed})")
    if args.initialization_json:
        os.makedirs(os.path.dirname(args.initialization_json) or '.', exist_ok=True)
        with open(args.initialization_json, 'w') as f:
            json.dump({'initialization_hash': init_hash,
                       'model_seed': int(args.model_seed),
                       'feat_dim': int(feat_dim),
                       'num_classes': int(num_classes),
                       'hid_dim': int(args.hid_dim),
                       'dropout': float(args.dropout)}, f, indent=1)
    if args.data_identity_json:
        os.makedirs(os.path.dirname(args.data_identity_json) or '.', exist_ok=True)
        with open(args.data_identity_json, 'w') as f:
            json.dump({'data_identity_hash': data_id_hash,
                       'dataset': args.dataset,
                       'task_mode': args.task_mode,
                       'partition_seed': int(args.partition_seed),
                       'allocation_seed': int(args.allocation_seed),
                       'split_seed': int(args.split_seed),
                       'split_support_mode': args.split_support_mode,
                       'support_params': support_params}, f, indent=1)

    # =================================================================
    #  S1: CKR (静态拓扑先验) + 动态 CKR 跟踪器
    # =================================================================
    print(f"\n[S1] 类别知识可靠性 (CKR mode = {args.ckr_mode})")
    ckr_raw = compute_ckr(subgraphs, num_classes, args, device,
                          data_identity=data_id_hash)
    # performance_only 为 dynamic_only 的规格别名
    ckr_mode_internal = 'dynamic_only' if args.ckr_mode == 'performance_only' else args.ckr_mode
    tracker = DynamicCKRTracker(
        num_clients=args.num_clients,
        num_classes=num_classes,
        mode=ckr_mode_internal,
        static_scaling=args.static_ckr_scaling,
        alpha=args.dynamic_ckr_alpha,
        ema_decay=args.dynamic_ckr_ema_decay,
        metric=args.dynamic_ckr_metric,
        min_support=args.reliability_min_support,
        log_dir=args.log_dir,
    )
    tracker.initialize(ckr_raw)
    normalized_ckr = tracker.get_server_weights()
    print(f"  CKR 形状: [{args.num_clients}, {num_classes}]")
    if args.ckr_mode != 'static_topology':
        print(f"  [DynamicCKR] eval_source={args.dynamic_ckr_eval_source} "
              f"alpha={args.dynamic_ckr_alpha} ema_decay={args.dynamic_ckr_ema_decay} "
              f"metric={args.dynamic_ckr_metric} min_support={args.reliability_min_support}")

    # ---- fit / reliability 划分 (仅从 train_idx) ----
    print(f"\n[Reliability Holdout] ratio={args.reliability_holdout_ratio} "
          f"min_support={args.reliability_min_support} seed={args.reliability_split_seed}")
    for ci in range(args.num_clients):
        sg = subgraphs[ci]
        fit_idx, rel_idx, sinfo = stratified_reliability_split(
            sg, sg.train_idx, num_classes,
            holdout_ratio=args.reliability_holdout_ratio,
            min_support=args.reliability_min_support,
            split_seed=args.reliability_split_seed + ci)
        sg.fit_idx = fit_idx
        sg.reliability_idx = rel_idx
        rep = split_report(fit_idx, rel_idx, sg.y, num_classes,
                           args.reliability_min_support)
        for c in range(num_classes):
            r = rep[c]
            print(f"  client {ci} class {c}: fit_support={r['fit']} "
                  f"reliability_support={r['reliability']} "
                  f"dynamic_available={r['available']}"
                  + (f" fallback_reason={r['fallback_reason']}" if not r['available'] else ''))

    # 动态 CKR 评估集警告
    if args.dynamic_ckr_eval_source == 'validation':
        print("WARNING: validation labels are being used to update CKR "
              "and are no longer a clean model-selection set.")

    # ---- 加权 CE 权重 (基于 fit_idx, 不用 val/test) ----
    class_weights = []
    if args.use_weighted_ce:
        for ci in range(args.num_clients):
            train_labels = subgraphs[ci].y[subgraphs[ci].fit_idx]
            w = compute_class_weights(
                train_labels, num_classes,
                args.class_weight_method, args.beta)
            class_weights.append(w.to(device))
        print(f"[S2] 加权 CE (fit_idx): method={args.class_weight_method}")

    # =================================================================
    #  Louvain 客户端统计 + 零异常审计 (anomaly_binary)
    # =================================================================
    print("\n[Data] 每客户端统计:")
    for ci in range(args.num_clients):
        sg = subgraphs[ci]
        nc = sg.x.shape[0]
        ec = sg.edge_index.shape[1]
        deg = ec / nc
        y_counts = [int((sg.y == c).sum()) for c in range(num_classes)]
        train_c = [int((sg.y[sg.train_idx] == c).sum()) for c in range(num_classes)]
        val_c = [int((sg.y[sg.val_idx] == c).sum()) for c in range(num_classes)]
        test_c = [int((sg.y[sg.test_idx] == c).sum()) for c in range(num_classes)]
        print(f"  client {ci}: nodes={nc}, edges={ec}, avg_deg={deg:.2f}, "
              f"y_dist={y_counts}")
        if args.task_mode == 'anomaly_binary':
            for split_name in ('fit', 'reliability', 'val', 'test'):
                has_anomaly = (sg.y[sg[f'{split_name}_idx']] == 1).sum().item()
                if has_anomaly == 0:
                    print(f"    ⚠ {split_name}_idx 零异常 (该划分无异常样本)")
            if (sg.y[sg.fit_idx] == 1).sum().item() == 0:
                print(f"    → fit 零异常: 异常类加权 CE 权重为 0, 无异常监督梯度; "
                      f"仍进行子图跨视图对比, 异常知识依赖联邦聚合与蒸馏")

    # 异质性汇总
    node_counts = [subgraphs[ci].x.shape[0] for ci in range(args.num_clients)]
    edge_counts = [subgraphs[ci].edge_index.shape[1] for ci in range(args.num_clients)]
    print(f"  [Heterogeneity] nodes: mean={torch.tensor(node_counts).float().mean():.1f}, "
          f"var={torch.tensor(node_counts).float().var():.1f}")
    print(f"  [Heterogeneity] edges: mean={torch.tensor(edge_counts).float().mean():.1f}, "
          f"var={torch.tensor(edge_counts).float().var():.1f}")

    # ---- 正确性断言 (泄漏防护) ----
    run_integrity_assertions(subgraphs, num_classes, args)

    # ---- 划分报告 (split_report_json): fit/rel/val/test 分布 + 零异常客户端 ----
    split_report_data = {
        'seed': args.seed,
        'config_hash': config_hash(args),
        'task_mode': args.task_mode,
        'num_clients': args.num_clients,
        'resplit_stratified': bool(args.resplit_stratified),
        'num_classes': int(num_classes),
        'clients': {},
    }
    for ci in range(args.num_clients):
        sg = subgraphs[ci]
        sets = {}
        for split_name in ('fit', 'reliability', 'val', 'test'):
            mask = sg[f'{split_name}_idx']
            sets[split_name] = {
                'total': int(mask.sum()),
                'counts': [int((sg.y[mask] == c).sum()) for c in range(num_classes)],
            }
            if args.task_mode == 'anomaly_binary':
                sets[split_name]['normal'] = int((sg.y[mask] == 0).sum())
                sets[split_name]['anomaly'] = int((sg.y[mask] == 1).sum())
        split_report_data['clients'][ci] = sets
    if args.task_mode == 'anomaly_binary':
        zero_anom = []
        for ci in range(args.num_clients):
            sg = subgraphs[ci]
            flags = {s: int((sg.y[sg[f'{s}_idx']] == 1).sum()) == 0
                     for s in ('fit', 'reliability', 'val', 'test')}
            split_report_data['clients'][ci]['zero_anomaly_flags'] = flags
            if all(flags.values()):
                zero_anom.append(ci)
        split_report_data['zero_anomaly_clients'] = zero_anom
        print(f"  [Zero-Anomaly Clients] 全集合零异常客户端: "
              f"{zero_anom if zero_anom else '无'}")
    if args.split_report_json:
        os.makedirs(os.path.dirname(args.split_report_json) or '.', exist_ok=True)
        with open(args.split_report_json, 'w') as f:
            json.dump(_sanitize_json(split_report_data), f, indent=1)
        print(f"[Split Report] 写入 {args.split_report_json}")

    # ---- RWR 结构缓存 (模式: off / safe_exact / epoch_reuse) ----
    if args.rwr_cache_mode == 'off':
        cache_enabled = False
        view1_persistent = False
    elif args.rwr_cache_mode == 'safe_exact':
        cache_enabled = True
        view1_persistent = False  # 只有所有 key 分量一致才命中 (不跨轮)
    else:  # epoch_reuse
        cache_enabled = (args.rwr_cache == 'enabled')
        view1_persistent = (args.rwr_cache_view1_persistent == 'enabled')
    rwr_cache = RWRSubgraphCache(
        max_entries=args.rwr_cache_max_entries,
        enabled=cache_enabled,
        view1_persistent=view1_persistent)
    print(f"[RWR Cache] mode={args.rwr_cache_mode} enabled={rwr_cache.enabled} "
          f"view1_persistent={rwr_cache._view1_persistent} "
          f"scope={args.rwr_cache_scope} "
          f"max_entries={args.rwr_cache_max_entries} rwr_seed={args.rwr_seed}")

    # =================================================================
    #  选择指标 + checkpoint 恢复
    # =================================================================
    selection_metric = resolve_selection_metric(args.task_mode, args.selection_metric)
    print(f"[Selection] metric={selection_metric} "
          f"checkpoint_dir={args.checkpoint_dir or '未启用'}")

    generator_cfg = dict(feat_dim=feat_dim, num_classes=num_classes,
                         hidden_dim=args.diffusion_hidden,
                         num_steps=args.diffusion_steps,
                         beta_start=args.diffusion_beta_start,
                         beta_end=args.diffusion_beta_end,
                         output_bound=args.generator_output_bound,
                         skip_mode=diffusion_skip_mode,
                         use_posterior_variance=args.use_posterior_variance)

    start_round = 0
    best_val_primary = 0.0
    best_test_primary = float('nan')
    best_round = -1
    pretrain_ref = None     # 联邦预训练后的参数锚 (L2-SP)
    radius_bank = None      # 预训练 radius bank [num_classes, bank_size]
    if args.resume_checkpoint:
        assert os.path.exists(args.resume_checkpoint), \
            f"--resume_checkpoint 不存在: {args.resume_checkpoint}"
        # Bug 2: 先按 checkpoint 记录的 freeze 策略重建 gen_optimizer 参数组,
        # 再 load optimizer state (避免 14/14 组 optimizer 装 13/14 组 state)
        # Bug A: checkpoint 与当前 CLI 的 freeze 策略不一致时 fail-fast
        ckpt_peek = torch.load(args.resume_checkpoint, map_location='cpu',
                               weights_only=False)
        saved_args = ckpt_peek.get('args') or {}
        freeze_policy = resolve_resume_freeze_policy(
            saved_args, args.diffusion_freeze_skip_scale)
        configure_generator_trainability(
            generator, True, freeze_skip_scale=freeze_policy)
        gen_optimizer = Adam([p for p in generator.parameters()
                              if p.requires_grad],
                             lr=args.generator_lr, weight_decay=0.0)
        del ckpt_peek
        ckpt = load_checkpoint(
            args.resume_checkpoint,
            global_model=global_model,
            generator=generator,
            global_optimizer=global_optimizer,
            gen_optimizer=gen_optimizer,
            ckr_tracker=tracker,
            local_optimizers=local_optimizers,
            expected_meta={'task_mode': args.task_mode,
                           'num_classes': num_classes,
                           'feat_dim': feat_dim},
            map_location=str(device))
        start_round = int(ckpt['round']) + 1
        best_val_primary = float(ckpt.get('best_metric', 0.0))
        best_round = int(ckpt['round'])
        # Bug 10: generator config 校验 (meta 存在时逐项核对, 缺失时依赖
        # strict state_dict + expected_meta 的缓冲校验)
        saved_cfg = (ckpt.get('meta') or {}).get('generator_cfg')
        if saved_cfg:
            validate_generator_cfg(saved_cfg, generator_cfg)
        else:
            print("  [Resume] legacy checkpoint: 无 generator_cfg metadata, "
                  "由 strict state_dict 加载保证一致性")
        # Bug 5: persistent 模式下本地 optimizer 状态必须存在, 否则 fail-fast
        if (args.local_optimizer_lifecycle == 'persistent'
                and ckpt.get('local_optimizer_states') is None
                and not args.allow_legacy_resume_without_local_optimizer_state):
            raise RuntimeError(
                "persistent local optimizer 模式的 checkpoint 缺少 "
                "local_optimizer_states, resume 后的训练轨迹与连续训练不一致。"
                "如需兼容旧 checkpoint, 显式加 "
                "--allow_legacy_resume_without_local_optimizer_state")
        if (ckpt.get('local_optimizer_states') is None
                and args.allow_legacy_resume_without_local_optimizer_state):
            print("  [Resume] legacy checkpoint: 本地 optimizer 状态缺失, "
                  "已显式允许 (轨迹可能与连续训练不同)")
        # 恢复 pretrained-manifold 约束状态 (避免 resume 后重建一个不同的 prior)
        if ckpt.get('pretrain_ref'):
            pretrain_ref = {k: v.to(device)
                            for k, v in ckpt['pretrain_ref'].items()}
        if ckpt.get('radius_bank') is not None:
            radius_bank = ckpt['radius_bank'].to(device)
        print(f"[Resume] 从 round {start_round} 继续训练 "
              f"(先前最佳 metric={best_val_primary:.4f}, "
              f"pretrain_ref={'有' if pretrain_ref else '无'}, "
              f"radius_bank={'有' if radius_bank is not None else '无'})")

    # =================================================================
    #  主循环
    # =================================================================
    primary_history = []

    round_start_times = []
    round_times = []
    gen_peak_mb_all = []

    # ---- 特征统计特性对齐: 客户端本地算 (μ,σ) 上行, 服务端聚合成全局统计量 ----
    # 一次性完成 (特征分布不随训练变化), 之后每轮生成伪特征时直接复用
    g_feat_mu = g_feat_std = None
    if args.feature_stats_align:
        g_feat_mu, g_feat_std = compute_global_feature_stats(subgraphs, device)
        print(f"[特征统计对齐] 已聚合全局特征统计量: "
              f"μ 均值={g_feat_mu.mean():.5f} σ 均值={g_feat_std.mean():.5f} "
              f"(客户端仅上行 2×{g_feat_mu.numel()} 个统计量, 未上传原始特征)")

    # ---- Stage 2 入口: 从 healthy pretrained checkpoint 初始化 ----
    # 只加载 generator 权重 (校验配置一致性), 不恢复预训练 Adam 状态;
    # θ_pre / radius bank 均取自加载后的状态, 对抗期 optimizer 全新重建。
    if getattr(args, 'diffusion_pretrained_checkpoint', ''):
        pretrain_ref, radius_bank, gen_optimizer = init_stage2_from_pretrained(
            generator, args.diffusion_pretrained_checkpoint, args, device,
            num_classes)
        # R0 基线 (任何对抗更新前); 用专用固定噪声, 不消耗全局 RNG (paired-arm 一致)
        try:
            diag_ckr = normalize_ckr_safe(tracker.static_ckr_scaled)
        except Exception:
            diag_ckr = (torch.ones(args.num_clients, num_classes,
                                   device=device) / args.num_clients)
        _fn = (128 // num_classes) * num_classes
        _dg0 = torch.Generator(device=device).manual_seed(args.seed + 8888)
        _dg1 = torch.Generator(device=device).manual_seed(args.seed + 9999)
        _dn = torch.randn(_fn, generator.feat_dim, generator=_dg0,
                          device=device)
        d0 = pretrain_only_diagnostics(generator, local_models, global_model,
                                       num_classes, device, diag_ckr,
                                       fake_nodes=_fn,
                                       initial_noise=_dn,
                                       noise_generator=_dg1)
        print(f"  [Stage2 R0 基线] raw μ={d0['raw_mu']:.4f} "
              f"σ={d0['raw_std']:.4f} r={d0['raw_radius_mean']:.4f} "
              f"L_sem={d0['L_sem']:.4f} L_dis={d0['L_dis']:.4f}")

    # ---- 联邦扩散预训练: 客户端本地训练去噪网络, 只上行参数 (专利 S3.1/S3.2 + 专利权1) ----
    # resume 时**不得**重复预训练: 恢复的 generator/optimizer 已含对抗期状态,
    # 再预训练会用新 prior 覆盖参数但 Adam 动量仍是旧的 (不一致)。
    if getattr(args, 'federated_diffusion_pretrain', False) and start_round == 0:
        print("\n" + "=" * 60)
        print("[联邦扩散预训练] 让去噪网络真正学到真实特征分布")
        print("=" * 60)
        federated_diffusion_pretrain(generator, subgraphs, args, device)

        # ---- pretrained-manifold 约束构建 (对抗期使用) ----
        pretrain_ref = {name: p.detach().clone()
                        for name, p in generator.named_parameters()}
        if args.radius_constraint:
            radius_bank = build_pretrain_radius_bank(
                generator, num_classes, device, args.radius_bank_size)
            r_mean = radius_bank.mean().item()
            r_std = radius_bank.std().item()
            print(f"  [Radius Bank] 每类 {args.radius_bank_size} 个样本, "
                  f"全局 r 均值={r_mean:.4f} σ={r_std:.4f} "
                  f"(冻结, 对抗期 fake_x = r_c·z/‖z‖)")

        # ---- pretrain-only 健康检查 (分级验证 Stage 1) ----
        try:
            diag_ckr = normalize_ckr_safe(tracker.static_ckr_scaled)
        except Exception:
            diag_ckr = (torch.ones(args.num_clients, num_classes,
                                   device=device) / args.num_clients)
        diag = pretrain_only_diagnostics(
            generator, local_models, global_model, num_classes, device,
            diag_ckr)
        print(f"  [Pretrain-Only 诊断] raw μ={diag['raw_mu']:.4f} "
              f"σ={diag['raw_std']:.4f} r={diag['raw_radius_mean']:.4f}±"
              f"{diag['raw_radius_std']:.4f} L_sem={diag['L_sem']:.4f} "
              f"L_dis={diag['L_dis']:.4f}")
        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(os.path.join(args.checkpoint_dir, 'pretrain_diag.json'),
                      'w') as f:
                json.dump(diag, f, indent=1)
        if args.pretrain_diagnostic_only:
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save(generator.state_dict(),
                           os.path.join(args.checkpoint_dir,
                                        'pretrained_generator.pt'))
                print(f"  [Pretrain-Only] generator 已保存: "
                      f"{os.path.join(args.checkpoint_dir, 'pretrained_generator.pt')}")
                if getattr(generator, 'skip_mode', 'none') == 'timestep_scalar':
                    with torch.no_grad():
                        ct = torch.exp(generator.skip_log_scale.weight).squeeze(-1)
                    json.dump({str(i): float(v) for i, v in enumerate(ct.cpu())},
                              open(os.path.join(
                                  args.checkpoint_dir,
                                  'learned_skip_coefficients.json'), 'w'))
            print("\n[Pretrain-Only 诊断模式] 不进入对抗主循环, 正常退出。")
            print("PRETRAIN_DIAG_ONLY_DONE")
            return 0

    use_radius_proj = bool(args.radius_constraint and radius_bank is not None)
    if args.radius_constraint and radius_bank is None:
        print("  ⚠ [Radius] --radius_constraint 开但 radius_bank 缺失 "
              "(预训练被关闭或旧 checkpoint 无 bank), 回退到 output_bound 路径")
    if args.paired_rng_diag:
        print(f"  [PairedRNG] P1 (Stage2 初始化后, 本地训练前): "
              f"rng={_rng_fingerprint()} classifier={_model_param_hash(global_model)}")
        _data_h = hashlib.md5()
        for _i in range(args.num_clients):
            with open(f"dataset/{args.dataset}/Client{args.num_clients}/"
                      f"Louvain/data{_i}.pt", 'rb') as _f:
                _data_h.update(_f.read())
        _ck_h = 'None'
        if getattr(args, 'diffusion_pretrained_checkpoint', ''):
            with open(args.diffusion_pretrained_checkpoint, 'rb') as _f:
                _ck_h = hashlib.md5(_f.read()).hexdigest()[:12]
        print(f"  [PairedID] data_hash={_data_h.hexdigest()[:12]} "
              f"init={_model_param_hash(global_model)} ckpt={_ck_h} "
              f"rng_protocol=v2(reseed+dedicated) "
              f"server_start_round={args.server_start_round}")

    for round_id in range(start_round, args.num_rounds):
        t_round0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"[Round {round_id}]")

        # ---- 平台集成: 轮次边界应用待生效参数 (热更新) ----
        if args.param_update_dir:
            _apply_pending_param_updates(
                args, round_id, local_optimizers, gen_optimizer, global_optimizer)
        _emit_event(args, 'round_started', round_id=round_id, stage='client_training')

        # ---- [RoundStart 诊断] 本轮任何本地训练前 val + θ 快照 (T0) ----
        if args.federated_mode == 'fedavg':
            _was_tr = global_model.training
            global_model.eval()
            with torch.no_grad():
                _vs_res = evaluate_global(global_model, subgraphs,
                                          args.task_mode, split='val',
                                          device=device)
            if _was_tr:
                global_model.train()
            _vs = (float(_vs_res.get('accuracy', float('nan')))
                   if isinstance(_vs_res, dict) else float('nan'))
            theta_start = [p.detach().clone() for p in global_model.parameters()]
            print(f"  [RoundStart] val={_vs:.4f} (本轮本地训练前)")
        else:
            _vs = float('nan')
            theta_start = None

        # -----------------------------------------------------------
        #  [1] 广播全局模型 -> [2] 客户端 fit_idx 本地训练
        # -----------------------------------------------------------
        print("\n[Client Local Training] (fit_idx 用于 weighted CE 与参数更新)")

        aug_edge_indices = []
        if args.contrastive_mode == 'subgraph_cross_view':
            for ci in range(args.num_clients):
                reseed_client_rng(args, round_id, ci)   # 客户端随机流确定性解耦
                aug_ei = edge_perturbation(
                    subgraphs[ci].edge_index,
                    subgraphs[ci].x.shape[0],
                    args.edge_perturb_ratio,
                )
                aug_edge_indices.append(aug_ei.to(device))
                if args.paired_rng_diag and ci == 0 and round_id == 0:
                    print(f"  [PairedRNG] client0 aug_edge hash="
                          f"{_tensor_hash(aug_ei)}")

        round_ce_loss = 0.0
        round_cl_loss = 0.0
        round_cl_count = 0

        for ci in range(args.num_clients):
            set_requires_grad(local_models[ci], True)
        if args.federated_mode == 'fedavg':
            # 广播全局模型
            for ci in range(args.num_clients):
                local_models[ci].load_state_dict(global_model.state_dict())
            if args.local_optimizer_lifecycle == 'reset_each_round':
                # Bug 5: 每轮广播后重建本地 optimizer, 不继承上轮 Adam 动量
                local_optimizers = [
                    Adam(local_models[ci].parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
                    for ci in range(args.num_clients)]

        # 锚点采样: random_each_epoch / fixed_per_round (轮内固定) / fixed_global (全程固定)
        round_anchors = {}
        if args.anchor_sampling_mode == 'fixed_global':
            global_anchors = {}
            for ci in range(args.num_clients):
                data = subgraphs[ci]
                pool = (torch.where(data.fit_idx)[0]
                        if args.contrastive_anchor_scope == 'train_nodes'
                        else torch.arange(data.x.shape[0], device=device))
                if args.anchor_pool_size > 0:
                    pool = pool[:min(args.anchor_pool_size, len(pool))]
                max_b = min(args.contrastive_batch_size, len(pool))
                perm = torch.randperm(len(pool), generator=torch.Generator()
                                      .manual_seed(args.seed + ci))[:max_b]
                global_anchors[ci] = pool[perm]

        for ci in range(args.num_clients):
            reseed_client_rng(args, round_id, ci)   # 客户端随机流确定性解耦
            local_models[ci].train()
            data = subgraphs[ci]

            _cli_ce_sum = 0.0
            _cli_cl_sum = 0.0
            _cli_cl_cnt = 0
            if args.paired_rng_diag and ci == 0 and round_id == 0:
                print(f"  [PairedRNG] P2 (client0 训练前): rng={_rng_fingerprint()} "
                      f"local0={_model_param_hash(local_models[0])}")
            for epoch_id in range(args.num_epochs):
                local_optimizers[ci].zero_grad()

                logits, embeddings = local_models[ci].forward(data, return_embedding=True)
                fit_logits = logits[data.fit_idx]
                fit_labels = data.y[data.fit_idx]

                if args.use_weighted_ce:
                    ce_loss = F.cross_entropy(fit_logits, fit_labels,
                                              weight=class_weights[ci])
                else:
                    ce_loss = loss_fn(fit_logits, fit_labels)
                loss = ce_loss

                # 子图-子图跨视图对比 (reliability_idx 不参与任何 backward)
                if args.contrastive_mode == 'subgraph_cross_view':
                    if args.contrastive_anchor_scope == 'train_nodes':
                        pool = torch.where(data.fit_idx)[0]
                    else:
                        pool = torch.arange(data.x.shape[0], device=device)
                    if args.anchor_pool_size > 0:
                        pool = pool[:min(args.anchor_pool_size, len(pool))]
                    max_b = min(args.contrastive_batch_size, len(pool))
                    if args.anchor_sampling_mode == 'fixed_global':
                        anchors = global_anchors[ci]
                    elif args.anchor_sampling_mode == 'fixed_per_round':
                        if ci not in round_anchors:
                            perm = torch.randperm(len(pool), generator=torch.Generator()
                                                  .manual_seed(args.seed + round_id * 1000 + ci))[:max_b]
                            round_anchors[ci] = pool[perm]
                        anchors = round_anchors[ci]
                    else:  # random_each_epoch
                        perm = torch.randperm(len(pool), device=device)[:max_b]
                        anchors = pool[perm]
                    cl_loss = subgraph_contrastive_step(
                        model=local_models[ci], data=data,
                        aug_edge_index=aug_edge_indices[ci],
                        anchor_nodes=anchors,
                        subgraph_size=args.rwr_subgraph_size,
                        tau=args.contrastive_temperature,
                        device=device,
                        rwr_cache=rwr_cache,
                        client_id=ci,
                        round_idx=round_id,
                        view1_persistent=(args.rwr_cache_view1_persistent == 'enabled'),
                        rwr_seed=args.rwr_seed,
                        cache_scope=args.rwr_cache_scope,
                    )
                    if (args.paired_rng_diag and ci == 0 and round_id == 0
                            and epoch_id == 0):
                        print(f"  [PairedRNG] client0 epoch0 anchors="
                              f"{_tensor_hash(anchors)} ce={ce_loss.item():.6f} "
                              f"cl={cl_loss.item():.6f}")
                    loss = loss + args.lambda_subgraph * cl_loss
                    round_cl_loss += cl_loss.item()
                    round_cl_count += 1
                    _cli_cl_sum += cl_loss.item()
                    _cli_cl_cnt += 1

                loss.backward()
                local_optimizers[ci].step()
                if (args.paired_rng_diag and ci == 0 and round_id == 0
                        and epoch_id == 0):
                    print(f"  [PairedRNG] client0 epoch0 后 local0="
                          f"{_model_param_hash(local_models[0])} "
                          f"rng={_rng_fingerprint()}")
                if math.isnan(loss.item()) or _nan_diag([local_models[ci]],
                                                        f'client{ci}_train'):
                    print(f"  [NAN-DIAG] round {round_id} client {ci} "
                          f"训练后出现 NaN (ce={ce_loss.item():.4f})")
                round_ce_loss += ce_loss.item()
                _cli_ce_sum += ce_loss.item()
            if args.paired_rng_diag and ci == 0 and round_id == 0:
                print(f"  [PairedRNG] P3 (client0 训练后): rng={_rng_fingerprint()} "
                      f"local0={_model_param_hash(local_models[0])}")

            # ---- 平台集成: 客户端训练结构化事件 ----
            fit_y = data.y[data.fit_idx]
            class_counts = torch.bincount(fit_y, minlength=num_classes).tolist()
            _emit_event(
                args, 'client_training_metric', round_id=round_id,
                stage='client_training', client_id=ci,
                ce_loss=round(_cli_ce_sum / max(1, args.num_epochs), 6),
                cl_loss=(round(_cli_cl_sum / max(1, _cli_cl_cnt), 6)
                         if _cli_cl_cnt else None),
                total_loss=round((_cli_ce_sum
                                  + args.lambda_subgraph * _cli_cl_sum)
                                 / max(1, args.num_epochs), 6),
                train_samples=int(fit_y.shape[0]),
                class_counts=class_counts,
                zero_anomaly=(bool((fit_y == 1).sum() == 0)
                              if args.task_mode == 'anomaly_binary' else False),
            )

        n_train_units = args.num_clients * args.num_epochs
        ce_avg = round_ce_loss / max(1, n_train_units)
        cl_avg = round_cl_loss / max(1, round_cl_count)
        if args.contrastive_mode == 'subgraph_cross_view':
            print(f"  [RWR Cache] hit_rate={rwr_cache.hit_rate:.3f} "
                  f"stats={rwr_cache.stats()}")

        # -----------------------------------------------------------
        #  [3-6] 服务端权重解析: distill_weighting 决定是否计算动态 CKR
        #        (local_only = 基线 B0, 无聚合/生成器/蒸馏)
        # -----------------------------------------------------------
        use_dynamic_ckr = (args.federated_mode == 'fedavg'
                           and args.distill_weighting == 'dynamic_ckr')
        per_client_metrics = {}
        val_diag = {}

        if args.federated_mode == 'local_only':
            # B0: 每客户端独立训练, 记录 val 诊断
            print("\n[Local Only] 无服务器聚合 (基线 B0)")
            for ci in range(args.num_clients):
                local_models[ci].eval()
                em = evaluate_client(local_models[ci], subgraphs[ci], args.task_mode,
                                     eval_mask=subgraphs[ci].val_idx)
                val_diag[ci] = {k: _sanitize_json(v) for k, v in em.items()
                                if not isinstance(v, dict)}
            normalized_ckr = (torch.ones(args.num_clients, num_classes, device=device)
                              / args.num_clients)
        elif use_dynamic_ckr:
            print("\n[Dynamic CKR Update]")
            eval_idx_by_client = []
            for ci in range(args.num_clients):
                if args.dynamic_ckr_eval_source == 'validation':
                    eval_idx_by_client.append(subgraphs[ci].val_idx)
                else:
                    eval_idx_by_client.append(subgraphs[ci].reliability_idx)

            # 客户端仅上传: 每类别分数 + support + available mask
            for ci in range(args.num_clients):
                local_models[ci].eval()
                pm = compute_per_class_reliability_metrics(
                    model=local_models[ci],
                    data=subgraphs[ci],
                    eval_idx=eval_idx_by_client[ci],
                    num_classes=num_classes,
                    min_support=args.reliability_min_support,
                    metric=args.dynamic_ckr_metric)
                per_client_metrics[ci] = pm

            fit_support_by_client = {
                ci: [int((subgraphs[ci].y[subgraphs[ci].fit_idx] == c).sum())
                     for c in range(num_classes)]
                for ci in range(args.num_clients)}
            tracker.update(
                client_ids=list(range(args.num_clients)),
                per_client_metrics=per_client_metrics,
                round_idx=round_id + 1,
                per_client_fit_support=fit_support_by_client)
            normalized_ckr = tracker.last_server_weights

            print(tracker.get_log_summary(list(range(args.num_clients)),
                                          per_client_metrics))
            if args.task_mode == 'anomaly_binary':
                print(tracker.anomaly_binary_summary(args.normal_classes))
        elif args.distill_weighting == 'equal':
            normalized_ckr = (torch.ones(args.num_clients, num_classes, device=device)
                              / args.num_clients)
            print("\n[Server Weights] equal (无 CKR 等权蒸馏)")
        elif args.distill_weighting == 'static_ckr':
            normalized_ckr = normalize_ckr_safe(tracker.static_ckr_scaled)
            print("\n[Server Weights] static_ckr (静态拓扑 CKR 蒸馏)")
        else:  # none
            normalized_ckr = normalize_ckr_safe(tracker.static_ckr_scaled)
            print("\n[Server Weights] none (跳过生成器与蒸馏)")

        # ---- CKR 诊断模式覆盖 (阶段三) ----
        # shuffled: 每类打乱客户端权重 (保留边际分布, 非均匀权重对照)
        # inverse : 每类反转权重 (负对照, 不得作为主方法)
        # oracle_validation: 客户端 validation 类别性能作权重 (诊断上界, 严禁 test)
        diag_tag = ''
        if args.ckr_diagnostic_mode == 'shuffled':
            g = torch.Generator()
            g.manual_seed(args.model_seed * 1000 + round_id)
            base_w = normalize_ckr_safe(tracker.static_ckr_scaled).clone()
            for c in range(num_classes):
                perm = torch.randperm(args.num_clients, generator=g)
                base_w[:, c] = base_w[perm, c]
            normalized_ckr = normalize_ckr_safe(base_w)
            diag_tag = 'shuffled_ckr'
            print("[Server Weights] shuffled_ckr (每类打乱, 保留边际分布)")
        elif args.ckr_diagnostic_mode == 'inverse':
            base_w = normalize_ckr_safe(tracker.static_ckr_scaled).clone()
            for c in range(num_classes):
                col = base_w[:, c]
                inv = col.max() - col
                if inv.sum() > 1e-8:
                    base_w[:, c] = inv / inv.sum()
                else:
                    base_w[:, c] = 1.0 / args.num_clients
            normalized_ckr = base_w
            diag_tag = 'inverse_ckr'
            print("[Server Weights] inverse_ckr (负对照: 权重反转, 仅诊断)")
        elif args.ckr_diagnostic_mode == 'oracle_validation':
            oracle_w = torch.zeros(args.num_clients, num_classes, device=device)
            for ci in range(args.num_clients):
                local_models[ci].eval()
                pm = compute_per_class_reliability_metrics(
                    model=local_models[ci], data=subgraphs[ci],
                    eval_idx=subgraphs[ci].val_idx,
                    num_classes=num_classes, min_support=1,
                    metric='f1')
                oracle_w[ci] = torch.tensor(pm['per_class_f1'],
                                            device=device).nan_to_num(0.0)
            normalized_ckr = normalize_ckr_safe(oracle_w)
            diag_tag = 'oracle_validation'
            print("[Server Weights] oracle_validation (诊断上界, 非隐私友好)")

        # ---- 公平性: validation_deficit_distillation (只使用 validation 标量) ----
        fairness_factor = None
        if (args.fairness_mode in ('validation_deficit_distillation', 'combined')
                and round_id >= args.fairness_warmup_rounds):
            metrics_k = {}
            for ci in range(args.num_clients):
                local_models[ci].eval()
                em = evaluate_client(local_models[ci], subgraphs[ci],
                                     args.task_mode,
                                     eval_mask=subgraphs[ci].val_idx)
                if args.task_mode == 'anomaly_binary':
                    m = em.get('roc_auc', float('nan'))
                    if math.isnan(m):
                        m = em.get('accuracy', float('nan'))
                else:
                    m = em.get('accuracy', float('nan'))
                metrics_k[ci] = m
            valid_m = [v for v in metrics_k.values() if not math.isnan(v)]
            if valid_m:
                median_m = sorted(valid_m)[len(valid_m) // 2]
                deficit = {ci: max(0.0, median_m - m)
                           for ci, m in metrics_k.items() if not math.isnan(m)}
                factor = {ci: math.exp(args.fairness_alpha * d)
                          for ci, d in deficit.items()}
                fmin, fmax = args.fairness_weight_min, args.fairness_weight_max
                factor = {ci: min(fmax, max(fmin, f))
                          for ci, f in factor.items()}
                fairness_factor = factor
                # 最终类别蒸馏权重: ckr_weight * fairness_factor, 按类别归一化
                w = normalized_ckr.clone()
                for ci, f in factor.items():
                    w[ci] *= f
                normalized_ckr = normalize_ckr_safe(w)
                print(f"  [Fairness] validation_deficit: "
                      f"median={median_m:.2f} factor范围="
                      f"[{min(factor.values()):.2f}, {max(factor.values()):.2f}] "
                      f"clip=[{fmin},{fmax}]")

        # ---- 教师混合诊断 (teacher_mixture.jsonl, 仅蒸馏开启时) ----
        if (args.teacher_mixture_jsonl and args.distill_weighting != 'none'
                and args.federated_mode == 'fedavg'):
            _write_teacher_mixture(args.teacher_mixture_jsonl, round_id,
                                   normalized_ckr, tracker, num_classes)

        # ---- 平台集成: CKR 结构化事件 (静态/动态权重矩阵 + fallback 状态) ----
        _emit_event(
            args, 'ckr_update', round_id=round_id, stage='server_weights',
            ckr_mode=args.ckr_mode,
            static_raw=_sanitize_json(tracker.static_ckr_raw.cpu()),
            static_scaled=_sanitize_json(tracker.static_ckr_scaled.cpu()),
            server_weights=_sanitize_json(normalized_ckr.cpu()),
            effective_ckr=_sanitize_json(tracker.current_ema.cpu()),
            fallback_reasons=getattr(tracker, 'last_fallback_reason', None) or None,
        )

        # -----------------------------------------------------------
        #  [7] FedAvg (qffl_aggregation 时按 validation loss 调整权重)
        # -----------------------------------------------------------
        if args.federated_mode == 'fedavg':
            print("\n[Server FedAvg]")
            total_nodes = sum(sg.x.shape[0] for sg in subgraphs)
            agg_w = {ci: subgraphs[ci].x.shape[0] / total_nodes
                     for ci in range(args.num_clients)}
            if (args.fairness_mode in ('qffl_aggregation', 'combined')
                    and round_id >= args.fairness_warmup_rounds):
                # q-FedAvg 风格: w_k ∝ n_k * exp(-q * loss_k / mean_loss)
                sigs = _client_val_signals(local_models, subgraphs,
                                           args.task_mode, device)
                losses = [sigs[ci]['val_loss'] for ci in range(args.num_clients)]
                mean_loss = sum(losses) / len(losses) if losses else 1.0
                q = args.fairness_q
                scores = {ci: agg_w[ci] * math.exp(-q * sigs[ci]['val_loss']
                                                   / max(mean_loss, 1e-8))
                          for ci in range(args.num_clients)}
                ssum = sum(scores.values())
                agg_w = {ci: s / ssum for ci, s in scores.items()}
                print(f"  [Fairness] qffl_aggregation (q={q}): 权重范围 "
                      f"[{min(agg_w.values()):.4f}, {max(agg_w.values()):.4f}]")
            with torch.no_grad():
                for ci in range(args.num_clients):
                    w = agg_w[ci]
                    for lp, gp in zip(local_models[ci].parameters(), global_model.parameters()):
                        if ci == 0:
                            gp.data.copy_(w * lp.data)
                        else:
                            gp.data.add_(w * lp.data)
            print(f"  聚合 {args.num_clients} 个客户端, 初始全局模型已生成")
            if args.paired_rng_diag and round_id == 0:
                print(f"  [PairedRNG] P4 (After FedAvg): rng={_rng_fingerprint()} "
                      f"global={_model_param_hash(global_model)}")
            # ---- [Drift 诊断] 本地更新范数 vs 轮起始全局模型 ----
            if theta_start is not None:
                gup = math.sqrt(sum(
                    ((gp - ts) ** 2).sum().item()
                    for gp, ts in zip(global_model.parameters(), theta_start)))
                cup = [math.sqrt(sum(
                    ((lp - ts) ** 2).sum().item()
                    for lp, ts in zip(local_models[ci].parameters(),
                                      theta_start)))
                    for ci in range(args.num_clients)]
                cup_sorted = sorted(cup)
                cup_mean = sum(cup) / len(cup)
                cup_med = cup_sorted[len(cup) // 2]
                cup_p95 = cup_sorted[int(len(cup) * 0.95) - 1]
                cup_max = cup_sorted[-1]
                th_norm = math.sqrt(sum((ts ** 2).sum().item()
                                        for ts in theta_start))
                # 客户端更新方向冲突度 (pairwise cosine, 轻量)
                deltas = [torch.cat([(lp - ts).flatten()
                                     for lp, ts in zip(
                                         local_models[ci].parameters(),
                                         theta_start)])
                          for ci in range(args.num_clients)]
                cos = []
                for i in range(args.num_clients):
                    for j in range(i + 1, args.num_clients):
                        cos.append(float(
                            torch.dot(deltas[i], deltas[j])
                            / (deltas[i].norm() * deltas[j].norm() + 1e-12)))
                cos_sorted = sorted(cos)
                print(f"  [Drift] client_update_norm mean={cup_mean:.4f} "
                      f"median={cup_med:.4f} p95={cup_p95:.4f} "
                      f"max={cup_max:.4f} | relative="
                      f"{cup_mean / max(th_norm, 1e-12):.1%} "
                      f"| global_update_norm={gup:.4f}")
                print(f"  [Drift] client update pairwise cosine: "
                      f"mean={sum(cos) / len(cos):.3f} "
                      f"median={cos_sorted[len(cos) // 2]:.3f} "
                      f"min={cos_sorted[0]:.3f} max={cos_sorted[-1]:.3f}")
        else:
            print("\n[Server FedAvg] 已跳过 (local_only)")

        # ---- [ServerDelta 诊断] 服务端更新前 val (本地聚合后, 蒸馏/对抗前) ----
        if args.federated_mode == 'fedavg':
            _was_tr = global_model.training
            global_model.eval()
            with torch.no_grad():
                _vb_res = evaluate_global(global_model, subgraphs,
                                          args.task_mode, split='val',
                                          device=device)
            if _was_tr:
                global_model.train()
            _vb = (float(_vb_res.get('accuracy', float('nan')))
                   if isinstance(_vb_res, dict) else float('nan'))
        else:
            _vb = float('nan')

        # -----------------------------------------------------------
        #  [8] 生成器更新 + [9] 全局蒸馏
        #      (local_only / distill_weighting=none / round < server_start_round
        #       时整体跳过; server_start_round 只推迟服务器阶段, 不影响
        #       Stage2 初始化)
        # -----------------------------------------------------------
        L_G_report = None
        L_D_report = None
        gen_peak_mb = 0.0
        server_ran = False
        if (args.distill_weighting != 'none' and args.federated_mode == 'fedavg'
                and round_id >= args.server_start_round):
            server_ran = True
            print("\n[Generator Update]")

            # 伪标签 (本轮 CKR 驱动)
            label_rng = make_server_rng(device, args.seed, round_id, 1)
            fake_labels, counts = sample_fake_labels(
                num_nodes=args.fake_nodes,
                num_classes=num_classes,
                strategy=args.fake_class_strategy,
                ckr_weights=normalized_ckr.sum(dim=0).tolist(),
                device=device,
                generator=label_rng,
            )
            print(f"  伪标签分布: {[(c, counts[c]) for c in range(num_classes)]}")

            # 冻结
            for ci in range(args.num_clients):
                set_requires_grad(local_models[ci], False)
            set_requires_grad(global_model, False)
            configure_generator_trainability(
                generator, True,
                freeze_skip_scale=args.diffusion_freeze_skip_scale)
            generator.train()
            for ci in range(args.num_clients):
                local_models[ci].eval()
            global_model.eval()

            is_warmup = round_id < args.generator_warmup_rounds
            phase_tag = "WARMUP" if is_warmup else "ADVERSARIAL"
            print(f"  阶段: {phase_tag} | 生成器可训练参数: {count_trainable_params(generator)}")
            print(f"  反向传播模式: {args.generator_backprop_mode} "
                  f"(truncate_interval={args.generator_truncate_interval})")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            sampling_steps = args.generator_sampling_steps or args.diffusion_steps
            if (round_id == 0 and 0 < sampling_steps < args.diffusion_steps):
                print(f"  ⚠ [sampling] generator_sampling_steps={sampling_steps} "
                      f"< diffusion_steps={args.diffusion_steps}: 从 "
                      f"t={sampling_steps - 1} 起步但 x_T 仍是纯噪声, "
                      f"alpha_bar[{sampling_steps - 1}]="
                      f"{generator.alpha_bars[sampling_steps - 1].item():.3f} ≠ 0, "
                      f"存在时序错配。建议满步 (--generator_sampling_steps 0) "
                      f"或实现正确 respacing。")

            def _make_fake_graph(fake_x):
                if args.fake_graph_topology == 'isolated':
                    # D8: 伪图只含自环 (检验 KNN 代理拓扑贡献)
                    B = fake_x.shape[0]
                    ei = torch.arange(B, device=device).repeat(2, 1)
                    return Data(x=fake_x, edge_index=ei)
                k = min(args.knn_k, fake_labels.shape[0] - 1)
                return build_knn_graph(fake_x, k=k)

            # 清空所有 generator 参数历史 grad (被冻结的 skip scale 不在
            # gen_optimizer 参数组, optimizer.zero_grad 清不到它)
            generator.zero_grad(set_to_none=True)

            for step in range(args.generator_steps):
                gen_optimizer.zero_grad()

                if args.generator_update_mode == 'frozen':
                    # D5: 生成器不更新 (随机参数生成器只用于采样)
                    L_G_report = None
                    L_G = torch.tensor(0.0, device=device)
                    print(f"  [D5] 生成器冻结: 不更新参数")
                    break

                # 可微生成 (full / checkpointed / truncated)
                # radius 约束下取 raw 输出再投影: x = r_c·z/‖z‖, 堵住幅度作弊
                init_rng = make_server_rng(
                    device, args.seed, round_id, 100 + step * 2)
                reverse_rng = make_server_rng(
                    device, args.seed, round_id, 101 + step * 2)
                init_noise = torch.randn(
                    fake_labels.shape[0], generator.feat_dim,
                    generator=init_rng, device=device)
                raw_fake_x = generator.differentiable_sample(
                    labels=fake_labels, num_steps=sampling_steps,
                    backprop_mode=args.generator_backprop_mode,
                    truncate_interval=args.generator_truncate_interval,
                    checkpoint_segments=args.checkpoint_segments,
                    apply_output_bound=(not use_radius_proj),
                    initial_noise=init_noise,
                    noise_generator=reverse_rng)
                fake_x = raw_fake_x
                if use_radius_proj:
                    fake_x = project_to_pretrain_radius(
                        raw_fake_x, fake_labels, radius_bank)
                if args.feature_stats_align:
                    fake_x = match_feature_stats(fake_x, g_feat_mu, g_feat_std)
                fake_graph = _make_fake_graph(fake_x)

                # 损失
                L_sem = compute_generator_semantic_loss(
                    fake_graph, fake_labels, local_models,
                    normalized_ckr, num_classes, device)
                L_dis = compute_generator_disagreement_loss(
                    fake_graph, fake_labels, local_models,
                    global_model, normalized_ckr, num_classes, device,
                    loss_type=args.distill_loss_type)

                # 多样性
                norm_x = F.normalize(fake_x, p=2, dim=1)
                sim_mat = torch.mm(norm_x, norm_x.t())
                mask = ~torch.eye(fake_x.shape[0], device=device, dtype=torch.bool)
                L_div = sim_mat[mask].mean()

                # 特征范数
                L_norm = torch.mean(fake_x ** 2)

                L_G = (args.lambda_sem * L_sem
                       + args.lambda_diversity * L_div
                       + args.lambda_feature_norm * L_norm)
                if not is_warmup:
                    L_G = L_G - args.lambda_disagreement * L_dis
                # L2-SP 参数锚: 限制生成器离预训练 prior (θ_pre) 太远
                L_anchor = None
                if (pretrain_ref is not None
                        and args.lambda_diffusion_anchor > 0):
                    L_anchor = diffusion_prior_anchor_loss(generator,
                                                           pretrain_ref)
                    L_G = L_G + args.lambda_diffusion_anchor * L_anchor

                formal_finite_guard(
                    args, round_id, f'generator_step_{step}_pre_backward',
                    tensors={'raw_fake_x': raw_fake_x, 'projected_fake_x': fake_x,
                             'L_sem': L_sem, 'L_dis': L_dis, 'L_div': L_div,
                             'L_norm': L_norm, 'L_G': L_G, 'L_anchor': L_anchor},
                    modules={'generator': generator})

                L_G.backward()
                formal_finite_guard(
                    args, round_id, f'generator_step_{step}_post_backward',
                    modules={'generator': generator}, check_grads=True)

                grad_norm = math.sqrt(sum(p.grad.norm().item()**2
                                          for p in generator.parameters() if p.grad is not None))
                if math.isnan(grad_norm):
                    _nan_diag([generator], f'generator round{round_id} '
                                           f'L_sem={L_sem.item():.6f} '
                                           f'L_dis={L_dis.item():.6f} '
                                           f'L_div={L_div.item():.6f} '
                                           f'L_norm={L_norm.item():.6f}')
                if grad_norm < 1e-8:
                    print("  ⚠ 生成器梯度为零!")
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                gen_optimizer.step()
                formal_finite_guard(
                    args, round_id, f'generator_step_{step}_post_step',
                    modules={'generator': generator})
                L_G_report = L_G.item()

                if torch.cuda.is_available():
                    gen_peak_mb = max(gen_peak_mb,
                                      torch.cuda.max_memory_allocated() / 1024 ** 2)

                # Diagnostics in float64: do not manufacture raw_r=inf merely
                # because float32 sum-of-squares overflowed while raw elements
                # are still finite.  The in-process health guard separately
                # rejects truly nonfinite raw/projected tensors.
                raw_r = diagnostic_l2_norm(raw_fake_x, dim=1).median().item()
                proj_r = diagnostic_l2_norm(fake_x, dim=1).median().item()
                ratio_extra = ""
                if use_radius_proj:
                    _M = radius_bank.shape[1]
                    _tgt = radius_bank[fake_labels,
                                       torch.arange(fake_labels.shape[0],
                                                    device=device) % _M]
                    _tgt_med = _tgt.median().item()
                    # cheating-pressure: raw 输出相对 target 的放大倾向
                    ratio_extra = (f" raw/target={raw_r / max(_tgt_med, 1e-9):.3f}"
                                   f" proj/target={proj_r / max(_tgt_med, 1e-9):.3f}")
                extra = (f" anchor={L_anchor.item():.4f}"
                         if L_anchor is not None else "")
                print(f"  step {step}: L_sem={L_sem.item():.4f} "
                      f"L_dis={L_dis.item():.4f} "
                      f"L_div={L_div.item():.4f} L_norm={L_norm.item():.4f} "
                      f"L_G={L_G.item():.4f} |grad|={grad_norm:.4f} "
                      f"fake_x μ={fake_x.mean().item():.4f} "
                      f"σ={fake_x.std().item():.4f} "
                      f"raw_r={raw_r:.4f} proj_r={proj_r:.4f}"
                      f"{ratio_extra}{extra}")

            if pretrain_ref is not None:
                drift = compute_theta_drift(generator, pretrain_ref)
                c_delta = None
                if (getattr(generator, 'skip_mode', 'none') == 'timestep_scalar'
                        and 'skip_log_scale.weight' in pretrain_ref):
                    c_delta = float(
                        (generator.skip_log_scale.weight
                         - pretrain_ref['skip_log_scale.weight'])
                        .abs().max().item())
                print(f"  [Prior Drift] theta_drift={drift:.4f} "
                      f"(‖θ−θ_pre‖/‖θ_pre‖)"
                      + (f" | skip_delta={c_delta:.2e} (冻结校验)"
                         if c_delta is not None else ""))

            if torch.cuda.is_available():
                print(f"  [Memory] 生成器阶段峰值显存: {gen_peak_mb:.1f} MB")
            else:
                print("  [Memory] CPU 环境, 跳过 CUDA 显存日志")

            # ---- 平台集成: 生成器结构化事件 ----
            if L_G_report is not None:
                _emit_event(
                    args, 'generator_metric', round_id=round_id,
                    stage='generator_update',
                    generator_loss=round(L_G_report, 6),
                    semantic_loss=round(L_sem.item(), 6),
                    disagreement_loss=round(L_dis.item(), 6),
                    diversity_loss=round(L_div.item(), 6),
                    feature_norm_loss=round(L_norm.item(), 6),
                    grad_norm=round(grad_norm, 6),
                    fake_x_mean=round(float(fake_x.mean().item()), 6),
                    fake_x_std=round(float(fake_x.std().item()), 6),
                    fake_x_min=round(float(fake_x.min().item()), 6),
                    fake_x_max=round(float(fake_x.max().item()), 6),
                    fake_x_norm=round(float(diagnostic_l2_norm(
                        fake_x, dim=1).mean().item()), 6),
                    raw_radius_median=round(float(raw_r), 6),
                    projected_radius_median=round(float(proj_r), 6),
                    fake_label_counts={int(c): int(n)
                                       for c, n in enumerate(counts)},
                    peak_gpu_mb=round(gen_peak_mb, 1),
                    warmup=bool(is_warmup),
                )

            # ---- [9] 全局蒸馏 (使用本轮 CKR) ----
            print("\n[Global Distillation]")

            # 用推理 sample 生成最终伪图
            # 与生成器更新**同一参数化**: raw → radius 投影 → KNN (两边对抗同一分布)
            with torch.no_grad():
                final_init_rng = make_server_rng(device, args.seed, round_id, 900)
                final_reverse_rng = make_server_rng(device, args.seed, round_id, 901)
                final_init_noise = torch.randn(
                    fake_labels.shape[0], generator.feat_dim,
                    generator=final_init_rng, device=device)
                raw_fake_x = generator.sample(
                    labels=fake_labels, num_steps=sampling_steps, device=device,
                    apply_output_bound=(not use_radius_proj),
                    initial_noise=final_init_noise,
                    noise_generator=final_reverse_rng)
                fake_x = raw_fake_x
                if use_radius_proj:
                    fake_x = project_to_pretrain_radius(
                        raw_fake_x, fake_labels, radius_bank)
                if args.feature_stats_align:
                    fake_x = match_feature_stats(fake_x, g_feat_mu, g_feat_std)
                fake_graph = _make_fake_graph(fake_x)

            # 蒸馏前快照 (预测变化诊断)
            global_model.eval()
            with torch.no_grad():
                logits_before = []
                for sg in subgraphs:
                    logits_before.append(global_model.forward(sg)[sg.val_idx])
            params_before = [p.clone() for p in global_model.parameters()]

            configure_generator_trainability(
                generator, False,
                freeze_skip_scale=args.diffusion_freeze_skip_scale)
            for ci in range(args.num_clients):
                set_requires_grad(local_models[ci], False)
            set_requires_grad(global_model, True)
            global_model.train()
            print(f"  全局模型可训练参数: {count_trainable_params(global_model)}")

            distill_grad_norm = 0.0
            for step in range(args.distill_steps):
                global_optimizer.zero_grad()
                L_D = compute_student_distillation_loss(
                    fake_graph=fake_graph,
                    fake_labels=fake_labels,
                    local_models=local_models,
                    global_model=global_model,
                    normalized_ckr=normalized_ckr,
                    num_classes=num_classes,
                    device=device,
                    temperature=args.distill_temperature,
                    loss_type=args.distill_loss_type,
                )
                formal_finite_guard(
                    args, round_id, f'distill_step_{step}_pre_backward',
                    tensors={'L_D': L_D, 'fake_x': fake_x},
                    modules={'global_model': global_model})
                L_D.backward()
                formal_finite_guard(
                    args, round_id, f'distill_step_{step}_post_backward',
                    modules={'global_model': global_model}, check_grads=True)
                if math.isnan(L_D.item()):
                    _nan_diag([global_model], f'distill round {round_id} '
                                              f'L_D={L_D.item()}')
                gn = math.sqrt(sum(p.grad.norm().item()**2
                                   for p in global_model.parameters() if p.grad is not None))
                distill_grad_norm = max(distill_grad_norm, gn)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10.0)
                global_optimizer.step()
                formal_finite_guard(
                    args, round_id, f'distill_step_{step}_post_step',
                    modules={'global_model': global_model})
                L_D_report = L_D.item()
                print(f"  step {step}: L_D={L_D.item():.4f} |grad|={gn:.4f}")

            # ---- [ServerUpdateNorm 诊断] 蒸馏对 global 模型的参数位移 ----
            s_up = math.sqrt(sum(
                ((p - pb) ** 2).sum().item()
                for p, pb in zip(global_model.parameters(), params_before)))
            s_norm = math.sqrt(sum((pb ** 2).sum().item()
                                   for pb in params_before))
            print(f"  [ServerUpdateNorm] norm={s_up:.4f} "
                  f"relative={s_up / max(s_norm, 1e-12):.4%}")

            # ---- 平台集成: 全局蒸馏结构化事件 (伪图统计 + 教师权重) ----
            fake_n = int(fake_x.shape[0])
            fake_e = int(fake_graph.edge_index.shape[1])
            _emit_event(
                args, 'distillation_metric', round_id=round_id,
                stage='global_distillation',
                distillation_loss=(round(L_D_report, 6)
                                   if L_D_report is not None else None),
                distill_grad_norm=round(distill_grad_norm, 6),
                fake_graph_nodes=fake_n,
                fake_graph_edges=fake_e,
                fake_graph_avg_degree=round(fake_e / max(1, fake_n), 4),
                fake_graph_components=_count_connected_components(
                    fake_graph.edge_index, fake_n),
                teacher_weights_per_class=_sanitize_json(
                    normalized_ckr.sum(dim=0).cpu()),
            )

            # 蒸馏诊断 (distillation_diagnostics.jsonl)
            if args.distillation_diagnostics_jsonl:
                with torch.no_grad():
                    param_delta = math.sqrt(sum(
                        (pa - pb).norm().item() ** 2
                        for pa, pb in zip(global_model.parameters(), params_before)))
                    pred_delta = 0.0
                    for ci, sg in enumerate(subgraphs):
                        lg = global_model.forward(sg)[sg.val_idx]
                        pred_delta += float((lg - logits_before[ci]).abs().mean().item())
                    pred_delta /= len(subgraphs)
                fake_deg = float(fake_graph.edge_index.shape[1]) / max(1, fake_x.shape[0])
                record = {
                    'round': round_id,
                    'distill_grad_norm': distill_grad_norm,
                    'global_param_update_norm': param_delta,
                    'global_prediction_change_val': pred_delta,
                    'distill_loss': L_D_report,
                    'generator_loss': L_G_report,
                    'fake_feature_std': float(fake_x.std().item()),
                    'fake_graph_mean_degree': fake_deg,
                    'teacher_mixture_entropy_mean': float(
                        torch.tensor([-float((normalized_ckr[:, c] *
                                              torch.log(normalized_ckr[:, c] + 1e-12)).sum())
                                      for c in range(num_classes)]).mean().item()),
                }
                os.makedirs(os.path.dirname(args.distillation_diagnostics_jsonl) or '.',
                            exist_ok=True)
                with open(args.distillation_diagnostics_jsonl, 'a') as f:
                    f.write(json.dumps(record) + '\n')

        else:
            if (args.distill_weighting != 'none'
                    and args.federated_mode == 'fedavg'):
                print(f"\n[Server Skipped] round {round_id} < "
                      f"server_start_round={args.server_start_round}: "
                      f"生成器更新与全局蒸馏均不执行 (仅客户端训练 + FedAvg)")

        # ---- [ServerDelta 诊断] 服务端蒸馏后 val ----
        if args.federated_mode == 'fedavg':
            _was_tr = global_model.training
            global_model.eval()
            with torch.no_grad():
                _va_res = evaluate_global(global_model, subgraphs,
                                          args.task_mode, split='val',
                                          device=device)
            if _was_tr:
                global_model.train()
            _va = (float(_va_res.get('accuracy', float('nan')))
                   if isinstance(_va_res, dict) else float('nan'))
        else:
            _va = float('nan')
        print(f"  [ServerDelta] val_before={_vb:.4f} val_after={_va:.4f} "
              f"server_delta={_va - _vb:+.4f}"
              + ("" if server_ran else "  [SKIPPED: server 未执行]"))

        # -----------------------------------------------------------
        #  [10-11] 校正全局模型 -> val_idx 模型选择 (最佳 checkpoint)
        # -----------------------------------------------------------
        print("\n[Evaluation]")
        if args.federated_mode == 'local_only':
            # B0: 无全局模型, 只记录每客户端 val 诊断
            primary_val = float('nan')
            primary_test = float('nan')
            val_metrics = {'accuracy': float('nan'), 'macro_f1': float('nan')}
            test_metrics = dict(val_metrics)
        else:
            global_model.eval()
            val_metrics = evaluate_global(global_model, subgraphs, args.task_mode,
                                          split='val', device=device)
            if (getattr(args, 'tuning_mode', False)
                    or getattr(args, 'final_test_only', False)):
                # 调参期/正式决赛均不做逐轮 test；决赛仅训练结束后算一次
                test_metrics = {k: float('nan') for k in val_metrics}
            else:
                test_metrics = evaluate_global(global_model, subgraphs,
                                               args.task_mode,
                                               split='test', device=device)

            primary_val = val_metrics[selection_metric]
            primary_test = test_metrics[selection_metric]

            if args.task_mode == 'multiclass':
                print(f"  [val] weighted acc={val_metrics['accuracy']:.2f} "
                      f"macro_f1={val_metrics['macro_f1']:.2f}")
                print(f"  [test] weighted acc={test_metrics['accuracy']:.2f} "
                      f"macro_f1={test_metrics['macro_f1']:.2f}")
            else:
                print(f"  [val] pooled_auc={val_metrics['pooled_auc']:.2f} "
                      f"pooled_pr_auc={val_metrics['pooled_pr_auc']:.2f} "
                      f"weighted_client_auc={val_metrics['weighted_client_auc']:.2f}")
                print(f"  [test] pooled_auc={test_metrics['pooled_auc']:.2f} "
                      f"pooled_pr_auc={test_metrics['pooled_pr_auc']:.2f} "
                      f"weighted_client_auc={test_metrics['weighted_client_auc']:.2f}")

            improved = (not math.isnan(primary_val)
                        and primary_val > best_val_primary)
            if improved:
                best_val_primary = primary_val
                best_test_primary = primary_test
                best_round = round_id
                if args.checkpoint_dir:
                    save_checkpoint(
                        args.checkpoint_dir, 'best.pt',
                        global_model, generator,
                        global_optimizer, gen_optimizer,
                        ckr_tracker=tracker, args=args,
                        round_idx=round_id, best_metric=best_val_primary,
                        task_mode=args.task_mode, num_classes=num_classes,
                        feat_dim=feat_dim,
                        normal_classes=args.normal_classes,
                        anomaly_classes=args.anomaly_classes,
                        selection_metric=selection_metric,
                        generator_cfg=generator_cfg, is_best=True,
                        pretrain_ref=pretrain_ref, radius_bank=radius_bank,
                        local_optimizers=local_optimizers)
                    print(f"  [Checkpoint] 保存 best.pt (round {round_id}, "
                          f"{selection_metric}={best_val_primary:.4f})")
                    _emit_event(args, 'checkpoint_saved', round_id=round_id,
                                stage='evaluation', checkpoint='best.pt',
                                round_idx=round_id,
                                best_metric=round(best_val_primary, 4))
            else:
                if math.isnan(primary_val):
                    print(f"  [Checkpoint] {selection_metric} 为 NaN, 不覆盖最佳模型")

            if args.checkpoint_dir and args.save_last_checkpoint:
                save_checkpoint(
                    args.checkpoint_dir, 'last.pt',
                    global_model, generator,
                    global_optimizer, gen_optimizer,
                    ckr_tracker=tracker, args=args,
                    round_idx=round_id, best_metric=best_val_primary,
                    task_mode=args.task_mode, num_classes=num_classes,
                    feat_dim=feat_dim,
                    normal_classes=args.normal_classes,
                    anomaly_classes=args.anomaly_classes,
                    selection_metric=selection_metric,
                    generator_cfg=generator_cfg, is_best=False,
                    pretrain_ref=pretrain_ref, radius_bank=radius_bank,
                    local_optimizers=local_optimizers)
                print(f"  [Checkpoint] 保存 last.pt (round {round_id})")
                _emit_event(args, 'checkpoint_saved', round_id=round_id,
                            stage='evaluation', checkpoint='last.pt',
                            round_idx=round_id)

            print(f"  [Global] {selection_metric}={primary_val:.2f} "
                  f"(best={best_val_primary:.2f} @ round {best_round})")

            # ---- 平台集成: 全局指标结构化事件 (仅供报告, 不参与训练) ----
            _emit_event(
                args, 'global_metric', round_id=round_id, stage='evaluation',
                selection_metric=selection_metric,
                global_val=round(primary_val, 6),
                global_test=(round(primary_test, 6)
                             if not math.isnan(primary_test) else None),
                best_val=round(best_val_primary, 6),
                best_test=(round(best_test_primary, 6)
                           if not math.isnan(best_test_primary) else None),
                best_round=best_round,
                val_metrics=_sanitize_json(val_metrics),
                test_metrics=_sanitize_json(test_metrics),
            )

        # ---- 每轮训练行为诊断 (metrics_jsonl) ----
        round_time = time.time() - t_round0
        round_times.append(round_time)
        gen_peak_mb_all.append(gen_peak_mb)
        round_metrics = {
            'round': round_id,
            'seed': args.seed,
            'ce_loss': ce_avg,
            'contrastive_loss': (cl_avg
                                 if args.contrastive_mode == 'subgraph_cross_view' else None),
            'generator_loss': L_G_report,
            'distill_loss': L_D_report,
            'val_primary': (None if math.isnan(primary_val) else primary_val),
            'best_val': best_val_primary,
            'rwr_hit_rate': rwr_cache.hit_rate,
            'gen_peak_mem_mb': gen_peak_mb,
            'fallback_ratio': (float(tracker.last_fallback.float().mean().item())
                               if per_client_metrics else None),
            'round_time_sec': round_time,
        }
        if args.metrics_jsonl:
            with open(args.metrics_jsonl, 'a') as f:
                f.write(json.dumps(round_metrics) + '\n')

        # ---- 平台集成: 每轮资源使用事件 ----
        _emit_event(
            args, 'resource_usage', round_id=round_id, stage='round_finished',
            round_time_sec=round(round_time, 3),
            generator_peak_gpu_mb=round(gen_peak_mb, 1),
            rwr_cache_hit_rate=round(rwr_cache.hit_rate, 4),
            rwr_cache_stats=_sanitize_json(rwr_cache.stats()),
        )

        # 终止 (仅 fedavg): 主指标连续三轮提升小于阈值
        if args.federated_mode == 'fedavg':
            primary_history.append(primary_val)
            if len(primary_history) >= 4:
                gains = [primary_history[i] - primary_history[i-1]
                         for i in range(len(primary_history)-3, len(primary_history))]
                if all(g < args.f1_threshold for g in gains):
                    print(f"[Termination] 主指标连续三轮提升 < {args.f1_threshold}, 正常收敛")
                    break

        # 低资源客户端检查
        if args.federated_mode == 'fedavg' and round_id >= 3:
            ckr_mean = normalized_ckr.mean(dim=1)
            _, bottom_idx = torch.topk(ckr_mean, k=max(1, args.num_clients // 10), largest=False)
            all_trig = True
            for ci in bottom_idx.tolist():
                ci = int(ci)
                em = evaluate_client(local_models[ci], subgraphs[ci], args.task_mode,
                                     eval_mask=subgraphs[ci].val_idx)
                if args.task_mode == 'anomaly_binary':
                    m = em.get('roc_auc', float('nan'))
                else:
                    m = em['accuracy']
                if not hasattr(main, '_low_resource_log'):
                    main._low_resource_log = {}
                if ci not in main._low_resource_log:
                    main._low_resource_log[ci] = []
                main._low_resource_log[ci].append(m)
                if len(main._low_resource_log[ci]) >= 3:
                    gain = main._low_resource_log[ci][-1] - main._low_resource_log[ci][-2]
                    if math.isnan(m) or gain >= args.auc_threshold:
                        all_trig = False
                else:
                    all_trig = False
            if all_trig:
                print(f"[Termination] 低资源客户端指标连续两轮改善 < {args.auc_threshold}")
                break

    # =================================================================
    #  训练结束: 默认加载 best.pt 再做最终 test (只做一次正式 final test)
    # =================================================================
    print(f"\n{'=' * 60}")
    print(f"训练结束. 最佳 round={best_round}, "
          f"best_val({selection_metric})={best_val_primary:.4f}, "
          f"best_test({selection_metric})={best_test_primary:.4f}")

    loaded_from = None
    if args.federated_mode == 'local_only':
        # B0: 每客户端本地模型 (local ensemble over disjoint client test sets,
        #     不合并模型参数); pooled 由各客户端本地模型对各自 test 的预测拼接得到
        print("[Final Test] local_only: local ensemble over disjoint client test sets")
        for ci in range(args.num_clients):
            local_models[ci].eval()
        final_eval = local_models
        eval_scope = 'local_ensemble'
    else:
        final_eval = global_model
        eval_scope = 'global_model'
        if args.checkpoint_dir and find_final_checkpoint(args.checkpoint_dir):
            final_path = find_final_checkpoint(args.checkpoint_dir)
            ckpt = load_checkpoint(final_path, global_model=global_model,
                                   expected_meta={'task_mode': args.task_mode,
                                                  'num_classes': num_classes,
                                                  'feat_dim': feat_dim},
                                   restore_rng=False, map_location=str(device))
            loaded_from = os.path.basename(final_path)
            print(f"[Final Test] 加载 {loaded_from} (round {ckpt['round']})")
        else:
            print("[Final Test] 无 checkpoint, 使用当前全局模型 (仅诊断)")
        global_model.eval()

    if getattr(args, 'tuning_mode', False):
        # test isolation: 调参模式不计算最终 test (objective 只读 validation);
        # 结构镜像 collect_final_metrics 的 multiclass 形状, 全部 NaN
        _nan = {'accuracy': float('nan'), 'macro_f1': float('nan')}
        final_test = {
            'pooled': dict(_nan), 'client_macro': dict(_nan),
            'client_weighted': dict(_nan), 'fedtad_official': dict(_nan),
            'per_client': {}, 'num_test_samples': 0,
            'evaluation_scope': eval_scope,
            'aggregation_levels': ['pooled', 'client_macro',
                                   'client_weighted', 'fedtad_official'],
            'valid_clients': {'all': True},
            'metric_comparability_group': 'tuning|test-not-evaluated'}
        print("[Final Test] [tuning] test 不计算 (test isolation)")
    else:
        final_test = collect_final_metrics(final_eval, subgraphs, args.task_mode,
                                           split='test', evaluation_scope=eval_scope)
    if args.task_mode == 'multiclass':
        print(f"  [Final test ({eval_scope})] pooled acc={final_test['pooled']['accuracy']:.2f} "
              f"macro_f1={final_test['pooled']['macro_f1']:.2f} | "
              f"client_macro macro_f1={final_test['client_macro']['macro_f1']:.2f} | "
              f"client_weighted macro_f1={final_test['client_weighted']['macro_f1']:.2f}")
    else:
        print(f"  [Final test ({eval_scope})] pooled roc_auc={final_test['pooled']['roc_auc']:.2f} "
              f"pr_auc={final_test['pooled']['pr_auc']:.2f} | "
              f"client_macro pr_auc={final_test['client_macro']['pr_auc']:.2f} | "
              f"client_weighted pr_auc={final_test['client_weighted']['pr_auc']:.2f} "
              f"(valid {final_test['valid_clients']['roc_auc']}/"
              f"{final_test['valid_clients']['total']})")

    # ---- 最终指标结构化输出 (含 seed 与配置哈希) ----
    h = config_hash(args)
    final_metrics = {
        'seed': args.seed,
        'config_hash': h,
        'task_mode': args.task_mode,
        'selection_metric': selection_metric,
        'best_round': best_round,
        'best_val_primary': best_val_primary,
        'loaded_from': loaded_from,
        'metrics': final_test,
        'zero_anomaly_clients': (split_report_data.get('zero_anomaly_clients', [])
                                 if 'split_report_data' in dir() else []),
    }
    if args.final_metrics_json:
        os.makedirs(os.path.dirname(args.final_metrics_json) or '.', exist_ok=True)
        with open(args.final_metrics_json, 'w') as f:
            json.dump(_sanitize_json(final_metrics), f, indent=1)
        print(f"[Final Metrics] 写入 {args.final_metrics_json}")

    # ---- 统一评估报告 (evaluation_report.json) ----
    if args.evaluation_report_json:
        eval_report = {
            'seed': args.seed,
            'config_hash': h,
            'task_mode': args.task_mode,
            'selection_metric': selection_metric,
            'split': 'test',
            'evaluation_scope': final_test.get('evaluation_scope'),
            'metric_comparability_group': final_test.get('metric_comparability_group'),
            'aggregation_levels': final_test.get('aggregation_levels'),
            'num_test_samples': final_test.get('num_test_samples'),
            'num_available_clients': (final_test.get('valid_clients', {})
                                      .get('roc_auc', 0) if args.task_mode == 'anomaly_binary'
                                      else final_test.get('valid_clients', {}).get('total', 0)),
            'num_unavailable_clients': (final_test.get('valid_clients', {}).get('total', 0)
                                        - final_test.get('valid_clients', {}).get('roc_auc', 0)
                                        if args.task_mode == 'anomaly_binary'
                                        else 0),
            'pooled': final_test.get('pooled'),
            'client_macro': final_test.get('client_macro'),
            'client_weighted': final_test.get('client_weighted'),
        }
        os.makedirs(os.path.dirname(args.evaluation_report_json) or '.', exist_ok=True)
        with open(args.evaluation_report_json, 'w') as f:
            json.dump(_sanitize_json(eval_report), f, indent=1)
        print(f"[Evaluation Report] 写入 {args.evaluation_report_json}")

    # ---- CKR 可观测性聚合 (ckr_availability.json) ----
    if args.ckr_availability_json:
        ckr_avail = {
            'seed': args.seed,
            'config_hash': h,
            'ckr_mode': args.ckr_mode,
            'rounds': tracker.aggregate_history,
        }
        os.makedirs(os.path.dirname(args.ckr_availability_json) or '.', exist_ok=True)
        with open(args.ckr_availability_json, 'w') as f:
            json.dump(_sanitize_json(ckr_avail), f, indent=1)
        print(f"[CKR Availability] 写入 {args.ckr_availability_json}")

    # ---- RWR 缓存指标 (cache_metrics.json) ----
    if args.cache_metrics_json:
        cache_metrics = {
            'seed': args.seed,
            'config_hash': h,
            'mode': args.rwr_cache_mode,
            'scope': args.rwr_cache_scope,
            'anchor_sampling_mode': args.anchor_sampling_mode,
            'local_epochs': args.num_epochs,
            'stats': rwr_cache.stats(),
        }
        os.makedirs(os.path.dirname(args.cache_metrics_json) or '.', exist_ok=True)
        with open(args.cache_metrics_json, 'w') as f:
            json.dump(cache_metrics, f, indent=1)
        print(f"[Cache Metrics] 写入 {args.cache_metrics_json}")

    # ---- 公平性 / 低资源分析 (fairness_metrics.json) ----
    if args.fairness_metrics_json:
        fair = _build_fairness_metrics(final_test, subgraphs, num_classes,
                                       args.task_mode, tracker)
        os.makedirs(os.path.dirname(args.fairness_metrics_json) or '.', exist_ok=True)
        with open(args.fairness_metrics_json, 'w') as f:
            json.dump(_sanitize_json(fair), f, indent=1)
        print(f"[Fairness] 写入 {args.fairness_metrics_json}")

    # ---- 资源开销 ----
    if args.resource_usage_json:
        res = {
            'seed': args.seed,
            'config_hash': h,
            'device': str(device),
            'total_wall_sec': time.time() - t_total0,
            'round_times_sec': round_times,
            'gen_peak_mem_mb_max': max(gen_peak_mb_all) if gen_peak_mb_all else None,
            'rwr_cache_hit_rate_final': rwr_cache.hit_rate,
            'rwr_cache_stats': rwr_cache.stats(),
        }
        os.makedirs(os.path.dirname(args.resource_usage_json) or '.', exist_ok=True)
        with open(args.resource_usage_json, 'w') as f:
            json.dump(res, f, indent=1)
        print(f"[Resource] 写入 {args.resource_usage_json}")

    # ---- 平台集成: 任务完成事件 ----
    _emit_event(
        args, 'task_finished', stage='finished',
        best_round=best_round,
        best_val=(round(best_val_primary, 6)
                  if not math.isnan(best_val_primary) else None),
        best_test=(round(best_test_primary, 6)
                   if not math.isnan(best_test_primary) else None),
        loaded_from=loaded_from,
        final_pooled=_sanitize_json(final_test.get('pooled', {})),
    )
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            _emit_event(args, 'task_failed', stage='failed',
                        error=str(e)[:2000],
                        error_type=type(e).__name__)
        except Exception:
            pass
        sys.exit(1)
