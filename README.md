# FedTAD — Formal V3 正式实验代码（FedTAD 改进方法）

> **⚠️ 本仓库根目录 = Formal Campaign V3 的 canonical 正式实验代码。**
> 论文引用请使用冻结 tag **`formal-v3-f38f7fe74ab5`**，不要用 `main`。
> 旧 anomaly/Web 平台开发线已隔离至 [`legacy/`](legacy/README.md)，与正式结果无关。

## Formal Campaign V3 identity

| 项 | 值 |
|---|---|
| campaign | **v3** |
| formal code hash | **`f38f7fe74ab5`** |
| search space | v2 |
| health | v2 |

完整算法身份与协议见 [FORMAL_V3_VERSION.md](FORMAL_V3_VERSION.md)。

方法 = **FedTAD**（IJCAI 2024, arXiv:2404.14061）+ 两个自研组件：

1. **客户端**：子图-子图跨视图对比学习（InfoNCE，denominator 含 positive）
2. **服务端**：条件 DDPM 伪图生成 + CKR 加权 KL 无数据知识蒸馏

Formal V3 关键机制：weighted CE · subgraph_cross_view InfoNCE ·
component-safe RWR · conditional DDPM（T=20, beta_end=0.5, posterior variance,
timestep-scalar residual）· **KL(global ‖ local)** · overflow-safe radius projection ·
validation-only tuning · final-test-only · study n_jobs=1。

目标：在 Cora / CiteSeer / PubMed / CS / Physics × 5/10/20 客户端三档上与
FedTAD 论文 Table 2 对比（论文数字见 `audit_docs/PAPER_TABLE2.md`）。

## 正式结果

| 数据集 | 5 clients | 10 clients | 20 clients |
|---|---|---|---|
| Cora | 84.26 ± 0.23 | 76.84 ± 0.25 | 67.89 ± 1.03 |
| CiteSeer | 71.12 ± 0.54 | 73.21 ± 0.33 | 70.10 ± 0.26 |
| PubMed / CS / Physics | campaign 进行中 | | |

> 口径：test accuracy（fedtad_official 聚合），3 种子 {2024,2025,2026}，
> 每 (数据集,档位) 独立 Optuna 调参（目标=validation only），
> 结果产物在 `runs/formal_campaign_v3/`（不入库）。
> 数字以各 dataset 的 `<DS>_V3_FINAL_SUMMARY.json/.md` 为准。

## 仓库结构

```
train_fedtad.py            # 主训练脚本（canonical 入口）
model.py                   # ConditionalDiffusionGenerator
util/                      # 训练管线（含 kl_utils / projection_utils / task_util）
experiments/accuracy_benchmark/   # formal_campaign.py（正式战役 runner）+ 总结脚本
scripts/                   # server smoke / Stage1 / cell / dataset / remaining launchers
tests/                     # V3 数学与协议回归测试
audit_docs/                # 管线审计 / 协议 / 论文主表（PAPER_TABLE2.md）
louvain/                   # vendored Louvain（partition 重建依赖）
FORMAL_V3_VERSION.md       # V3 身份标记（hash / 协议 / 不变式）
SERVER_MIGRATION_BEGINNER.md  # 服务器部署与环境复现手册
legacy/                    # ⚠️ 旧 anomaly/platform 开发线，不使用、不复现
```

## 复现（Formal V3）

```bash
git checkout formal-v3-f38f7fe74ab5

# 1. 环境（torch 2.7.1+cu128 / pyg 2.6.1 / scatter+sparse+cluster 冻结版本）
conda env create -f environment.yml -n fedtad5060

# 2. 强制 smoke gate（必须看到 ALL REQUIRED CHECKS PASSED 才能跑正式实验）
bash scripts/server_smoke_test.sh

# 3. 单 cell（Stage1 -> 12 healthy Optuna trials -> 3-seed final）
bash scripts/generate_stage1_cell.sh Cora 5
bash scripts/run_formal_cell.sh Cora 5 12 20 1
bash scripts/run_formal_final.sh Cora 5

# 4. 单数据集（3 档跨 cell 并行, MAX_CELL_JOBS=3）
MAX_CELL_JOBS=3 bash scripts/run_dataset_v3.sh Cora 12

# 5. 剩余数据集链式（CiteSeer -> PubMed -> CS -> Physics, 逐数据集自动推进）
MAX_CELL_JOBS=3 bash scripts/run_remaining_v3.sh
```

详细部署/数据/断点恢复见 [SERVER_MIGRATION_BEGINNER.md](SERVER_MIGRATION_BEGINNER.md)
与 [audit_docs/PROTOCOL.md](audit_docs/PROTOCOL.md)。

## 关键参数速览（与代码默认值一致）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task_mode` | anomaly_binary | 历史默认；**正式 campaign 固定 multiclass** |
| `--diffusion_steps` | 20 | Formal V3 固定 |
| `--diffusion_beta_end` | 0.5 | Formal V3 固定 |
| `--use_weighted_ce` | 关 | 正式 multiclass campaign 显式 ON |
| `--contrastive_mode` | subgraph_cross_view | 核心客户端创新 |
| `--local_optimizer_lifecycle` | reset_each_round | 正式协议固定 |
| `--tuning_mode` / `--final_test_only` | 关 | 正式 campaign 必须使用（test 隔离） |
| `--use_posterior_variance` | ON | Formal V3 固定 |
| `--diffusion_skip_mode` | timestep_scalar | Formal V3 固定 |
| `--lambda_diffusion_anchor` | 1e-2 | Formal V3 固定 |

完整参数见 `train_fedtad.py --help`；正式协议固定项见
`experiments/accuracy_benchmark/formal_campaign.py::FIXED_FLAGS`。
