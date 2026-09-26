# Formal Campaign V3 — Version Marker

> 本文档是 Formal V3 正式实验的**算法身份标记**。正式结果只能由满足下述全部
> 条件的代码生成；任何一条不满足即不是 Formal V3。

## Identity

| 项 | 值 |
|---|---|
| campaign | v3 |
| formal code hash | `f38f7fe74ab5` |
| formal search space | v2 |
| formal health | v2 |
| 冻结 tag | `formal-v3-f38f7fe74ab5` |

> ⚠️ Git commit SHA 与 formal code hash 不是一回事。formal code hash 由
> `experiments/accuracy_benchmark/formal_campaign.py::code_hash()` 按固定文件清单
> （训练数学 + 数据/CKR + runner/health/protocol + 关键脚本）计算。

## Core protocol

- Conditional DDPM：T=20，beta_end=0.5，posterior variance ON，
  timestep-scalar residual ON，Stage2 timestep scalar **frozen**；
- federated diffusion Stage1 pretraining（真实特征只在客户端本地使用，
  只上传去噪参数）；
- radius constraint ON（overflow-safe projection：数学等价于 r·z/‖z‖，
  数值上先逐行 max 重标定避免 float32 平方和溢出）；
- feature_stats_align OFF；
- L2-SP anchor λ=1e-2；
- Stage2 generator optimizer：fresh（不恢复 Stage1 Adam moments），weight_decay=0；
- theta_pre = Stage1 checkpoint load 后重新 snapshot；
- local optimizer lifecycle = `reset_each_round`（每轮广播后重建，不继承 stale Adam moments）；
- weighted CE ON；
- 客户端子图-子图跨视图 InfoNCE（`subgraph_cross_view`），positive 包含在 denominator；
- component-safe RWR：局部子图不足时只在 anchor 连通分量内 BFS 扩展，
  禁止跨分量随机补点；
- edge perturbation 以无向边为单位（删/补都保持成对无向）；
- KL(global || local)：FedTAD Eq.(10) / 专利 S4.2 方向；
  generator **maximizes** 该 divergence，global/student **minimizes** 同一 divergence；
  禁止 KL(local || global)，禁止 L1 替代；
- CKR-guided data-free KD；pseudo graph = cosine + KNN；
- Optuna：TPESampler(seed=2024)，每 study n_jobs=1，无 pruner；
  目标 = 12 个 health=PASS 的完整 100 轮 trial 中 best **validation** accuracy；
  max attempts = 20；
- tuning 全程 `--tuning_mode`：test evaluation = 0；
- final：seed {2024,2025,2026}，每 seed 100 轮 validation 选 best checkpoint，
  重新 load 后 test **exactly once**（`--final_test_only`）；
- formal health v2：loss/grad/参数/raw generator 输出/projected 输出/
  raw radius/projected radius/distillation 任一 NaN/Inf → trial FAIL，
  projected finite 不豁免 raw nonfinite。

## 正式结果

Cora 与 CiteSeer（以及后续数据集）的正式结果均在上述 identity 下、
在冻结代码（tag `formal-v3-f38f7fe74ab5`）上生成。结果数字见
`runs/formal_campaign_v3/<DS>/<DS>_V3_FINAL_SUMMARY.*`（runs/ 不入库，
只存在于实验服务器与本地 runs 目录）。

## 复现

1. `git checkout formal-v3-f38f7fe74ab5`
2. 部署/环境/数据/流程见 `SERVER_MIGRATION_BEGINNER.md` 与
   `audit_docs/PROTOCOL.md`
3. 启动前必须通过 `bash scripts/server_smoke_test.sh`
   （`SERVER V3 SMOKE TEST: ALL REQUIRED CHECKS PASSED`）
