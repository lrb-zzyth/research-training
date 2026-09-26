# 交接文档 —— FedTAD 对比实验项目

> 生成于 2026-09-24 23:25 · 交接给新的模型/会话
> **先读本文档，再读记忆文件与下方索引，可无缝接手。**

---

## 0. 一句话现状

在 `/home/lrb/federated/FedTAD` 上做一个「**在 FedTAD 基础上做到全面超越 FedTAD**」的实验项目。
**实验队列正在后台自动运行**（已脱离终端，切换会话不会中断）。
当前 Cora / CiteSeer 两数据集完成，**vs FedAvg 全胜，vs FedTAD 3 胜 3 负**。

---

## 1. 任务目标（用户原话）

> "核心就是参考我给你的专利申请书的思路，在 FedTAD 的基础上实现我的方法，
> 我需要它是**真真正正地有提升**，然后给我**总表**等物。记得我的硬件条件不行。"

- **我的方法** = FedTAD + ①客户端子图跨视图对比学习 + ②服务器端条件 DDPM 生成器
- **目标** = 在 Cora / CiteSeer / PubMed / CS / Physics 五个数据集 × **5/10/20 客户端**三档上
  **全面超越 FedTAD 论文数字**（arXiv:2404.14061 Table 2）
- **只跑我的方法**；FedTAD 及其余基线一律**引用论文 Table 2**，不自己跑

---

## 2. ⚠️ 实验队列（正在跑，不要动）

```bash
# 查看进度
tail -f runs/accuracy_benchmark/TUNE3_RUN.log

# 查看当前在跑哪个数据集
ps -eo args | grep "^/home/lrb.*train_fedtad.py" | head -1

# 已完成的结果
ls runs/accuracy_benchmark/TUNED_*/seed_*/final_metrics.json
```

**队列内容**：`experiments/accuracy_benchmark/tune_then_run.py`
对 5 个数据集逐个执行「Optuna 调参（目标=验证集精度）→ 用最优配置在 5/10/20 档各跑 3 种子」。

| 数据集 | 调参预算 | 单 run 耗时 | 状态 |
|---|---|---|---|
| Cora | 20 trials | ~10 min | ✅ 完成 |
| CiteSeer | 20 trials | ~10 min | ✅ 完成 |
| CS | 16 trials | ~40 min | 🔄 进行中 |
| PubMed | 16 trials | ~26 min | ⏳ |
| Physics | 10 trials | ~97 min | ⏳ |

**总计还需约 35~40 小时。**

### 🔴 硬件硬约束（用户反复强调）

**7GB 内存 / 8GB 显存。单个 run 峰值 3.6GB。必须严格串行，绝对不能并行多个训练进程。**
（历史上曾因 6 进程并行 OOM，整个 WSL 被打挂。）
所有脚本都内置了「检测到第二个训练进程就 ABORT」+「可用内存 <3GB 就 ABORT」的安全闸。

---

## 3. 当前结果（3 种子，固定种子集 {2024,2025,2026}）

| 数据集 | 档 | 我的方法 | 论文 FedTAD | Δ | 论文 FedAvg | Δ |
|---|---|---|---|---|---|---|
| Cora | 5 | 84.87±0.28 | 85.1 | **−0.23** ❌ | 80.6 | +4.27 |
| Cora | 10 | 77.66±0.45 | 75.3 | **+2.36** ✅ | 73.6 | +4.06 |
| Cora | 20 | 69.00±0.80 | 61.3 | **+7.70** ✅ | 56.0 | +13.00 |
| CiteSeer | 5 | 72.33±0.15 | 73.5 | **−1.17** ❌ | 71.5 | +0.83 |
| CiteSeer | 10 | 72.90±0.87 | 71.7 | **+1.20** ✅ | 68.9 | +4.00 |
| CiteSeer | 20 | 70.01±0.16 | 70.2 | **−0.19** ❌ | 66.3 | +3.71 |

**规律**：vs FedAvg **6/6 全胜**；vs FedTAD **3 胜 3 负**，但赢的幅度大（+1.2~+7.7）、输的幅度小（−0.19~−1.17）。

**逐数据集调参是必需的**——搜出的最优参数几乎处处相反：

| | Cora | CiteSeer |
|---|---|---|
| `distill_steps` | **25** | **1** |
| `generator_steps` | 3 | 5 |
| `contrastive_temperature` | 0.35 | 0.05 |
| `knn_k` | 10 | 3 |
| `fake_nodes` | 200 | 100 |

---

## 4. 核心技术问题（新模型最该看的）

详见 `PROMPT_for_review.md`（这是为让外部模型审查代码而写的自包含 prompt）。
要点：

### 问题 A：生成器「最大化分歧」的目标可被廉价作弊

```
L_G = λ_sem·L_sem + λ_div·L_div + λ_norm·L_norm − λ_dis·L_dis
                                                    ↑ 生成器最大化这一项
```

`L_dis` 作用在模型输出上 → **只要放大伪特征的输出幅度，logits 差就变大，分歧自然变大**，
不需要真的产出有信息量的节点。

实测：伪特征 σ 首轮**恒为 0.655 与数据集无关**（tanh 输出界的特征），而真实 σ 为
0.112 / 0.092 / **0.018**——差 6~48 倍。

**核心矛盾**：
- 真实尺度 → 教师轻易分类且互相一致 → **分歧归零，蒸馏无信号**
- 放大 → 分歧大但源于 OOD、logits 失真 → **假信号**

### 问题 B：扩散网络此前从未被训练过去噪（已修，但未根治）

原实现有 DDPM 结构但**没有前向加噪、没有噪声预测损失**，只被 L_sem/L_dis/L_div 训练。

**已修**：
1. 噪声调度：`diffusion_steps` 10→20、`diffusion_beta_end` 0.02→0.5，
   使 `alpha_bar_T` 从 **0.904 → 0.002**（原值下 x_T 残留 90% 信号，噪声预测无解）
2. 新增**联邦扩散预训练**（`federated_diffusion_pretrain`，默认开）：
   客户端本地对真实特征做前向加噪+噪声预测，**只上传去噪网络参数**，数据不出域

**未根治**：预训练只是初始化，后续 `L_dis` 的作弊路径**又把尺度推回去了**。

### 已失败的三种修复（新模型请勿重走）

| 尝试 | 结果 |
|---|---|
| 去掉 tanh 输出界 | 生成器收敛了（L_sem 80.7→0.225），但 σ 涨到 1.13，准确率反而降 1.24 |
| 逐维 z-score 重标定到真实统计 | σ 精确对齐（0.0916 vs 真实 0.092），但**梯度塌缩**（L_sem/L_dis/|grad| 全归零） |
| 全局标量重标定 | **仍塌缩**，准确率 71.53 |

---

## 5. 🚫 专利硬约束（绝对不可违反）

**不得为了"对齐 FedTAD 原版"或"提高分数"而修改以下技术特征：**

| 权利要求 | 内容 | 代码位置 |
|---|---|---|
| 权1 | CKR 客户端本地计算，**仅上传分值**，不上传原始数据 | `compute_ckr` |
| 权3 / S2 | 子图-子图跨视图对比，InfoNCE，**分母必须含正样本自身** | `util/task_util.py:233` |
| 权4 | 客户端类别加权 CE | `--use_weighted_ce` |
| 权5 / S3 | 服务器端**条件 DDPM** 生成伪图 + **KNN** 构图 | `model.py:56` |
| 权6 / S4.2 | 散度损失用 **KL 散度**，衡量**预测分布**差异，CKR 加权 | `compute_student_distillation_loss` |
| 权7 | 双重终止机制 | `--f1_threshold` / `--auc_threshold` |
| S3.3 | 伪拓扑 = **余弦相似度** + **K 近邻** | `build_knn_fake_graph` |

### ⚠️ 两条防呆警告（前任模型在这里差点犯错）

1. **不要把 KL 改成 L1。** FedTAD **论文 Eq.10 写的是 KL**（原文：
   "where KL(·||·) denotes the Kullback-Leibler divergence function"），
   但**上游代码实现的是 L1**（`torch.abs(global_pred - local_pred.detach())`）——**论文与代码自相矛盾**。
   我们的专利站在论文这边。实测 KL 版本比 L1 高约 1 分。
2. **不要把扩散生成器换成 MLP。** 那是原版 FedTAD 的做法，扩散是创新点。

---

## 6. 实验硬性要求

1. 主指标 = **test accuracy (%)**，与论文主表口径一致
2. **验证集选轮**，严禁用测试集选参或选轮
3. 每个数据集单独调参（Optuna，目标=验证集精度）
4. **3 种子**（对齐论文），固定 {2024,2025,2026}，**不按结果挑选**
5. **诚实性**：负结果如实报；不许挑种子、不许只报最好一轮、不许把引用数字伪装成自跑
6. 所有数字可追溯到 `runs/` 下的产物

---

## 7. 文件索引

| 文件 | 作用 |
|---|---|
| `runs/accuracy_benchmark/HANDOFF.md` | **本文档** |
| `runs/accuracy_benchmark/PROMPT_for_review.md` | 给外部模型审查代码的自包含 prompt |
| `runs/accuracy_benchmark/AUDIT_vs_original.md` | **17 环节管线审计**（vs 原版 FedTAD） |
| `runs/accuracy_benchmark/PROTOCOL.md` | 实验协议（种子/轮数/选轮口径/划分 artifact） |
| `runs/accuracy_benchmark/REPORT.md` | 阶段报告 |
| `/home/lrb/.claude/projects/-home-lrb-federated-FedTAD/memory/` | **记忆文件（新会话自动加载）** |
| `experiments/accuracy_benchmark/tune_then_run.py` | **正在跑的调参+重跑队列** |
| `experiments/accuracy_benchmark/summarize.py` | 结果汇总（含配对显著性） |
| `experiments/accuracy_benchmark/make_main_table.py` | 主表生成 |

---

## 8. 关键代码位置

```
train_fedtad.py                       3259 行   主训练脚本
  :1273  federated_diffusion_pretrain()     联邦扩散预训练
  :1559  compute_generator_semantic_loss()   语义损失 L_sem
  :1584  compute_generator_disagreement_loss() 分歧损失（生成器最大化）
  :1624  compute_student_distillation_loss()   蒸馏损失（全局模型最小化，KL）
  compute_ckr() / build_knn_fake_graph()

model.py                              304 行   条件扩散生成器
  :56   class ConditionalDiffusionGenerator
  :128  denoise_loss()                       标准 DDPM 去噪损失

util/task_util.py                     376 行
  :233  subgraph_contrastive_loss()          InfoNCE（专利 S2.5 形式）

util/base_data_util.py                436 行
  :40   louvain_partition()                  客户端划分（与作者 Cora 版逐位一致）
  :361  construct_subgraph_dict_from_node_dict()
```

---

## 9. 已核验为「与原版等价」的环节（勿重复调查）

数据加载无归一化 / 图构建 `to_networkx(undirected, rm_self_loops)` 逐行相同 /
Louvain 划分（我们的 `random_state=2024` 与原版全局 RNG 等价，且 Cora 重建与作者发布版**逐位一致**）/
聚合权重 `n_k/Σn_k ≡ n_k/N_global` / 诱导子图与索引映射逐行相同 / 类内分层切分逐行相同。

**已修复的三处管线差异**（详见 AUDIT 文档）：
1. 训练集：原只用 `train_idx` 的 80% → 已改为 100%（`--reliability_holdout_ratio 0`）
2. 服务端蒸馏迭代：搜索空间上限 10 → 加入 25（原版 = glb_epochs5×it_d5）
3. 评估聚合：新增 `fedtad_official` 层级，复刻原版按客户端总节点数加权的公式

---

## 10. 无法消除的差异（如实记录）

作者仓库**只发布了 Cora 的子图**。CiteSeer / PubMed / CS / Physics 的 5/10/20 划分
**全部是用同一套算法自行生成的**（同 seed 2024、同 delta、同分配算法），
故这四个数据集的**跨客户端档位趋势与论文不可直接比较**
（实测 CiteSeer 出现 5 档 < 10 档，与论文递降趋势相反；Cora 则正常递降）。

---

## 11. 2026-09-25 凌晨更新（新会话接手注意）

### 11.1 任务 A 已完成：model.py 类注释已修正（只改注释）

`ConditionalDiffusionGenerator` 的注释已改为准确的**两阶段算法**描述：
联邦条件 DDPM 预训练（客户端本地前向加噪+噪声预测, 只上传去噪参数, 默认开启）
→ 教师引导对抗精调（L_sem/L_dis/L_div）→ KNN 伪图 → CKR 加权 KL 蒸馏。
逻辑零改动, `import model` 已验证。**待提交时一起 commit（勿推送）。**

### 11.2 edge_perturbation 无向性修复：定稿存档, 队列跑完再应用

- 完整方案 + 决策记录: **`PENDING_edge_perturbation_fix.md`**（含最终代码、验证步骤）
- 用户拍板: **不要动队列**; 队列全部跑完后应用修复, 先只重跑 Cora 探测改前/改后,
  有效再推广全 5 数据集。
- 诚实告知: 该修复是正确性修复（第二视图被削成部分有向图）, **不是提分开关**,
  预期分数小幅变化（±1 分内甚至噪声级）。想提分靠调参与审计, 不靠它。

### 11.3 论文 Table 2 全表已核验: `PAPER_TABLE2.md`

CS/PubMed/Physics 的论文数字已从 `references/fedtad.pdf` 原文提取。
关键待比目标: CS = 94.3/90.2/88.7, PubMed = 87.9/84.4/83.5, Physics = 96.2/94.1/93.3。

### 11.4 「10 档 > 5 档」与 5 档两个负分的原因（已定位, 勿重复调查）

1. **调参档位效应**: `tune_then_run.py:37` `TUNE_TIER = 10` —— 只在 10 档调参,
   5/20 档直接复用 10 档最优超参。恰好 10 档在两数据集都赢。
2. **垃圾袋客户端**: 均衡分配把切块残渣打包进个别客户端（跨种子稳定, 非噪声）:
   - CiteSeer-c5 client4: 301 个连通分量 / 密度 0.0017（其余客户端 4~91 分量）→ test 63.7,
     以 1/5 权重拉低整档 2.15 点。剔除后 CiteSeer-c5 ≈ 74.4 > 论文 73.5。
   - Cora-c5 client1: 38 分量 + 类别 3 占 43% → test 71.3, 拉低 3.35 点。剔除后 ≈ 88.2 > 论文 85.1。
   - 10 档也有弱客户端（CiteSeer-c10 client9: 156 分量 → 63.8）, 但每个只占 1/10 权重, 拖累小。
3. 行动含义: 5 档的负分**不是方法失效**, 是「未针对 5 档调参 + 单个垃圾袋客户端」的叠加。

### 11.5 用户已拍板: 每档单独调参 + 修复后全部重跑（2026-09-25 凌晨, 最终版）

- **新协议**: 15 个 (数据集 × 5/10/20 档) 组合各自独立 Optuna 调参（目标=验证集）→ 本档位 3 种子决赛。
- **旧队列 (TUNE3) 已终止**（2026-09-25 00:06, 用户批准; CS 已跑 9 个调参 trial 保留在
  optuna_tune.db 的 tune4 研究里, 只作参考）。
- **edge_perturbation 无向性修复已应用**（util/task_util.py, 2026-09-25）并通过:
  单元检查(删/补计数、成对性、确定性、真实客户端图) + test_smoke.py 26 项(CPU) + test_smoke_train.py。
  Task C 同步完成: --edge_perturb_ratio help 文本 + 函数 docstring + AUDIT D-4。
- **跑序**: 已调好参跑好的档位（仅 Cora c10 / CiteSeer c10, 旧协议下在自己档位调的参）放**最后**;
  其余 13 组合先行。驱动脚本 `experiments/accuracy_benchmark/per_tier_campaign.py`
  （CELL_ORDER; study 名 tune5_* 在 optuna_tune5.db; 输出 TUNED5_<ds>_c<tier>/seed_<s>,
  不覆盖旧 TUNED_* 参考）。
- 预计总时长 ≈ 6.2 天。日志: `runs/accuracy_benchmark/TUNE5_RUN.log`。
- **旧 TUNED_* 结果全部降级为「修复前参考」**, 最终表一律用 TUNED5_* 数字。
