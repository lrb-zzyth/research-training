# 阶段三报告：动态 CKR 独立机制复现、蒸馏失效诊断、最差客户端优化与跨数据集

日期：2026-08-01（数据填充进行中）
git commit：见各 runs/*/config.json

## 1. 阶段二完整性审计结论

1. **种子保存**：阶段二 config.json 只有单一 `seed`，partition/allocation/split/model
   种子未分别保存 → 四类随机过程完全混杂（本阶段已拆分，见 §3）。
2. **M7/M8 配对严格性**：kwargs 仅差 ckr_mode ✓；但**同一 partition（Louvain 一次抽取）
   × 10 个 model seeds**——分区级稳定性完全未知；且 enriched 划分与 natural 的
   Louvain 随机种子不同（阶段二修复引入的确定性种子），存在混杂。
3. **per-seed 差值**：[-0.6, +1.10] 分布，均值 +0.234；两个种子（0.64, 1.10）
   贡献约 74% 的累计和——不是单种子主导，但幅度极小（多数种子 <0.3pp）。
4. **三种 support 模式各改变了什么**：natural=原划分；boost=val 异常占比提升 +
   eval_source=validation（availability 0.60→0.95）；stratified_local=train/val 最小
   support 约束（不改变 reliability availability）；enriched=社区移动 2 个小组
   （c7 异常 2→14），availability 不变。**只有 boost 真正改变 available ratio**。
5. **CKR 权重差**：阶段二无法计算（static 模式不写 ckr.jsonl）→ 本阶段新增
   teacher_mixture.jsonl 解决。
6. **蒸馏前后变化**：阶段二无 Δθ/Δpred 记录 → 本阶段新增 distillation_diagnostics.jsonl。
7. **fairness 无泄漏**：fairness_metrics.json 全部基于 final test 评估，不进训练 ✓。
8. **CiteSeer 失败分类**：10 条记录全部发生在下载阶段（aiohttp connect 错误）；
   阶段二 status.json 只有 'failed'，无法区分代码失败与外部阻塞 → 本阶段新增
   external_blocked / failed_code / failed_config / skipped_unavailable_resource 分类
   （已把阶段二遗留的 10 条记录升级为 external_blocked）。

## 2. Louvain 混杂是否真实存在

**是**。阶段二修复固定了 python-louvain 的 random_state（2024），但：
- natural 磁盘缓存是**未固定种子**时代构建的 → enriched（固定种子）与 natural
  的社区结构来自不同的 Louvain 抽取；
- 阶段二 enriched 正效应（+0.234pp）无法排除"来自 Louvain 抽取差异而非
  重分配机制"的解释；
- 本阶段的 blocked replication 用 5 个独立 partition_seed 消除该混杂。

## 3. 四类随机种子的拆分方式

--seed 兼容入口展开为：partition_seed（Louvain random_state）、allocation_seed
（社区重分配 RNG + support_rebalance_seed）、split_seed（train/val/test 划分）、
model_seed（seed_everything）、bootstrap_seed（统计重采样）。
- 全部写入 config.json（config_hash 含全部种子）；
- data_identity_hash：dataset+task_mode+节点分配+划分索引+标签映射+support 模式/参数；
- initialization_hash：模型初始化 RNG 状态哈希（同 model_seed → 同初始化）；
- 严格配对条件：data_identity_hash + initialization_hash + model_seed + 除 CKR
  模式外全部训练参数一致（test 3/4 验证）。

## 4. fixed split artifact 设计

runs/splits/<dataset>/{natural,enriched}_p{p}/：
- split_config.json（全部种子与 support 参数）、client_assignments.json、
  train_val_test_indices.json、label_mapping.json、data_identity.json、
  class_support_matrix.csv；
- --build_split_artifact：构建并退出（不训练）；--frozen_split：加载并校验
  data_identity_hash（不运行 Louvain/不重划分，test 6 验证）；
- 同一 artifact 被 static/dynamic 共享 → 严格配对。

## 5-7. enriched 正效应独立复现结果

**结论：阶段二 enriched 正效应未能跨独立 partition 复现。**

`ckr_enriched_independent_replication`（5 partition blocks × 5 model seeds ×
static/dynamic = 55 runs，全部通过 frozen split 严格配对：
data_identity_hash 与 initialization_hash 一致，唯一差异为 CKR mode）：

| partition block | n | mean diff (dynamic−static, pp) | wins |
|---|---|---|---|
| p0 | 5 | +0.011 | 3/5 |
| p1 | 5 | +0.007 | 4/5 |
| p2 | 5 | +0.234 | 4/5 |
| p3 | 5 | +0.031 | 3/5 |
| p4 | 5 | **−0.071** | 2/5 |
| **OVERALL** | **25** | **+0.042** | 16/25 |

总体 blocked 统计：
- model-seed 95% CI：[-0.064, +0.149]（跨 0）
- **partition-cluster bootstrap 95% CI：[-0.034, +0.145]（跨 0）**
- Wilcoxon p=0.458；sign test p=0.230；Cohen's dz=0.164（negligible）
- P(improvement)=0.64；P(Δ>0.1pp)=0.32；P(Δ>0.25pp)=0.08；P(Δ>0.5pp)=0.04；
  P(Δ>1.0pp)=0.00
- 方差分解：partition 方差 0.065 ≈ residual 0.067（分区级方差不主导）
- **阶段二 +0.234pp 的"正效应"是 p2 块特有的**（该块恰好复现阶段二单分区
  的 Louvain 抽取）；p4 块为负。按阶段三结论规则：效应未跨多个固定 partition
  稳定 → **不可称为可复现**；"statistically undetectable and practically small"。

## 8. support 不足解释的最终状态

- **H1 部分成立**：只有 boost 真正提高 available ratio（0.60→0.95）；
  enriched 受社区大小约束无法提升 availability。
- **H2 不被支持**：boost（avail 0.95）下 dynamic−static ≈ 0；复现套件显示
  enriched 下总体 ≈ 0（分区级）。
- **H3 结论**：阶段一/二的负结果**不能归因于 support 不足**——在
  availability=0.95 的条件下动态 CKR 依然无收益；且 enriched 划分下的小正效应
  无法跨分区复现。**"支持度不足"解释被削弱；动态 CKR 在 Cora 上未显示
  稳定实际价值。**

## 9-13. static/dynamic CKR 信息量比较（ckr_information_diagnostics，57/57 完成）

natural split（5 seeds，8 轮）：

| 权重模式 | pooled PR-AUC |
|---|---|
| equal | 60.07 ± 3.95 |
| static_ckr | 60.08 ± 3.99 |
| dynamic_ckr | 60.05 ± 3.96 |
| performance_only | 60.07 ± 4.41 |
| shuffled_ckr（负对照） | 60.22 ± 4.17 |
| inverse_ckr（负对照） | 60.03 ± 4.02 |
| oracle_validation（诊断上界） | 60.22 ± 4.58 |

enriched split：equal 59.62 / static 59.72 / dynamic 59.73 / oracle 59.91（同样持平）。

教师混合诊断（teacher_mixture_analysis.csv）：
- dynamic 相对 static 的均值权重差 = **0.025**（微小扰动）；
  dynamic 轮间变化 = 0.0007（几乎冻结）；
- shuffled/inverse 的扰动更大（0.07-0.10），但性能完全相同（60.0-60.2）；
- oracle 的扰动 0.069，也无增益。

**结论（回答"为什么 B4≈B5"）**：
1. dynamic 确实只是 static 的微小扰动（0.025），但这不是主因；
2. **蒸馏对教师权重完全不敏感**——连 shuffled/inverse/oracle 都无法改变结果
   （结论规则 7 命中：应怀疑蒸馏对权重不敏感）；
3. 结合因果链（D1-D7 ≈ D0）：权重方案差异无意义是因为蒸馏整体无贡献；
   教师预测在客户端间高度一致（FedAvg 后收敛），重加权改变不了学生蒸馏结果；
4. oracle 不比 static 好 → "更准确的可靠性权重"在此蒸馏框架内没有潜在价值。

## 14-17. 蒸馏因果链 D0-D8（distillation_causal_chain，61/61 完成）

固定 frozen natural_p0 split，8 轮，5 model seeds（关键配置另跑 16 轮）：

| 配置 | pooled PR-AUC (8 轮) | 16 轮 |
|---|---|---|
| D0 FedAvg only | 60.36 ± 4.41 | 63.34 ± 2.56 |
| D1 equal 蒸馏 | 60.07 ± 3.95 | — |
| D2 static CKR | 60.08 ± 3.99 | — |
| D3 dynamic CKR | 60.05 ± 3.96 | 63.51 ± 2.49 |
| D4 oracle 权重 | 60.23 ± 4.58 | — |
| D5 冻结生成器 | 60.52 ± 3.09 | — |
| D6 无 disagreement | 60.04 ± 3.93 | — |
| D7 完整 (static) | 60.08 ± 3.99 | 63.54 ± 2.49 |
| D8 isolated 拓扑 | 59.42 ± 4.59 | — |

诊断（distillation_chain_analysis.csv）：蒸馏梯度范数各权重方案相近（~0.19-0.20，
D8 例外 1.01）；全局参数更新范数 ~0.13；val 预测变化 ~0.026。

**结论（按阶段三结论规则 8）**：
- 蒸馏本身相对 FedAvg **无稳定增益**（D1-D7 ≈ D0，16 轮亦然）；
- CKR 加权（equal/static/dynamic/oracle）**无差异**——不是权重信息量问题，
  oracle 也不优于 static；
- 生成器训练**无收益**（D5 冻结生成器名义最高 60.52）；
- disagreement 项**无贡献**（D6 ≈ D7）；
- KNN 伪拓扑**无贡献**（D8 略差 59.42，仅改变梯度尺度）；
- 因果链不支持"蒸馏内部组件有效"的任何单项主张。

## 18-20. 公平性方法结果（worst_client_fairness，31/31 完成）

| 配置 | pooled PR-AUC | client-macro | worst client | performance gap |
|---|---|---|---|---|
| F0 B3 (FedAvg) | 60.36 ± 4.41 | 54.57 ± 3.54 | 15.86 ± 7.67 | 72.3 |
| F1 B5 (no fairness) | 60.05 ± 3.96 | 53.90 ± 4.28 | 17.06 ± 8.29 | 71.2 |
| F2 B3 + qffl(q=2) | **42.98 ± 3.91** | 44.25 ± 4.49 | 12.28 ± 9.36 | 68.5 |
| F3 B5 + qffl(q=2) | **43.29 ± 3.39** | 44.67 ± 4.40 | 12.91 ± 9.61 | 68.4 |
| F4 B5 + deficit 蒸馏 | 59.99 ± 3.94 | 53.91 ± 4.29 | 17.06 ± 8.29 | 71.2 |
| F5 B5 + combined | **43.31 ± 3.40** | 44.67 ± 4.39 | 12.91 ± 9.61 | 68.4 |

结论（按结论规则 9：worst-client 提升伴随 pooled 明显下降 → 报告权衡）：
- **qffl(q=2) 聚合严重损害 pooled（−17pp）且未改善最差客户端**（反而 −3~−4pp）；
  q-FedAvg 对 validation loss 的指数加权在本配置下过度压制了高损失客户端
  （它们正是低资源客户端，被压权重后更差）；
- **validation_deficit 蒸馏完全无效**（F4 = F1，worst 完全相同）——与
  教师权重不敏感诊断一致：蒸馏权重怎么调都不影响结果；
- 两个机制都未满足"改善最差客户端"标准 → **公平性优化在此框架内未实现**；
  Pareto 前沿无 win-win 点（fairness_pareto.png）；
- 隐私暴露面：qffl 需上传客户端 validation loss，deficit 需上传 validation
  metric（均为标量，不涉及节点级数据）；报告明确此新增暴露面。

## 21-23. CiteSeer/PubMed

- CiteSeer：**external_blocked**（10 条记录，全部为下载阶段连接失败；
  阶段三不再重复制造失败，代码/套件/校验就绪，数据可用后运行）；
- PubMed：同样 blocked，只保留加载/smoke 配置；
- 不伪造外部数据；synthetic_test_graph 仅限 smoke，不进正式结果。

## 24. 测试结果

test_all 59 + test_smoke 26 + test_experiments 10 + test_phase2 25 +
**test_phase3 32** 全部通过（截至报告时）。

## 25. 所有 suite 计划/成功/失败/external_blocked（填充中）

## 26-30. 结论（填充中）

## 结论矩阵

conclusion_matrix.csv（结果就绪后由 generate_conclusion_matrix.py 生成）
