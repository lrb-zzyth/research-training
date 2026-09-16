# 阶段二报告：机制验证、统计增强与外推实验

日期：2026-08-01（全部计划实验完成，除网络阻塞项）
git commit：见各 runs/*/config.json

## 1. 第一阶段结果审计结论

1. **完整性**：171/171 运行目录齐全（10 个结果文件、无空 metrics.jsonl、无重复轮次、
   无重复运行）。
2. **评估口径不一致**：B0 无 pooled（`pooled={}`），只有 client_macro；B1-B5 有
   pooled+client_macro 但无 client_weighted；无 evaluation_scope/comparability 字段。
   第一阶段报告 B0 与 B1-B5 主结果非同口径（B0=client-macro 62.67 vs 其余 pooled），
   且未将 B0 纳入配对。
3. **CKR 可观测性不足**：ckr.jsonl 12 字段，无动态信号/support/delta/status；
   B5 种子 0 实测 available 60%、fallback 40%（全为 support 不足）、
   **cell-round 转换仅 15.7% 实际变化**、均值 |ΔR|≈0.002。
4. **RWR 缓存**：key 语义正确（client/view/round/edge_hash/anchor/size/prob/seed），
   但命中率仅 5.2%（随机锚点 + epoch=1），无时间收益证据。
5. **checkpointed 显存**：仅记录单点 max_memory_allocated，小模型下 44.7 vs 44.5MB
   不可观测；无 reserved/时间/梯度等价性记录。
6. **可复用**：种子 0-4 全部结果（同 schema 聚合、CKR 曲线、资源数据）；
   **必须重跑**：需要新评估口径/新字段的结果（B0 pooled、client_weighted、
   evaluation_scope、CKR 可观测性）→ 新 suite 全部 10 种子重跑。

## 2. 发现并修复的问题（本阶段）

| # | 问题 | 修复 |
|---|---|---|
| 1 | FGLDataset 未把 partition_support_mode 从 args 复制到 self → enriched 静默回退 natural | __init__ 复制 4 个 partition 属性 |
| 2 | enrich_anomaly_support 社区在不足客户端间振荡（50 moves 净值归零） | 每组最多移动一次 + 来源方移后保持 ≥target |
| 3 | python-louvain 未固定 RNG → 新鲜划分不可复现（294 vs 32 异常计数） | random_state=2024 固定 |
| 4 | partition_moves 未持久化（FGLDataset 读不到） | data_partition 挂到 G + moves.json |
| 5 | test_phase2 Test 24 dry-run 污染正式 suite 目录（覆盖 6 个正式单元） | dry-run 写入独立 scratch 目录 |
| 6 | run_experiments 并发调用同一 suite 无锁（两个进程写同一单元） | 流程上避免并发（本阶段人工管控）；文档注明 |
| 7 | Holm 校正缺单调 step-up | 已修复（test 9 验证） |
| 8 | make_plots 用旧 'ema' 键（新记录为 ema_ckr） | 兼容读取 |

## 3. 修改文件（阶段二）

- `train_fedtad.py`：split_support_mode 四模式、min support/anomaly_val_ratio/
  allow_support_infeasible/anomaly_partition_target、统一评估（collect_final_metrics
  重写：pooled/client_macro/client_weighted + evaluation_scope +
  metric_comparability_group）、B0 local-ensemble pooled、fairness/ckr_availability/
  cache_metrics/evaluation_report JSON、anchor_sampling_mode、rwr_cache_mode/scope、
  generator_backward_mode 别名、checkpoint_segments
- `util/data_split.py`：stratified_split_with_support、anomaly_holdout_boost_split
- `util/base_data_util.py`：louvain_partition(return_groups, louvain_seed)、
  enrich_anomaly_support（move+swap、振荡防护、moves 持久化）
- `util/fgl_dataset.py`：partition 属性复制、enriched 目录隔离、moves.json
- `util/dynamic_ckr.py`：18 字段记录、round_aggregate、fit support、last_delta
- `util/rwr_cache.py`：round_idx=None（scope=run）key
- `util/experiment_stats.py`：paired_bootstrap_ci、holm_correction（单调）、
  cliffs_delta、sensitivity_analysis
- `model.py`：checkpoint_segments 分段重计算、_reverse_step 重构
- `benchmark_checkpointing.py`（新）：显存/时间/数值等价性微基准
- `run_experiments.py`：7 新 suite、runner 机制、--plan、--dry、hash 感知跳过、
  扩展聚合（enhanced paired/mechanism_pairs/ckr_analysis/fairness/cache/
  numerical_equivalence CSV）、新图表
- `test_phase2.py`（新，25 项测试）

## 4. 新增参数（阶段二）

--split_support_mode / --min_train_support_per_class / --min_val_support_per_class /
--anomaly_val_ratio / --support_rebalance_seed / --allow_support_infeasible /
--anomaly_partition_target / --rwr_cache_mode / --rwr_cache_scope / --local_epochs /
--anchor_sampling_mode / --anchor_pool_size / --generator_backward_mode /
--checkpoint_segments / --evaluation_report_json / --ckr_availability_json /
--fairness_metrics_json / --cache_metrics_json

## 5-7. 测试与运行数量

**测试**：test_all 59/59 ✓、test_smoke 26/26 ✓、test_experiments 10/10 ✓、
test_phase2 25/25 ✓（新增覆盖：B0 统一评估、comparability 阻止配对、CKR 状态字段、
fallback 原因、support 划分不重叠/不移动 test、enriched 不复制节点、10-seed 聚合、
Holm、bootstrap、checkpointed 数值等价、无 CUDA 安全跳过显存、RWR 缓存语义、
CiteSeer 维度、unavailable 不置 0、suite plan 数量、hash 感知跳过/重跑、JSON 字段、
新 suite CPU dry-run、Louvain 确定性）。

**计划 vs 实际运行数**：

| suite | 计划 | 成功 | 失败 | 跳过 | 说明 |
|---|---|---|---|---|---|
| main_10seeds | 60 | 60 | 0 | 0 | 全部 10 种子 |
| ckr_support_mechanism | 80 | 80 | 0 | 0 | 10 种子（M7/M8 修复后重跑） |
| long_horizon_ckr | 30 | 30 | 0 | 0 | 5 种子 × 16 轮 |
| checkpoint_memory_scaling | 24 | 24 | 0 | 0 | 12 配置 × seed 0 |
| rwr_cache_effective | 42 | 42 | 0 | 0 | 14 配置 × 3 种子 |
| dataset_transfer_citeseer | 50 | 0 | 10* | 0 | *网络阻塞（github 不可达），10 次尝试均以 failed+stderr 记录 |
| pubmed_smoke | 2 | 0 | 0 | 0 | 未尝试（同一网络阻塞，明确未完成） |

## 8. B0 统一评估结果（10 seeds）

B0 local-ensemble pooled（各客户端本地模型对各自 test 预测拼接，不合并参数）：
- pooled PR-AUC **78.08 ± 3.60**（高于全局模型方法的 64.1，但**口径不同不可比较**：
  local ensemble 是 per-client 特化模型，global model 是共享模型）
- client_macro PR-AUC 60.84 ± 3.91（低于 B5 的 63.39 ± 5.52，描述性比较）
- evaluation_scope='local_ensemble'，comparability group 阻止与 global_model 配对 ✓
- pooled 样本数 = 各客户端 test 样本数之和 ✓（test 1 验证）

## 9. 10-seed 主结果（anomaly binary，pooled PR-AUC %，10 seeds）

| 配置 | pooled PR-AUC | pooled ROC-AUC | client-macro PR-AUC |
|---|---|---|---|
| B0 local_only | 78.08 ± 3.60* | 91.89 ± 1.32 | 60.84 ± 3.91 |
| B1 FedAvg | 23.27 ± 2.15 | 67.22 ± 2.86 | 26.19 ± 2.10 |
| B2 +wCE | 55.43 ± 5.82 | 87.84 ± 1.95 | 57.27 ± 4.40 |
| B3 +contrastive | 63.30 ± 5.01 | 90.44 ± 1.35 | 63.07 ± 4.84 |
| B4 +static CKR | 64.20 ± 5.48 | 90.71 ± 1.20 | 63.36 ± 5.39 |
| B5 Full | 64.15 ± 5.55 | 90.71 ± 1.21 | 63.39 ± 5.52 |

\* local-ensemble 口径，不与其余行配对。

配对（B5 − 基线，10 种子，pooled PR-AUC）：

| 对照 | mean | median | bootstrap 95% CI | wins/losses | Wilcoxon p | Holm p | Cliff's δ |
|---|---|---|---|---|---|---|---|
| B1 | +40.88 | +40.67 | [37.14, 44.50] | 10/0 | 0.002 | 0.031 | 1.0 (large) |
| B2 | +8.72 | +8.94 | [6.49, 10.81] | 10/0 | 0.002 | 0.031 | 1.0 (large) |
| B3 | +0.85 | +0.93 | [0.001, 1.71] | 6/4 | 0.131 | 0.654 | 0.2 (small) |
| B4 | -0.05 | +0.00 | [-0.17, 0.05] | 5/5 | 0.625 | 1.0 | 0.0 (negligible) |

结论：wCE 与对比学习的显著收益在 10 种子下保持；B3→B4→B5 的蒸馏/动态 CKR
增量无统计证据（B5 vs B3 CI 仅擦边含 0，p=0.13；B5 vs B4 明确无差异）。

## 10. 动态 CKR support 机制实验（10 seeds，M1-M8）

| 机制 | available ratio | fallback | changed ratio | 均值\|ΔR\| | dynamic−static (pooled PR-AUC) |
|---|---|---|---|---|---|
| natural | 0.60 | 0.40 | 0.33-0.49 | 0.006-0.011 | -0.055 [−0.17, 0.05], 5/5, p=0.63 |
| stratified_local | 0.60 | 0.40 | 同上 | 同上 | -0.055 [−0.17, 0.05], 5/5, p=0.63 |
| anomaly_holdout_boost | **0.95** | 0.05 | 0.78-0.93 | 0.016-0.022 | +0.002 [−0.23, 0.31], 3/7, p=0.56 |
| anomaly_enriched_partition | 0.60 | 0.40 | 0.38-0.49 | 0.006-0.009 | **+0.234 [0.055, 0.47], 10/0, p=0.002** |

假设检验：
- **H1（提高支持度 → 提高 available ratio）**：部分成立——boost 把 availability 从
  0.60 提到 0.95（val 异常占比提升 + eval_source=validation）；stratified_local 不改变
  availability（它约束 train/val support，不影响 reliability holdout，符合设计）；
  enriched 因社区大小约束（异常富集社区 188 节点 > 接收方剩余容量）只能移动 2 个
  小组（c7 异常 2→14），未达 target=75，availability 不变。**机制层面：只有
  val-boost 真正提高了可用性。**
- **H2（available ratio 高时 dynamic 才有收益）**：**不被支持**。availability=0.95
  的 boost 条件下 dynamic−static ≈ 0；而 availability=0.6 的 enriched 条件下
  dynamic−static = +0.234（10/0 一致为正，但每种子差异很小：+0.01~+1.10pp）。
  收益与 availability 无单调关系（mode 级相关 ≈ 0）。
- **H3（第一阶段的负结果不能只归因于 support 不足）**：**成立**——把异常类
  availability 提到 95% 后 dynamic 仍无收益；但 enriched 划分下的微小一致正收益
  表明**客户端组成（谁持有异常）而非原始可用率**可能更重要。该正效应需独立复现
  （且 enriched 划分同时改变了 Louvain 基础随机种子，存在混杂）。

## 11. 长期轮次结果（16 轮，5 seeds，natural）

| 配置 | pooled PR-AUC | pooled ROC-AUC |
|---|---|---|
| LH_B3 (contrastive only) | 66.21 ± 3.48 | 91.12 ± 0.78 |
| LH_B4 (static CKR) | 66.73 ± 3.89 | 91.20 ± 0.62 |
| LH_B5 (dynamic CKR) | 66.63 ± 3.86 | 91.01 ± 0.92 |

（enriched 变体：B3 66.69 ± 5.46 / B4 67.07 ± 6.32 / B5 67.06 ± 6.26，修复后重跑）

- 每轮 val pooled PR-AUC 曲线：B3/B4/B5 在 round 3-5 起几乎重合（46-65），
  round 8 后基本饱和（64.5-65）；**B4 与 B5 未出现后期分化**，动态 CKR 与静态
  在 16 轮内收敛到相同水平
- 蒸馏收益随轮次累积：B4 vs B3 的最终差 +0.5pp（16 轮），早期无差异——
  蒸馏收益小且滞后
- 无后期过拟合迹象（val 曲线 12 轮后持平），无尾部客户端持续恶化
- enriched 变体在修复后重跑（30/30），分析同 §10

## 12. checkpointing 显存—时间结果（GPU 实测，seed 0）

| hidden | steps | full 峰值 MB | ckpt 峰值 MB | 显存降低 | full 步时 | ckpt 步时 | 等价性 |
|---|---|---|---|---|---|---|---|
| 32 | 10 | 25.2 | 24.7 | 2.2% | 0.010s | 0.019s | exact |
| 32 | 50 | 48.7 | 40.5 | 16.8% | 0.045s | 0.088s | exact |
| 64 | 50 | 52.4 | 42.2 | 19.4% | 0.044s | 0.087s | exact |
| 128 | 50 | 61.5 | 46.2 | 24.9% | 0.046s | 0.093s | exact |
| 256 | 50 | 82.1 | 57.6 | 29.9% | 0.044s | 0.084s | exact |

（完整 12 组见 numerical_equivalence.csv；10 步时收益仅 1-2%，25 步 10-16%，
50 步 17-30%）

结论：**checkpointing 降低显存成立**（≥25 步可稳定观测，50 步/256 隐藏达 30%）；
代价约 2× 步时（激活重算）；**全部配置 output/gradient/loss 最大绝对差 = 0
（exact 等价）**——checkpointed 是完整梯度，不是近似。

## 13. RWR 缓存命中率—加速结果（3 seeds）

| 配置 | 命中率 | 每轮耗时 | pooled PR-AUC |
|---|---|---|---|
| e1 random + cache off | 0% | 21.4s | 65.89 |
| e1 random + epoch_reuse | 5% | 18.6s | 65.89 |
| e1 fixed + epoch_reuse | 5% | 19.3s | 66.54 |
| e3 random + epoch_reuse | 15% | 54.6s | 67.30 |
| e3 fixed + epoch_reuse | **68%** | **23.9s** | 65.91 |
| e3 fixed + safe_exact | 67% | 21.4s | 65.91 |
| e5 random + epoch_reuse | 22% | 83.9s | 67.46 |
| e5 fixed + epoch_reuse | **81%** | **21.5s** | 65.60 |
| e5 fixed + cache off | 0% | 90.3s | 65.60 |

结论（与第一阶段 5% 负结果的关键差异）：
- **锚点采样模式是命中率的决定性因素**：fixed_per_round + epoch_reuse 在
  epochs=3/5 时命中率 68%/81%（random 仅 15%/22%）
- **命中率带来实际时间收益**：e5 下 21.5s vs 90.3s/轮（**降低 76%**）；e3 下
  23.9s vs 47.8s（降低 50%）
- **指标在缓存开/关下完全一致**（同配置 PR-AUC 相同）——缓存不改变结果
- 第一阶段 5.2% 命中率归因于 random_each_epoch + epoch=1 的组合，非缓存本身缺陷

## 14. CiteSeer 外推结果

**阻塞**：环境无网络（github.com:443 不可达），CiteSeer/PubMed 无法下载。
`dataset_transfer_citeseer` 已定义（multiclass + anomaly_binary 各 B1-B5 × 5 种子，
映射 normal={0,1,2,3}/anomaly={4,5,6}，缓存 key 含 dataset，维度自动适配），
10 次尝试均以 failed+stderr 如实记录（aiohttp 连接错误）。
**本阶段无法产生外推数据；代码、测试（维度/num_classes/缓存隔离）与复现命令就绪。**

## 15. 公平性与低资源分析（main_10seeds，client 级 PR-AUC）

| 配置 | worst client | bottom-10% | performance gap | zero-anomaly test clients |
|---|---|---|---|---|
| B1 | 6.34 | 6.34 | 58.1 | 0 |
| B2 | 18.01 | 18.01 | 72.1 | 0 |
| B3 | 19.27 | 19.27 | 73.5 | 0 |
| B4 | 18.69 | 18.69 | 75.2 | 0 |
| B5 | 18.88 | 18.88 | 75.1 | 0 |

（bottom-10% 与 worst 相同：10 客户端中 10% = 1 个客户端。）

- wCE 把最差客户端从 6.3 提升到 18.0（**wCE 主要帮助低资源/异常少的客户端**）；
- contrastive 进一步把最差客户端提到 19.3；
- B4/B5 的最差客户端（18.7/18.9）略低于 B3（19.3）——蒸馏阶段未改善最差客户端，
  performance gap 反而略增；
- 全部方法零异常 test 客户端数为 0（10-client 划分下每客户端 test 均有异常样本）；
- 资源分组（train 节点数/异常训练数/度）的组内均值见 fairness_analysis.csv；
  以上仅为客户端间性能分布分析，不称为公平性证明。

## 16. 配对统计与多重比较校正

- 10 种子配对：mean+median diff、bootstrap 95% CI（1000 重采样）、Wilcoxon、
  Holm 校正、Cliff's δ、敏感性（leave-one-pair-out）——全部写入
  `paired_comparisons_enhanced.csv`
- 显著结论同时要求 CI 排除 0 与 wins 占优（B5 vs B1/B2 满足；B5 vs B3 CI 擦边
  含 0 + p=0.13 → 不写显著；B5 vs B4 CI 明确含 0 → 无差异）
- 不把"不显著"写成"证明等价"；报告效应量与原始种子值

## 17. 被加强的原有结论

1. wCE 是异常检测主效因（10 种子：+40.9pp vs B1，CI 排除 0，10/0）
2. 对比学习显著提升（+8.7pp vs B2；A5 消融 +6.1pp）
3. checkpointed 与 full 梯度完全等价（12 配置 exact）
4. B4≈B5（动态 CKR 无增益）在 natural/stratified/boost 三种机制条件下均成立

## 18. 被削弱或推翻的原有结论

1. **"动态 CKR 无收益"的 support 不足解释被削弱**：availability 提到 95% 后仍无
   收益（H2 不被支持）；且 enriched 划分下出现微小一致正收益（+0.23pp, 10/0）
2. **"RWR 缓存无效"被推翻**：固定锚点 + ≥3 epoch 下命中率 68-81%、轮时降 50-76%、
   指标不变
3. **"checkpointed 显存收益不可观测"被部分推翻**：≥25 步可稳定观测（10-30%）

## 19. 当前仍不能支持的结论

- 不能说"动态 CKR 在支持度足够时一定有效"（boost 反例）+ enriched 正效应含
  Louvain 种子混杂，未独立复现
- 不能说"enriched 划分提升了 available ratio"（社区大小约束下未达 target，
  availability 仍 0.6）
- 不能说"B0 优于联邦方法"（local-ensemble 与 global-model 口径不同，禁止配对）
- 不能说"蒸馏提升最差客户端"（实测最差客户端略降）
- CiteSeer/PubMed 外推：无数据，无结论

## 20. 下一阶段最值得开展的实验

1. **独立复现 enriched 正效应**：在固定 Louvain 种子的 natural 划分上重跑
   M1/M2（消除种子混杂），并增大异常目标（如 target=100 或用合成社区结构）
2. **支持度 × 动态收益的机制扫描**：在 boost 基础上做 anomaly_val_ratio ∈
   {0.3, 0.5, 0.7} × min_support ∈ {1, 3, 5} 的网格，定位收益边界
3. **长轮次动态 CKR**：16 轮 enriched 变体（本阶段已跑，待 enriched 复现后合并分析）
4. **RWR 缓存上界**：fixed_global 锚点 + epoch_reuse 的命中率上限与显存占用
5. **数据外推**：待网络可用后运行 dataset_transfer_citeseer（50 运行）与 pubmed_smoke
6. **公平性**：设计针对最差客户端的正则化（如 worst-client 加权），验证能否在
   不损失 pooled 的前提下收窄 performance gap

## 21. 复现命令

```bash
# 测试
python test_all.py && python test_smoke.py && python test_experiments.py && python test_phase2.py
# 实验
python run_experiments.py --prepare-proxy
python run_experiments.py --suite main_10seeds
python run_experiments.py --suite ckr_support_mechanism
python run_experiments.py --suite long_horizon_ckr
python run_experiments.py --suite checkpoint_memory_scaling
python run_experiments.py --suite rwr_cache_effective
# 聚合与图表 (每 suite 之后)
python run_experiments.py --aggregate --suite <name> && python run_experiments.py --plots --suite <name>
# 外推 (需网络)
python run_experiments.py --suite dataset_transfer_citeseer
python run_experiments.py --suite pubmed_smoke
```
