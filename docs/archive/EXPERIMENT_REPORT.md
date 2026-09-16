# 正式实验报告：FedTAD 动态 CKR 与模块消融

日期：2026-08-01（实验全部完成，171/171 运行成功）
git commit：见各 `runs/*/config.json`

> 本报告只陈述实际运行结果。所有 mean±std 均来自 5 个随机种子 (0-4) 的完整运行
> （标注 3 种子的除外）。Cora 不是原生异常检测数据集：anomaly_binary 的异常标签
> 来自实验性类别映射（见 §3）。

## 1. 实验目的

验证各模块（动态 CKR、reliability holdout、RWR 缓存、扩散生成器反向传播模式、
代理预训练）是否带来可统计验证的性能/资源收益，并建立可复现、无数据泄漏的
正式实验体系。

## 2. 数据集与任务构造

- Cora（Planetoid），特征 1433 维，7 类；Louvain 图划分 10 客户端
  （`dataset/Cora/Client10/Louvain` 磁盘缓存），每客户端 232-286 节点，
  标签分布高度异质（部分客户端整类缺失）
- 划分比例 train 0.2 / val 0.4 / test 0.4；重划分由
  `stratified_split(seed=--seed+client_id)` 控制，跨种子划分不同

## 3. 标签映射（anomaly_binary）

- 实验性映射：normal={0,2,3,4,5}，anomaly={1,6}；映射前后分布见 `split_report.json`
- 全图异常比例 ≈ 13%（类 1+6 合计）；**非 Cora 原生属性**
- 映射后 `--resplit_stratified` 重划分（默认 true），保证按二分类标签分层

## 4. 数据划分

- fit_idx：加权 CE + 参数更新（唯一参与训练损失的集合）
- reliability_idx：动态 CKR 指标（20% holdout，不参与 backward）
- val_idx：模型选择（anomaly=pooled PR-AUC，multiclass=macro-F1）
- test_idx：最终评价（加载 best.pt 后运行一次）
- 启动断言：四集合两两不重叠、fit+rel==train、rel⊆train

## 5. 客户端划分

Louvain 固定 10 客户端；train/val/test 每种子独立重划分（§2）。

## 6. 基线

B0 local_only（无聚合）→ B1 FedAvg（普通 CE）→ B2 +weighted CE →
B3 +跨视图对比 → B4 +静态 CKR 蒸馏 → B5 Full（hybrid 动态 CKR + EMA）。
配置逐级叠加，见 `runs/*/config.json` 与 `run_experiments.py::COMMON_ARGS`。

## 7. 完整方法（B5 Full）

weighted CE + 对比学习 + reliability holdout + hybrid 动态 CKR（α=0.5, γ=0.8, f1）
+ 教师引导扩散伪图生成（4 步）+ 可靠性加权蒸馏 + checkpointed 反向传播 + RWR 缓存。

## 8. 统一超参数

num_clients=10, hid=32, dropout=0.2, lr=1e-2, wd=5e-4, epochs=1, rounds=8,
contrastive batch=8, subgraph=3, diffusion steps=4, hidden=32, fake nodes=48,
distill steps=2, min support=3, holdout=0.2, f1/auc 终止阈值关闭（跑满 8 轮）。

## 9. 模型选择规则

- anomaly：validation pooled PR-AUC；multiclass：validation macro-F1
- NaN 不保存 best；test 不进选择条件；每运行仅一次 final test（加载 best.pt）
- `final_metrics.json::loaded_from` 记录实际加载文件（全部为 best.pt）

## 10. 多随机种子设置

种子 0-4 统一控制 Python/NumPy/torch CPU+CUDA RNG、划分、RWR（rwr_seed=0 +
采样种子）、边扰动、扩散噪声。同种子确定性由 test_experiments.py Test 2 验证
（两次运行最终 pooled PR-AUC 完全一致）。

## 11. 主结果

### Anomaly binary（pooled PR-AUC %，5 seeds）

| 配置 | pooled PR-AUC | pooled ROC-AUC | client-macro PR-AUC | anomaly recall* |
|---|---|---|---|---|
| B0 local_only | — | — | 62.67 ± 1.91 | 56.9 ± 33.1 |
| B1 FedAvg | 23.95 ± 2.93 | 67.85 ± 4.02 | 27.09 ± 2.70 | 0.0 ± 0.0 |
| B2 +wCE | 58.65 ± 2.38 | 88.63 ± 0.69 | 60.19 ± 0.77 | 26.1 ± 28.4 |
| B3 +contrastive | 64.62 ± 4.51 | 90.64 ± 0.90 | 64.95 ± 3.78 | 57.8 ± 29.3 |
| B4 +static CKR | 65.02 ± 5.11 | 90.72 ± 0.77 | 65.17 ± 4.02 | 58.5 ± 29.0 |
| B5 Full | 65.05 ± 5.11 | 90.73 ± 0.79 | 65.28 ± 3.90 | 58.3 ± 28.9 |

\* 每客户端 test anomaly recall，客户端等权（全部 50 个 client×seed 单元均有效）。
B1 的 recall=0：普通 CE 完全坍缩到多数类（永不预测异常）。

配对分析（B5 − 基线，5 种子，pooled PR-AUC）：

| 对照 | Δmean | 95% CI | wins/losses | Wilcoxon p |
|---|---|---|---|---|
| B1 | +41.10 | [32.16, 50.03] | 5/0 | 0.0625 |
| B2 | +6.40 | [2.10, 10.69] | 5/0 | 0.0625 |
| B3 | +0.43 | [-0.83, 1.68] | 2/3 | 0.8125 |
| B4 | +0.03 | [-0.13, 0.19] | 3/2 | 0.6250 |

### Multiclass（5 seeds）

| 配置 | pooled accuracy | pooled macro-F1 |
|---|---|---|
| B1 FedAvg | 33.79 ± 6.00 | 0.188 ± 0.073 |
| B5 Full | 58.07 ± 7.96 | 0.490 ± 0.086 |

配对（B5−B1）：accuracy +24.28 [17.46, 31.10]，5/0，p=0.0625；
macro-F1 +0.302 [0.223, 0.381]，5/0，p=0.0625。两个 CI 均排除 0。

## 12. 消融结果（anomaly，pooled PR-AUC，配对 vs 本套件 reference）

| 消融 | 配置 | mean±std | Δ vs ref（95% CI） | wins/losses | p |
|---|---|---|---|---|---|
| A1 CKR mode | static_topology | 65.02 ± 5.11 | +0.03 [-0.13, 0.20] | 3/2 | 0.625 |
| | performance_only | 65.34 ± 4.61 | -0.29 [-1.17, 0.59] | 3/2 | 1.000 |
| | hybrid（ref） | 65.05 ± 5.11 | — | — | — |
| A2 metric | f1（ref） | 65.05 ± 5.11 | — | — | — |
| | recall | 65.03 ± 5.02 | +0.02 [-0.12, 0.16] | 2/3 | 1.000 |
| | confidence | 64.97 ± 5.08 | +0.08 [-0.11, 0.26] | 3/2 | 0.438 |
| A3 EMA | no EMA (γ=0) | 65.05 ± 5.11 | +0.002 [-0.02, 0.02] | 2/3 | 1.000 |
| | default (γ=0.8, ref) | 65.05 ± 5.11 | — | — | — |
| | strong (γ=0.95) | 65.04 ± 5.10 | +0.008 [-0.007, 0.024] | 4/1 | 0.188 |
| A5 contrastive | off | 58.96 ± 2.13 | +6.09 [1.23, 10.95] | 5/0 | 0.0625 |
| | on（ref） | 65.05 ± 5.11 | — | — | — |
| A6 CE | plain | 25.94 ± 3.62 | +39.11 [29.58, 48.65] | 5/0 | 0.0625 |
| | weighted（ref） | 65.05 ± 5.11 | — | — | — |
| A7 distill | equal | 65.38 ± 3.85 | -0.33 [-2.35, 1.69] | 4/1 | 0.625 |
| | static_ckr | 65.02 ± 5.11 | +0.03 [-0.13, 0.19] | 3/2 | 0.625 |
| | dynamic（ref） | 65.05 ± 5.11 | — | — | — |
| A8 gen init | scratch（ref） | 65.05 ± 5.11 | — | — | — |
| | proxy_pretrained | 65.28 ± 4.86 | +0.23 [-0.14, 0.59] | 4/1 | 0.188 |
| A9 backprop | full | 65.05 ± 5.11 | 0.000（与 ref 完全一致） | — | — |
| | checkpointed（ref） | 65.05 ± 5.11 | — | — | — |
| | truncated (i=2) | 65.03 ± 5.18 | +0.02 [-0.13, 0.17] | 3/2 | 1.000 |
| A10 RWR cache | off | 65.05 ± 5.11 | -0.002 [-0.04, 0.03] | 1/1 | — |
| | on（ref） | 65.05 ± 5.11 | — | — | — |
| A4 source | holdout（ref，3 seeds） | 65.89 ± 6.72 | — | — | — |
| | validation（3 seeds） | 66.08 ± 6.44 | -0.19 [-0.97, 0.59] | 1/2 | — |

注：A7 中 equal 名义最高（65.38），但相对 dynamic 的配对差为 -0.33（种子方向不一致），CI 大幅跨 0。

## 13. 客户端级结果

`runs/main_anomaly_binary/per_client_results.csv`（50 个 client×seed 单元/方法）。
B5_full/seed_0 异常类 per-set support：

| client | fit | reliability | val | test | reliability 异常缺失（<3 → 回退） |
|---|---|---|---|---|---|
| c0 | 5 | 1 | 12 | 14 | ✓ |
| c1 | 4 | 1 | 10 | 10 | ✓ |
| c2 | 6 | 1 | 14 | 15 | ✓ |
| c3 | 27 | 6 | 67 | 68 | — |
| c4 | 4 | 0 | 9 | 10 | ✓ |
| c5 | 12 | 3 | 30 | 32 | — |
| c6 | 2 | 0 | 5 | 7 | ✓ |
| c7 | 0 | 0 | 0 | 2 | ✓（fit/val 零异常） |
| c8 | 2 | 0 | 4 | 4 | ✓ |
| c9 | 2 | 0 | 4 | 4 | ✓ |

零异常客户端（fit/rel/val/test 全零）：无（c7 在 test 有 2 个异常，正确保留）。
单类 test 客户端 AUC/PR-AUC 标记 NaN 并按有效客户端数聚合（不填 0）。

## 14. CKR 行为分析

来自 `runs/main_anomaly_binary/B5_full/seed_*/ckr.jsonl`（8 轮 × 10 客户端 × 2 类）：

- 轮间波动 ΔR^t（EMA 相邻轮平均绝对变化）：每轮 0.0004–0.0054，5 种子均值
  **0.0020 ± 0.0012**——γ=0.8 下 CKR 极其稳定
- fallback 比例恒为 **40%**（800 单元中 320 个），全部为
  `nan_metric_first_round_static`（异常类 reliability support < 3）
- 8/10 客户端的异常类动态指标无法计算 → 异常类权重实际基本是静态先验
- 这与 A1（三模式无差异）、A3（EMA 无差异）及主结果（B4≈B5）自洽：
  **动态 CKR 在本划分下缺少足够的动态信号**

## 15. 生成器分析（A9）

- full 与 checkpointed 的 pooled PR-AUC **完全相同（65.048 vs 65.048）**——
  实证 checkpointed 保留完整梯度（激活重算不影响结果）
- 峰值显存：full 44.7 MB / checkpointed 44.5 MB / truncated 44.3 MB——
  本小模型配置（hidden=32, 4 步, 48 伪节点）下差异 <0.5 MB，不可观测；
  checkpointed 的显存收益需更大模型
- truncated 指标与 checkpointed 无差异（CI 跨 0）；每轮耗时读数受并发脚本
  CPU 争用影响，仅作参考
- 无 OOM 发生（171/171 成功）

## 16. 缓存与资源开销（A10）

- RWR 缓存命中率仅 **5.2%**：锚点每轮随机抽样、num_epochs=1，
  同锚点跨轮重复极少（view_1 持久缓存无对象可命中）
- 指标完全一致（65.05 vs 65.05），无时间收益（命中率过低时
  key 哈希开销占主导）；结论：**本采样模式下 RWR 缓存既不损也不赚**
- 峰值显存/每轮耗时等资源数据见 `resource_results.csv`

## 17. 统计分析

- 配对差值（同种子）、95% CI（t 分布）、Wilcoxon、wins 计数：
  `runs/*/paired_comparisons.csv`
- 5 种子时 Wilcoxon 最小可达 p=0.0625，报告不以 p 值为唯一依据，
  同时给出 Δmean、CI 与 wins
- 显著结论判据：配对 CI 排除 0 且 5/5 胜出

## 18. 数据泄漏检查

1. fit/rel/val/test 两两不重叠 ✓（启动断言 + 测试 4）
2. 标签映射后重划分 ✓（--resplit_stratified 默认 true）
3. 训练损失只使用 fit ✓（代码路径 + 断言）
4. CKR 指标只使用 reliability（默认）✓；A4 validation 对照显式打印
   `WARNING: validation labels are being used to update CKR...` ✓
5. 模型选择只使用 val ✓；6. final test 加载 best.pt ✓（loaded_from 全部为 best.pt）
7. test 不进 checkpoint 条件 ✓
8. RWR 缓存拒绝 requires_grad 张量 ✓（测试 48）
9. view_2 按 round 失效 ✓（测试 47）
10. resume 后 EMA/优化器/RNG 连续 ✓（smoke A 实测 11 单元 EMA 链）
11. 任务模式元数据校验 ✓；12. proxy 维度/类别数校验 ✓（测试 53）
13. unavailable 保持 NaN 不置 0 ✓（测试 43）
14. 结果文件含 seed + config_hash ✓
15. CKR 磁盘缓存 key 含 seed（重划分后跨种子复用旧 CKR 会引入泄漏——已修复）

## 19. 隐私边界

客户端上传仅含每类别分数/support/mask；服务端生成器/蒸馏函数签名不含 client
Data（测试 42）；代理预训练仅用本地合成数据（逐类高斯簇 512→1433 维
FeatureAdapter），checkpoint 记录维度/模式，联邦加载校验维度与 conditional 类别数。

## 20. 局限性

1. 单数据集（Cora）、单划分（Louvain-10）；结论外推有限
2. 动态 CKR 负结果的一个机制性解释：异常类 reliability support 不足导致
   40% 单元全年回退——在异常样本更充足的数据集上结论可能不同
3. 小模型配置下 checkpointed 显存收益不可观测（44.7→44.5MB）
4. RWR 缓存命中率 5% 源于锚点随机抽样；train_nodes 锚点作用域或更多本地
   epoch 才可能产生实际命中
5. 5 种子时 Wilcoxon p 分辨率有限（≥0.0625）
6. 每轮耗时读数受分析脚本 CPU 争用影响
7. 代理数据为合成高斯簇，与真实代理分布的差距未知

## 21. 失败运行与未完成项

- **已实际完成**：13 个套件、171/171 运行成功、0 失败（含 5 种子全部 8 轮）
- **已实现但未运行**：无（所有已定义 suite 均已完成）
- **因资源限制未完成**：无；10 种子扩展（任务建议可选）未运行——单种子
  额外成本约 2 分钟，如需可随时补齐

## 22. 修改文件清单

新增（本阶段）：
- `run_experiments.py`（741 行）— 统一实验入口：13 个 suite、run 目录、
  skip/失败恢复、聚合 CSV、配对统计、英文图表
- `util/experiment_stats.py`（125 行）— mean±std/CI/Wilcoxon/ΔR^t
- `test_experiments.py`（247 行）— 10 项实验体系测试
- `EXPERIMENT_REPORT.md` — 本报告
- （上一阶段已有：`pretrain_diffusion.py`、`test_smoke_train.py`、
  `util/dynamic_ckr.py`、`util/checkpoint.py`、`util/rwr_cache.py`、
  `util/data_split.py`、`DELIVERY_REPORT.md`）

修改（本阶段）：
- `train_fedtad.py` — `--federated_mode local_only`（B0）、`--distill_weighting`
  none/equal/static_ckr/dynamic_ckr（B1-B5 阶梯与 A7）、`performance_only` 别名、
  `--resplit_stratified` 通用化（multiclass 也按种子重划分）、CKR 缓存 key 加 seed、
  `run_integrity_assertions`、每轮 metrics JSONL、final_metrics/split_report/
  resource_usage JSON 输出、per-client 最终指标收集、`--use_weighted_ce`
  BooleanOptionalAction、device fallback（CPU dry run）
- `run_experiments.py` 内 suite 定义（reference 配对比较）

均不影响上一阶段默认配置下的核心训练逻辑（默认参数与原有 59+26+10 项测试全部通过）。

## 23. 完整复现命令

```bash
python run_experiments.py --prepare-proxy
for suite in dry_run main_anomaly_binary main_multiclass ckr_ablation \
             metric_ablation ema_ablation generator_init_ablation \
             backprop_ablation rwr_ablation contrastive_ablation \
             ce_ablation distill_ablation source_ablation; do
  python run_experiments.py --suite $suite
  python run_experiments.py --aggregate --suite $suite
  python run_experiments.py --plots --suite $suite
done
# 测试
python test_all.py && python test_smoke.py && python test_experiments.py
```

## 结论（仅依据上述数据）

1. **加权 CE 是异常检测主效因**：pooled PR-AUC 从 23.95 → 58.65（+34.7pp，
   5/5 种子，CI 排除 0）；anomaly recall 从 0 → 26.1
2. **跨视图对比学习进一步显著提升**：+6.09pp（CI [1.23, 10.95]，5/5 种子）
3. **蒸馏与动态 CKR 在 5 种子下无统计显著差异**：B4≈B5（Δ+0.03，CI 跨 0），
   A1 三模式、A3 EMA、A2 指标选择、A7 权重方案均无显著差异——
   机制上由异常类 reliability support 不足（40% 单元回退）解释
4. **multiclass 上 Full 方法显著优于纯 FedAvg**（accuracy +24.3pp，CI 排除 0）
5. **checkpointed 与 full 梯度等价**（指标完全一致）；本小模型下显存收益不可观测
6. **代理预训练、RWR 缓存：无显著收益**（分别 CI 跨 0 / 命中率仅 5%）
7. **validation 泄漏对照**：3 种子下与 holdout 无显著差异（不推荐，仅对照）
