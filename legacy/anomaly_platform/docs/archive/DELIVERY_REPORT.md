# 交付报告：动态 CKR + 最佳模型 checkpoint + RWR 缓存 + 扩散生成器显存优化 + 可选公开代理数据预训练

日期：2026-07-31

## 1. 修改前 CKR 为什么是静态

修改前 `compute_ckr()`（`train_fedtad.py` 的 S1）只做一件事：对每个客户端每个训练节点，
计算节点特征+拓扑嵌入与邻居的余弦相似度，按类别累加得到一个
`[num_clients, num_classes]` 的静态拓扑可靠性矩阵，缓存到 `./ckr/*.pt` 并永久复用。
该矩阵只依赖固定节点特征与图结构，**不依赖模型，不随训练轮次变化**，因此称为静态 CKR。
服务端生成器语义损失/分歧损失/蒸馏损失对它的归一化结果直接加权，无法反映
"本轮这个客户端对类别 c 的真实表现如何"，低资源客户端与类别缺失场景会被固定权重误导。

## 2. 动态 CKR 的准确公式

设 k=客户端, c=类别, t=通信轮次（1-based）：

```
R_dynamic[k,c,t] = α · R_topology_scaled[k,c] + (1-α) · metric_local[k,c,t]   (hybrid_dynamic)
R_ema[k,c,t]     = γ · R_ema[k,c,t-1] + (1-γ) · R_dynamic[k,c,t]               (首轮直接取 candidate)
R_server[k,c,t]  = R_ema[k,c,t] / Σⱼ R_ema[j,c,t]                              (按类别在参与客户端维度归一化)
```

- `R_topology_scaled`：静态 CKR 经 `scale_static_ckr()` 缩放，∈[0,1]（max/minmax/rank，默认 max）
- `metric_local`：`--dynamic_ckr_metric` 指定（f1/recall/confidence，默认 f1）
- 参数：`--dynamic_ckr_alpha`（默认 0.5）、`--dynamic_ckr_ema_decay`（默认 0.8）
- **不得**直接用未缩放静态 CKR 与 [0,1] 的 F1 相加（静态 CKR 是累加值，量纲随类别样本数变化）

## 3. 为什么使用 reliability holdout 而不是默认 val

动态 CKR 会参与服务端训练（生成器/蒸馏权重），若用正式验证集标签计算，
验证集就成了训练信号——模型选择指标会被污染。
因此每个客户端从原 `train_idx` 内部再分层划分出 `fit_idx`（加权 CE + 梯度更新）与
`reliability_idx`（只算动态 CKR，不参与任何 backward/CE/参数更新），
val_idx 只用于模型选择与早停，test_idx 只用于最终评价。
默认 `--dynamic_ckr_eval_source reliability_holdout`；只有显式 `validation` 才允许用 val_idx，
并打印：`WARNING: validation labels are being used to update CKR and are no longer a clean model-selection set.`

## 4. 每轮 CKR 数据流

严格顺序（与实现代码一致）：

1. 服务器广播当前全局模型
2. 客户端在 fit_idx 上本地训练（加权 CE + 子图-子图跨视图 InfoNCE）
3. 本地训练结束
4. 客户端在 reliability_idx 上计算每类别 F1/召回率/置信度（eval + no_grad，无 backward）
5. 客户端**仅上传**：每类别分数 + 每类别 support + available mask（不上传节点/标签/预测明细/Data）
6. 服务器更新 `DynamicCKRTracker`（EMA 递推 + 回退 + 按类别归一化）
7. FedAvg 得到 initial global model
8. 生成器更新阶段使用**本轮**动态 CKR（伪标签采样与损失加权）
9. 全局蒸馏阶段使用**本轮**动态 CKR
10. 得到 corrected global model
11. 干净 val_idx 做模型选择（`--selection_metric`，默认 multiclass=macro_f1 / anomaly_binary=pooled_pr_auc）
12. 进入下一轮

未参与本轮训练的客户端保持原 EMA，不强制清零。

## 5. 类别缺失时如何回退

每类别的 `available = (reliability support >= --reliability_min_support)` 且指标非 NaN。
unavailable 时：

- 已有历史（`seen=True`）：保持上一轮 EMA（reason=`previous_ema`）
- 第一轮无历史：
  - hybrid/static → 静态拓扑先验（reason=`first_round_static`）
  - dynamic_only → 均匀权重 1/K（reason=`first_round_uniform`）
- 指标 NaN → reason 前缀 `nan_metric_*`
- 不得把 unavailable 当 F1=0，不得产生 NaN（`get_server_weights` 对 NaN/全零列做均匀回退并打印警告）

## 6. EMA 如何更新

`seen[k,c]` 标记该单元是否已有动态历史（替代脆弱的浮点相等判定）：

- 首次 available：`ema = candidate`，置 seen=True
- 后续 available：`ema = γ·ema_prev + (1-γ)·candidate`
- 任何 unavailable：ema 值不变（回退语义见上）
- 恢复 checkpoint 后 `seen`/`ema` 随 `state_dict()` 载入，轮次连续，EMA 链不断

## 7. 三种 CKR 模式的区别

| 模式 | candidate 公式 | 缺失回退 | 用途 |
|---|---|---|---|
| `static_topology` | 恒为 `R_topology_scaled` | 恒静态 | 消融基线 |
| `dynamic_only` | 恒为 `metric_local` | 上一轮 EMA / 均匀权重 | 动态消融 |
| `hybrid_dynamic` | `α·静态 + (1-α)·动态` | 上一轮 EMA / 静态先验 | 默认 |

单元测试在构造数据上证明三模式输出两两不同（test 34）。

## 8. 最佳模型 checkpoint 保存内容

`util/checkpoint.py::save_checkpoint` 每次验证主指标提升且非 NaN 时保存 `best.pt`（可选 `last.pt`）：

- global_model_state_dict / generator_state_dict
- global_optimizer / generator_optimizer state_dict
- DynamicCKRTracker state_dict（含 EMA/seen/support/round）
- communication_round、best_validation_metric、task_mode、num_classes、normal/anomaly_classes
- selection_metric、feat_dim、生成器配置（diffusion_steps/hidden/betas/output_bound）
- 全部关键超参数（args）、随机种子
- Python random / NumPy RNG / PyTorch CPU RNG / CUDA RNG state

## 9. 训练结束是否加载 best

是。训练循环结束后，`find_final_checkpoint()` 优先返回 `best.pt`（否则最新 `round_N.pt`），
`load_checkpoint()` 载入 global_model 后在 test_idx 上输出最终指标
（打印 `[Final Test] 加载 best.pt (round N)`）。
恢复训练（`--resume_checkpoint`）时校验 task_mode/num_classes/feat_dim 元数据，
不兼容抛 `RuntimeError` 明确报错；RNG 状态恢复到保存时刻，轮次/优化器/CKR EMA 连续
（修复：`torch.load(map_location='cuda')` 会把 RNG 状态张量搬到 GPU，`set_rng_state` 前需 `.cpu()`）。

## 10. RWR 缓存了什么、没有缓存什么

`util/rwr_cache.py::RWRSubgraphCache` 只缓存**采样结构**：

- 缓存：`sampled_global_node_ids`、`center_local_idx`（view 的 local_edge_index 由结构现场重建）
- 不缓存：`x_subgraph`、`conv1` 输出、readout 表示、任何含计算图的张量
- 防御：`put()` 对 `requires_grad=True` 的张量抛 ValueError

## 11. view_2 缓存如何失效

- view_1（原始图）：结构固定，`--rwr_cache_view1_persistent enabled` 时 key 不含 round，
  跨本地 epoch、跨通信轮命中
- view_2（扰动图）：key 含 `round_idx` 与 view_2 的 edge hash，新一轮新视图自动 miss；
  `invalidate_round()` 可主动清理
- key 组成：`client_id + view + round + edge_hash + anchor + subgraph_size + restart_prob + rwr_seed`
- 容量 `--rwr_cache_max_entries`（默认 4096），满时逐出最早插入项；`--rwr_cache disabled` 可关闭
- 输出 hit/miss/rate（每轮打印）

## 12. 三种生成器反向传播模式

`--generator_backprop_mode`（默认 checkpointed）：

- `full`：保留当前完整计算图（严格基线）
- `checkpointed`：每一步 `self.forward` 用 `torch.utils.checkpoint(use_reentrant=False)`
  激活重计算，生成器参数梯度完整（默认推荐）
- `truncated`：每隔 `--generator_truncate_interval` 步对采样状态 `detach()`，
  截断反向传播，梯度不回传到更早采样状态（低显存近似，**不与 full 等价**，代码注释明示）

初始 x_T 由 `randn` 生成、无需 `requires_grad`；生成器参数梯度来自每一步 eps_pred 对参数的依赖，三模式均非零。
训练循环记录生成器阶段峰值显存（`torch.cuda.reset_peak_memory_stats`），CPU 环境跳过 CUDA 显存日志。

## 13. 公开代理预训练如何保证不访问私有数据

`pretrain_diffusion.py` 独立脚本，只使用本地生成的合成代理特征
（`make_synthetic_proxy_data`：逐类高斯簇，无需联网），不读取任何客户端私有子图。
标准 DDPM：`x_t = √ᾱ·x0 + √(1-ᾱ)·ε`，`L_diff = MSE(ε_θ(x_t,t,[label]), ε)`。
`proxy_feature_dim != target_feature_dim` 时用显式 `FeatureAdapter`（MLP，禁止截断/补零/reshape），
参数随 checkpoint 保存。公开类别与目标语义不一致时默认 `unconditional` 预训练
（生成器 num_classes=1，不假设类别 ID 语义），联邦阶段通过教师语义损失注入目标类别信息。
联邦加载（`--generator_init proxy_pretrained`）校验 `target_feature_dim` 与 `num_classes`（conditional），
维度不兼容明确报错；只加载形状匹配的参数（unconditional 时 class_embed 允许跳过）。

## 14. 修改的文件和函数

- `train_fedtad.py`：main 流程重构（fit/reliability 划分、每轮动态 CKR、resplit、RWR 缓存、
  生成器 backprop 模式、checkpoint 保存/恢复/最终加载、selection metric、零异常审计）；
  `evaluate_client` 增加 eval_mask；新增 `evaluate_global`、`resolve_selection_metric`；
  `subgraph_contrastive_step` 增加 rwr_cache/client_id/round_idx/rwr_seed 参数；
  `compute_ckr` 缓存 key 区分 resplit
- `model.py`：`ConditionalDiffusionGenerator.differentiable_sample` 增加
  `backprop_mode`/`truncate_interval`
- `util/task_util.py`：`rwr_subgraph_sampling` 增加 seed 参数（确定性采样）
- `util/dynamic_ckr.py`：`scale_static_ckr`（全零类警告）、`compute_per_class_metrics`/别名、
  `DynamicCKRTracker` 重写（seen 掩码、candidate/fallback 跟踪、JSONL 日志、anomaly 摘要）
- `util/checkpoint.py`：元数据校验 `validate_checkpoint_meta`、`find_final_checkpoint`、RNG 恢复 .cpu()
- `util/rwr_cache.py`：key 含 client/view/round、`invalidate_round`、`stats`、no-grad 防御
- `util/data_split.py`：`stratified_reliability_split`（seed 生效）、新增 `stratified_split`、`split_report`
- `pretrain_diffusion.py`：**新文件**（标准 DDPM 预训练 + FeatureAdapter + checkpoint 校验/加载）
- `test_all.py`：新增 test 43-59；`test_smoke_train.py`：**新文件**（4 类训练 smoke）

## 15. 新增参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckr_mode` | `hybrid_dynamic` | static_topology / dynamic_only / hybrid_dynamic |
| `--dynamic_ckr_eval_source` | `reliability_holdout` | reliability_holdout / validation（后者打印警告） |
| `--reliability_holdout_ratio` | 0.2 | train 中划出 reliability 的比例 |
| `--reliability_split_seed` | 42 | 划分种子（+client_id） |
| `--reliability_min_support` | 3 | 低于该值类别动态指标 unavailable |
| `--dynamic_ckr_metric` | f1 | f1 / recall / confidence |
| `--dynamic_ckr_alpha` | 0.5 | 静态先验权重 |
| `--dynamic_ckr_ema_decay` | 0.8 | EMA 平滑系数 |
| `--static_ckr_scaling` | max | max / minmax / rank |
| `--resplit_after_label_mapping` | true | anomaly_binary 映射后按新标签重分层划分 |
| `--checkpoint_dir` | '' | 最佳模型保存目录（空=不启用） |
| `--resume_checkpoint` | '' | 从 checkpoint 恢复训练 |
| `--save_last_checkpoint` | false | 每轮保存 last.pt |
| `--selection_metric` | '' | 默认 multiclass=macro_f1 / anomaly_binary=pooled_pr_auc |
| `--rwr_cache` | enabled | enabled / disabled |
| `--rwr_cache_max_entries` | 4096 | 缓存容量 |
| `--rwr_cache_view1_persistent` | enabled | view_1 跨轮缓存 |
| `--rwr_seed` | 0 | RWR 采样确定性种子 |
| `--generator_backprop_mode` | checkpointed | full / checkpointed / truncated |
| `--generator_truncate_interval` | 2 | truncated 每 N 步 detach |
| `--generator_sampling_steps` | 0 | 0=用 diffusion_steps |
| `--generator_init` | teacher_guided_scratch | teacher_guided_scratch / proxy_pretrained |
| `--proxy_checkpoint` | '' | 代理预训练权重路径 |
| `--log_dir` | ./logs | 动态 CKR JSONL 日志目录 |

（`pretrain_diffusion.py` 另有 --proxy_dataset/--proxy_root/--proxy_pretrain_mode/--proxy_epochs/
--proxy_lr/--proxy_checkpoint/--proxy_feature_dim/--target_feature_dim 等。）

## 16. 新增测试和运行结果

**单元测试（`python test_all.py`，共 59 项，全部通过）：**
- 原 1-42 项（含既有 26 项对应的全部核心测试）不变通过
- 新增 43-59：
  43 缺失类别不置零 / 44 默认不用 val 评估动态 CKR / 45 结束加载 best 优先于 last /
  46 checkpoint 保存生成器 / 47 view_2 轮内失效 + view_1 跨轮命中 / 48 缓存拒绝含梯度张量 /
  49 三模式生成器梯度非零 / 50 checkpointed 参数更新 / 51 truncated 截断但不断梯度 /
  52 合成代理 DDPM 冒烟（L_diff 有限 + checkpoint 往返）/ 53 维度不匹配明确报错 /
  54 映射后按二分类重分层划分 / 55 零异常 reliability 安全回退 + 加权 CE 权重为 0 /
  56 缓存前后 InfoNCE 数值一致（固定 seed）/ 57 缓存后 center_local_idx 正确 /
  58 split_report 回退原因 / 59 dynamic_only 首轮均匀回退

**冒烟测试（`python test_smoke.py`）：26 项全部通过。**

**训练冒烟（`python test_smoke_train.py`）：A/B/C/D 全部通过，见下。**

## 17. multiclass / anomaly_binary smoke test

环境：Cora（2 clients，Louvain），GPU，`python test_smoke_train.py`。以下为真实运行结果。

**A. multiclass（hybrid_dynamic + checkpointed 生成器 + 保存/恢复 checkpoint）**
- phase1（2 轮）保存 best.pt/last.pt，JSONL 轮次 {1,2}；phase2 `--resume_checkpoint best.pt`
  `--num_rounds 3` 从 round 3 继续（stdout `[Resume] 从 round 3 继续训练`），JSONL 轮次 {3}
- EMA 连续性跨 checkpoint 边界验证：11 个 (client,class) 单元满足
  `ema(t) = 0.8·ema(t-1) + 0.2·(0.5·static_scaled + 0.5·f1)`（round 2→3）
- 最小类别回退示例：`class 6: static_raw=2.4791 static_scaled=0.6175 support=2 f1=nan
  available=False candidate=0.6175 ema=0.6175 server_w=0.4401 fallback=True (nan_metric_first_round_static)`
- 结束自动加载 best.pt（round 2）：`[Final test] acc=70.44 macro_f1=64.72`
- 生成器阶段峰值显存 ~39-41 MB（小模型，仅验证日志路径）

**B. anomaly_binary（normal=[0,2,3,4,5]，anomaly=[1,6]，hybrid + reliability holdout）**
- 映射后重划分执行（`[Resplit after label mapping]`）
- 强制回退：`--reliability_min_support 5`，client 1 异常类 support=3 < 5 →
  `fallback=True (nan_metric_first_round_static)`，ema=0.4046（静态先验，非 0 非 NaN）；JSONL 共 2 条回退记录
- 异常 CKR 摘要：`client 0: normal_ckr=0.8406 anomaly_ckr=0.8943
  anomaly_reliability_support=12 zero_anomaly_reliability_split=False`；
  `client 1: normal_ckr=0.9950 anomaly_ckr=0.4046 anomaly_reliability_support=3 ...`
- 无 val 警告（默认 reliability_holdout）；val pooled_pr_auc=78.85 为 best（round 1）；
  结束加载 best.pt：`[Final test] pooled_auc=93.39 pooled_pr_auc=77.22 weighted_client_auc=93.59`

**C. static_topology 消融**
- 1 轮可运行；所有 candidate 恒为 static_scaled（如 `candidate=1.0000 ema=1.0000 fallback=True (static_mode)`），
  与 hybrid（candidate=0.5·static+0.5·f1）数值明确不同
- 结束加载 best.pt：`[Final test] acc=51.82 macro_f1=44.38`

**D. synthetic proxy pretrain（proxy 32 维 → 目标 1433 维，FeatureAdapter 生效）**
- `pretrain_diffusion.py --proxy_feature_dim 32 --target_feature_dim 1433 --proxy_epochs 4`：
  L_diff 首末 `1.010415 -> 1.005852`（有限，轻微下降）
- 联邦训练 `--generator_init proxy_pretrained` 加载成功（`加载代理 DDPM 预训练权重`），
  维度校验通过，1 轮训练正常；结束加载 best.pt：`[Final test] acc=51.73 macro_f1=44.39`

说明：以上数值为 smoke 配置（1-2 轮、小模型）下的真实运行结果，
仅验证功能路径正确，**不构成性能提升声明**。

## 18. 仍存在的研究限制

1. 动态 CKR 每轮在 reliability holdout 上评估，holdout 占比（0.2）缩小了本地训练样本，
   极小客户端上所有类别都可能 unavailable（全部回退，动态信息为零）
2. `truncated` 反向传播与 `full` 不等价，超参数 `generator_truncate_interval` 无理论最优
3. 代理预训练是类别无语义的高斯合成数据；真实公开数据集（如带语义类别的 Planetoid）
   未接入，unconditional 模式与目标类别语义的桥接完全依赖教师蒸馏
4. checkpoint 最佳模型以 val 指标选择，val 上的过拟合轮次仍可能被选中（与标准早停同源问题）
5. RWR 缓存命中率依赖锚点重复度（锚点每轮随机抽样）；低 `contrastive_batch_size` 下命中率有限
6. 动态指标仅 F1/召回率/置信度三选一，未做多种指标融合
