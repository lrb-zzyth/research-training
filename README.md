# FedTAD — 联邦图学习研究实现（FedTAD 改进方法 + 正式实验战役）

> 方法 = **FedTAD**(IJCAI 2024, arXiv:2404.14061) + 两个自研组件：
> ① 客户端**子图-子图跨视图对比学习**（InfoNCE）；② 服务端**条件 DDPM 伪图生成 + CKR 加权 KL 无数据知识蒸馏**。
> 目标：在 Cora / CiteSeer / PubMed / CS / Physics × 5/10/20 客户端三档上全面超越 FedTAD 论文 Table 2。
> 附 Web 可视化训练/实验管理平台。

本文档对应仓库当前状态。详细文档：
- 平台使用：[docs/PLATFORM_GUIDE.md](docs/PLATFORM_GUIDE.md)
- 训练参数：[docs/TRAINING_PARAMETERS.md](docs/TRAINING_PARAMETERS.md)
- 架构：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API：[docs/API.md](docs/API.md)
- 复现：[docs/REPRODUCTION.md](docs/REPRODUCTION.md)
- **正式战役协议与服务器迁移**：`SERVER_MIGRATION_BEGINNER.md`、`server_migration_manifest.json`

---

## 0. 版本更新说明（本次提交）

> 本次提交 = **最终 correctness 修复 + DDPM 机制修复 + 正式战役 v3 协议 + 服务器迁移准备** 的完整收口。
> 自上一版（`6318224`）以来累积：4 个文件改动 + 新增 tests/、formal campaign runner、迁移脚本与文档。

### 0.1 变更总览

| 模块 | 主要变更 |
|---|---|
| **对比增强修复** | `edge_perturbation` 无向性 bug 修复：按无向边为单位删/加并展开双向（原实现按有向条目删/加，第二视图被削成部分有向图） |
| **v3 数学修复** | KL 统一为 FedTAD Eq.(10)/专利 S4.2 的 `KL(global‖local)`；RWR 不再用全图随机节点补足局部子图，只在 anchor 连通分量内扩展；radius projection 改为等价但抗 float32 范数溢出的稳定实现。旧 v2 tuning 结果不得与 v3 混用。 |
| **DDPM 三大根因修复** | ① 去噪网络信息瓶颈 → `timestep_scalar` 直通残差 `eps_pred = exp(g_t)·x_t + residual`（`--diffusion_skip_mode`）；② 联邦扩散预训练欠训练 → 样本加权损失统计 + S2 逐层诊断；③ reverse 链爆炸 → posterior variance β̃_t（`--use_posterior_variance`） |
| **radius + anchor 机制** | pretrained radius bank 流形约束（`--radius_constraint`，`x = r_c·z/‖z‖` 堵幅度作弊）+ L2-SP 参数锚（`--lambda_diffusion_anchor`）+ Stage1 冻结 `c(t)`（`--diffusion_freeze_skip_scale`）+ Stage2 warm-up（`--server_start_round`） |
| **优化器生命周期** | 新增 `--local_optimizer_lifecycle`（`persistent` 旧行为 / `reset_each_round` 正式主线）：persistent Adam + 每轮广播的陈旧动量是 PubMed-10 崩塌根因（peak drop −15.4 → −3.2） |
| **RNG 协议 v2** | 客户端 round/client 确定性重播种 + 服务端专用 generator，配对实验 C/B 严格可比；定位并记录 CUDA scatter 类算子的跨进程数值非确定性 |
| **正式战役 v3** | `experiments/accuracy_benchmark/formal_campaign.py`：每 (dataset,tier) 独立 Optuna study、validation-only objective、test isolation（`--tuning_mode` 调参期零 test 计算）、formal health v2（任何 nonfinite → trial FAIL 且排除候选）、protocol/data-identity 校验与安全 resume；单 GPU 正式协议固定 `--n_jobs 1` 严格串行。决赛用 `formal_final_eval.py`，每轮只看 validation，加载 val-best 后 test 仅计算一次。 |
| **formal guard** | `--formal_campaign_guard`：启动校验 9 项正式协议（weighted CE ON / holdout 0 / accuracy 选轮 / reset optimizer / radius ON / align OFF / skip 冻结 / 不二次预训练 / Stage1 checkpoint 存在），违反即 ValueError |
| **搜索空间 v2** | `lambda_sem ∈ {0.01,0.1,1.0}`（v1 的 0.001 因 Cora-5 实测 8/9 nonfinite/raw-runaway 移除）；`distill_steps ∈ {1,3,5}`（25 为 legacy 剔除）；新增 `server_start_round ∈ {0,1}` |
| **测试** | 新增 `tests/`（generator 冻结 / checkpoint 状态管理 / resume freeze 策略 / Stage2 smoke / paired RNG / formal campaign 守卫与 test isolation 等 6 套件） |
| **服务器迁移** | `server_migration_manifest.json`、`environment.yml` + `requirements_frozen.txt`（已验证环境导出）、`scripts/server_smoke_test.sh`、`scripts/run_formal_cell.sh`、`SERVER_MIGRATION_BEGINNER.md`（从零租服务器指南） |

### 0.2 破坏性变更：命令行参数重命名（仍适用，自 `c434f91` 起）

| 旧参数（≤ `c434f91`） | 新参数（当前版本） | 说明 |
|---|---|---|
| `--contrastive_type {supcon,node_node,gradate}` | `--contrastive_mode {subgraph_cross_view,none}` | 三种对比模式统一为跨视图子图对比 |
| `--use_contrastive` | `--contrastive_mode` | 布尔开关 → 枚举 |
| `--lambda_cl` | `--lambda_subgraph` | 对比损失权重（默认 0.1 不变） |
| `--cl_tau` | `--contrastive_temperature` | InfoNCE 温度（默认 0.5 不变） |
| `--generator_type {mlp,diffusion}` | `--generator_init {scratch,proxy_pretrained}` | 生成器固定为扩散式，该参数语义改为「初始化方式」 |
| `--num_gen` | `--fake_nodes` | 每轮生成伪节点数（默认 100 不变） |
| `--topk` | `--knn_k` | KNN 伪图近邻数（默认 5 不变） |
| `--it_g` / `--it_d` | `--generator_steps` / `--distill_steps` | 生成器 / 蒸馏步数（默认 1 / 5 不变） |
| `--lr_g` | `--generator_lr`（另新增 `--distill_lr`） | 生成器 / 蒸馏学习率 |
| `--lam1` / `--lam2` | `--lambda_sem` / `--lambda_diversity` | 语义 / 多样性损失权重 |
| `--fedtad_mode {raw_distill,rep_distill}` | 已移除 | 蒸馏统一在伪图特征空间进行 |
| `--glb_epochs`、`--num_dims` | 已移除 | — |
| `--gradate_*`（8 项） | 已移除 | 随 `gradate` 对比模式一并删除 |

**默认值变化**（不改命令也会改变行为，务必注意）：

| 参数 | 旧默认 | 新默认 |
|---|---|---|
| `--use_weighted_ce` | 关（`store_true`） | `BooleanOptionalAction` + task-mode 自动；正式 multiclass campaign 显式 **ON** |
| `--class_weight_method` | `effective_num` | **`inverse`** |
| `--diffusion_steps` | 50 | **20** |

因此 **B1 基线必须显式加 `--no-use_weighted_ce`**（见 §8.4）。

### 0.3 未纳入版本控制的内容

clone 后仓库**不含**任何实验结果与 checkpoint，以下内容需自行生成（均已在 `.gitignore` 中）：

| 内容 | 说明 |
|---|---|
| `dataset/` 除 Cora + `Client10/Louvain` 外的数据 | CiteSeer、`Cora/Client2`、`Louvain_enriched*`、`Louvain_frozen*` 等切分变体较大，由划分脚本按需重新生成 |
| `runs/` | 训练产物（checkpoint / 日志 / `metrics.jsonl` / `events.jsonl` / `final_metrics.json`），各机器本地生成 |
| `ckr/` | 静态 CKR 磁盘缓存，运行时自动生成 |
| `references/` | 外部参考实现（原版 FedTAD 见 <https://github.com/xkLi-Allen/FedTAD>，GRADATE 见其原仓库） |
| `.env` | 按 `research-training/backend/.env.example` 自行创建 |

复现请从 §8 的训练命令重新跑。

---

## 1. 解决的问题

FedTAD 解决的是 **联邦图学习（Federated Graph Learning, FGL）** 中的 **非独立同分布（Non-IID）异构性** 问题。

在联邦图学习中，全局图被社区发现算法（Louvain / Metis）切分成若干子图，每个客户端持有一个子图。由于社区天然具有标签分布不均衡和结构异质性，各客户端子图存在严重的 Non-IID 数据漂移（label shift + structure shift），导致 naive FedAvg 性能严重下降。

FedTAD 的核心思路是：**服务端通过一个可训练的生成器合成伪图（synthetic graph），用该伪图对齐各客户端的局部模型与全局模型的输出分布，从而缓解异构性。**

---

## 2. 环境与依赖

**硬件环境**：Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, NVIDIA GeForce RTX 3090 with 24GB memory（原始论文）；
当前版本已在 RTX 5060 Laptop GPU (8GB) 上验证通过。

**软件环境**：
- Python 3.9+, PyTorch 2.0+, CUDA 11.8+
- PyTorch Geometric (PyG) 2.3+
- 依赖包：`pip install -r requirements.txt`

安装步骤：
1. 参照 [PyTorch](https://pytorch.org/get-started/locally/) 和 [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) 安装基础环境；
2. `pip install -r requirements.txt` 安装其余依赖；
3. 若使用 Metis 分区，需额外安装 `metispy`。

---

## 3. 数据流与预处理

### 3.1 数据集加载

`FGLDataset`（`util/fgl_dataset.py`）支持 Cora、CiteSeer、PubMed、ogbn-arxiv、ogbn-products、CS、Physics、Computers、Photo、NELL、Reddit、Flickr 等多种图数据集。

完整图从 `torch_geometric.datasets` 加载后，自动检查是否已有切分好的子图缓存，若无则执行图切分并保存。

### 3.2 图切分

`data_partition()`（`util/base_data_util.py`）提供两种分区策略：

| 分区策略 | 说明 |
|---|---|
| **Louvain**（默认） | 运行 Louvain 社区发现算法，将检测到的社区合并到 `num_clients` 个子图中（保持图结构完整性） |
| **Metis** | 调用 Metis 图划分算法直接分割 |

每个子图内的节点按类别比例分配 `train / val / test` 的 boolean mask（`train_idx`、`val_idx`、`test_idx`），子图保存为 `.pt` 文件。

---

## 4. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  Server                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Generator (MLP / Diffusion)                    │   │
│  │  → 合成伪节点特征 node_logits                    │   │
│  │  → kNN 构造伪图 pseudo_graph                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                │                        │
│  ┌─────────────────────────────┴─────────────────────┐ │
│  │  Local Models → Global Model（蒸馏对齐）           │ │
│  │  语义损失 + 分歧损失 + 多样性损失                   │ │
│  └───────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │ broadcast / aggregate
┌────────────┬─────────────┼──────────────┬──────────────┐
│ Client 0   │ Client 1    │ ...          │ Client N-1   │
│ GCN Subgr0 │ GCN Subgr1  │              │ GCN SubgrN-1 │
│ train/val  │ train/val   │              │ train/val    │
└────────────┴─────────────┘              └──────────────┘
```

---

## 5. 核心组件

### 5.1 CKR — Class-wise Knowledge Rate

CKR 是一个 `[num_clients, num_classes]` 矩阵，衡量每个客户端在每个类别上的知识贡献度。对于每个客户端的每个训练节点，计算其与邻居的余弦相似度均值作为该节点的「知识率」，按标签累加。归一化后的 `normalized_ckr` 用于加权语义损失和分歧损失中各类别的贡献权重。

### 5.2 生成器

#### 教师引导扩散式生成器（ConditionalDiffusionGenerator / TeacherGuidedDiffusionGenerator）— 核心

条件扩散式伪节点生成器，从高斯噪声与类别条件出发，在冻结的客户端教师模型与当前全局模型反馈下训练。实现为条件 DDPM：Stage1 在客户端真实特征上执行标准前向加噪/噪声预测联邦预训练，Stage2 在服务器端进行教师引导的对抗精调并生成伪节点特征。

| 项目 | 说明 |
|---|---|
| 输入 | 高斯噪声 `x_t` + 时间步 `t` + 生成类别条件 `labels` |
| 结构 | 时间嵌入 + 类别嵌入 → 噪声预测 MLP |
| 训练 | 由生成类别条件 + 冻结教师模型/全局模型反馈（语义/分歧/多样性损失）引导，梯度可回传到生成器与伪特征 |
| 推理 | `@torch.no_grad()` 采样伪节点特征，供 KNN 构图与蒸馏 |
| 输出 | `[fake_nodes, feat_dim]` 合成节点特征 |

通过 `--generator_init scratch`（默认，随机初始化）或 `proxy_pretrained`（公开代理数据预训练，实验性）初始化。

### 5.3 GCN 模型

两层 `GCNConv`，`hid_dim` 默认为 64：

- `forward(data)`：`conv1 → ReLU → Dropout → conv2 → logits`
- `rep_forward(data)`：直接从 conv2 开始（用于 representation distillation）

---

## 6. 训练流程（每轮）

每轮训练包含以下阶段：

### 阶段 1：全局模型广播

服务端将 `global_model` 的权重复制给所有 `local_models`。

### 阶段 2：全局评估

服务端对全局模型在各客户端子图的 val/test 上做加权评估（按客户端节点数加权；只用于模型选择与报告，不进入任何训练权重）。

### 阶段 3：客户端本地训练

每个客户端在自己的子图上训练 GCN：

```
loss = CrossEntropyLoss(logits[train_idx], labels[train_idx])
```

**核心改进**（通过 `--use_weighted_ce`、`--contrastive_mode subgraph_cross_view` 启用）：

```
loss = weighted_ce_loss + lambda_subgraph × subgraph_contrastive_loss
```

| 改进 | 说明 |
|---|---|
| **Weighted CE** | 仅使用客户端 train 标签计算类别权重。支持 `inverse`（默认，`weight_c = N/(num_classes × n_c)`）和 `effective_num`（`weight_c = (1-β)/(1-βⁿᶜ)`）两种方法。缺失类别权重置 0（避免 NaN/除零），非零权重归一化到均值约 1 |
| **Subgraph-Subgraph Cross-View Contrastive Loss**（`--contrastive_mode subgraph_cross_view`，核心）| 自监督子图-子图跨视图对比：本地边扰动构造两个图视图，RWR 采样每个中心节点的局部子图副本，中心节点特征在局部副本中置零，共享 GCN 编码后 mean readout。 当 RWR 在碎片连通分量中无法达到目标大小时，只在 anchor 所在连通分量内 BFS 扩展；连通分量不足则接受可变大小子图，绝不从其它连通分量随机补节点。同一中心节点跨视图为正样本，不同中心节点为负样本，InfoNCE 损失。正负样本构造不读取标签 |
| **Static Topology CKR**（`--ckr_mode static_topology`，核心）| 基于固定节点属性与本地图拓扑计算类别知识可靠性，作为服务端融合客户端教师知识的权重。不依赖任何验证/测试指标，不随时间变化 |
| **Teacher-Guided Diffusion-Style Pseudo-Graph Distillation**（`--distill_weighting static_ckr`，核心）| 服务器从高斯噪声与类别条件出发，使用冻结的客户端教师模型反馈训练扩散式伪节点生成器；以 KNN 依据伪节点特征构造伪图；按 CKR 融合教师知识后，在伪图上对全局模型做 KL 蒸馏（生成器阶段只更新生成器，蒸馏阶段只更新全局模型） |

### 阶段 4：服务端蒸馏

这是 FedTAD 的核心创新，分两个子阶段：

#### 子阶段 4a：训练生成器（`generator_steps` loop）

1. **伪标签采样**：按 `--fake_class_strategy`（默认 balanced）采样生成类别条件
2. **生成伪节点特征**：`generator.differentiable_sample(labels)` → `fake_x`（从高斯噪声出发，梯度经完整计算图回传）
3. **构造伪图**：`build_knn_graph(fake_x, k=--knn_k)` — 使用伪节点特征相似度构图，不使用客户端真实 edge_index；离散选邻居时使用 `fake_x.detach()`
4. **教师反馈**：冻结的 local_models（教师）与 global_model 在伪图上推理
5. **三种损失**：

   | 损失 | 公式 | 作用 |
   |---|---|---|
   | **语义损失** | CKR 加权 CE（教师预测对齐伪标签）| 使生成特征可分 |
   | **分歧损失** | `KL(global‖local)`（CKR 加权；生成器最大化）| 挖掘全局/本地模型分歧节点 |
   | **多样性损失** | 特征多样性正则 | 鼓励生成特征多样化 |

6. **生成器更新**：只更新生成器参数（`set_requires_grad` 冻结全部教师与全局模型，置 eval；梯度仍能传回 `fake_x` 与生成器）

#### 子阶段 4b：训练全局模型（`distill_steps` loop）

固定生成器与客户端教师（冻结、eval），伪图由 `generator.sample` 在 `no_grad` 下生成；教师预测 detach；使用静态 CKR 加权教师分布，对全局模型最小化 `KL(global‖local)`（只更新全局模型），方向与 FedTAD Eq.(10) / 专利 S4.2 一致。

### 阶段 5：全局聚合

FedAvg：按客户端训练样本量（节点数）对参数做加权平均聚合；`no_grad` 下执行，不反向传播（`--fairness_mode` 非 none 时为实验性扩展）。

### 阶段 5：全局聚合

加权平均各客户端模型参数，加权系数为 `subgraph_size / total_size`。

---

## 7. 参数说明

### 任务模式

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task_mode` | anomaly_binary | 二分类异常检测（正常=0，异常=1）；`multiclass` 仅为兼容模式 |
| `--normal_classes` | 0,1,2,3 | anomaly_binary：正常类 ID（逗号分隔） |
| `--anomaly_classes` | 4,5,6 | anomaly_binary：异常类 ID（逗号分隔） |

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--seed` | 2024 | 随机种子（其余四类种子默认 -1，即跟随 seed） |
| `--model_seed` / `--partition_seed` / `--split_seed` | -1 | 模型/划分/切分种子（-1 回退到 seed） |
| `--root` | `./dataset` | 数据根目录 |
| `--dataset` | Cora | 数据集名称 |
| `--gpu_id` | 0 | GPU 编号 |
| `--num_clients` | 10 | 客户端数量 |
| `--num_rounds` | 100 | 联邦通信轮数 |
| `--num_epochs` | 3 | 客户端本地训练 epoch |
| `--local_epochs` | 0 | 本地 epoch 覆盖（0 = 使用 num_epochs） |
| `--partition` | Louvain | 图分区策略（Louvain / Metis），仅读取图拓扑 |

### 模型参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hid_dim` | 64 | GCN 隐藏层维度 |
| `--dropout` | 0.5 | Dropout 比率 |
| `--lr` | 1e-2 | 客户端学习率 |
| `--weight_decay` | 5e-4 | 权重衰减 |

### 类别加权交叉熵

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_weighted_ce` | True | 启用类别加权交叉熵（仅使用客户端 train 标签计算权重） |
| `--class_weight_method` | inverse | 加权方法（inverse / effective_num） |
| `--beta` | 0.999 | effective number 参数 |

### 子图-子图跨视图对比学习

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--contrastive_mode` | subgraph_cross_view | `subgraph_cross_view`（核心）/ `none` |
| `--edge_perturb_ratio` | 0.2 | 本地边扰动比例 |
| `--rwr_restart_prob` | 0.5 | RWR 重启概率 |
| `--rwr_subgraph_size` | 5 | RWR 局部子图采样大小 |
| `--contrastive_batch_size` | 64 | 对比锚点批大小 |
| `--contrastive_temperature` | 0.5 | InfoNCE 温度 |
| `--lambda_subgraph` | 0.1 | 对比损失权重 |
| `--contrastive_anchor_scope` | all_nodes | 锚点作用域（all_nodes / train_nodes） |

### 核心方法（CKR 与伪图蒸馏）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckr_mode` | static_topology | 静态拓扑 CKR（核心）；dynamic_only / performance_only / hybrid_dynamic 为实验性扩展 |
| `--distill_weighting` | static_ckr | `static_ckr`=B4 完整方法；`none`=B1/B2/B3 基线；`equal`=等权；`dynamic_ckr`=实验性扩展 |
| `--fake_class_strategy` | balanced | 伪节点类别策略（balanced / prior / reliability） |
| `--fake_nodes` | 100 | 每轮生成伪节点数 |
| `--knn_k` | 5 | KNN 伪图近邻数 |
| `--generator_steps` | 1 | 生成器更新步数 |
| `--distill_steps` | 5 | 全局蒸馏步数 |
| `--generator_init` | scratch | `scratch`（核心，随机初始化）/ `proxy_pretrained`（公开代理数据预训练，实验性） |

### 教师引导扩散式生成器

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--diffusion_steps` | 10 | 扩散步数 |
| `--diffusion_hidden` | 256 | 隐藏层维度 |
| `--diffusion_beta_start` | 1e-4 | beta 起始值 |
| `--diffusion_beta_end` | 0.02 | beta 终止值 |
| `--generator_lr` | 1e-3 | 生成器学习率 |
| `--distill_lr` | 1e-3 | 蒸馏学习率 |
| `--lambda_sem` | 1.0 | 语义损失权重 |
| `--lambda_diversity` | 0.1 | 多样性损失权重 |

### 双重终止机制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--f1_threshold` | 0.1 | 主指标连续三轮提升低于此值则正常收敛 |
| `--auc_threshold` | 1.0 | 10% 尾部客户端连续两轮增幅低于此值则提前终止 |

---

## 8. 训练示例

默认配置即核心完整方法（B4）：anomaly_binary + Weighted CE + 子图-子图跨视图对比 + 静态拓扑 CKR + FedAvg + 教师引导扩散式伪图蒸馏。

### 8.1 核心完整方法 B4（默认配置）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain
```

### 8.2 B3：Weighted CE + 子图对比（无伪图蒸馏）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --distill_weighting none
```

### 8.3 B2：Weighted CE（无对比、无蒸馏）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --distill_weighting none --contrastive_mode none
```

### 8.4 B1：纯 FedAvg 基线（普通交叉熵）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --distill_weighting none --contrastive_mode none --no-use_weighted_ce
```

### 8.5 显式指定完整方法（等价于 8.1）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --task_mode anomaly_binary --normal_classes 0,1,2,3 --anomaly_classes 4,5,6 \
  --contrastive_mode subgraph_cross_view --ckr_mode static_topology \
  --distill_weighting static_ckr --fake_class_strategy balanced \
  --generator_init scratch
```

### 8.6 历史 FedAvg 基线（legacy）

旧版独立基线脚本已移至 `legacy/train_fedavg.py`（B1 的等价实现，保留历史复现价值）：

```bash
python legacy/train_fedavg.py --dataset Cora --num_clients 2 --partition Louvain
```

---

## 9. 项目文件结构

```
FedTAD/
├── train_fedtad.py              # 核心训练主脚本（S1 静态 CKR → S2 本地训练 → FedAvg
│                                #   → 教师引导扩散式伪节点生成 → KNN 伪图 → CKR 加权蒸馏
│                                #   + [EVENT] 结构化事件 + 轮次边界参数热更新）
├── model.py                     # GCN（共享编码器）+ ConditionalDiffusionGenerator
├── pretrain_diffusion.py        # 公开代理数据预训练（默认关闭，实验性可选）
├── requirements.txt             # Python 依赖
├── util/                        # 核心工具（Louvain 划分 / 采样 / InfoNCE / CKR / checkpoint）
├── research-training/           # Web 可视化管理平台（FastAPI + Vue 3）
│   ├── backend/app/             # config_schema（canonical 参数契约）/ 路由 / 训练 runner
│   │                            # log_parser（结构化事件 + 正则 fallback）/ 数据库模型
│   ├── backend/test_backend_smoke.py   # 平台端到端 smoke（38+ 项）
│   ├── backend/test_param_contract.py  # 参数契约测试（16 项）
│   └── frontend/                # 前端应用（训练配置 / B1-B4 预设 / 监控 / 热更新面板）
├── test_all.py                  # 主测试套件（RWR / 生成器 / 蒸馏 / 指标 / 动态 CKR / checkpoint）
├── test_smoke.py                # 生成器与蒸馏模块 smoke test
├── test_smoke_train.py          # 真实训练 smoke test（Cora，2 客户端，少量轮次）
├── experimental/                # 实验性扩展（run_experiments 实验系统、阶段二/三测试）
├── legacy/                      # 历史脚本（train_fedavg.py 旧基线）
├── docs/                        # 架构 / API / 参数 / 平台指南 / 复现 / 历史审计归档
├── RELEASE_PLATFORM_REPORT.md   # 平台化改造验收报告（算法主线与平台功能总览）
├── louvain/                     # Louvain 社区发现算法实现（vendored）
├── ckr/                         # 静态 CKR 磁盘缓存（gitignored）
├── dataset/                     # 数据（Cora 全局图 + Louvain 客户端子图）
├── update.sh                    # 本地提交推送脚本
└── runs/platform/exp_<id>/      # 平台任务产物（gitignored）：checkpoints / 日志 / 事件
```

---

## 10. 引用

若使用本代码，请引用原论文：

```bibtex
@inproceedings{fedtad2024,
  title={FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning},
  author={...},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2024}
}
```

---

## 11. Web UI 可视化管理平台

项目包含一个**面向联邦图异常检测的可视化训练与参数调节平台**（`research-training/`），与核心算法同等重要、必须同时保留。

### 功能

- 用户注册/登录（JWT 认证 + bcrypt）
- **训练参数配置**：47 项 canonical 参数表单 + **B1–B4 正式方案预设模板**
- 后端严格校验（类型/范围/choices/组合/未知参数 422 拒绝），参数经 argv 数组直传训练进程（无 shell 拼接）
- 一键启动/停止训练；任务生命周期（pending/running/finished/failed/stopped）完整入库
- **实时可视化**（WebSocket 推送 + 结构化事件）：
  - 任务状态卡（当前轮次/总轮次、当前阶段、已运行时间、最佳轮次、git commit、退出码）
  - global_val / global_test / best_val / best_test 曲线
  - 客户端本地训练损失（weighted CE / 子图 InfoNCE / total）曲线
  - 生成器与蒸馏损失曲线（semantic / diversity / distillation / fake_x 统计）
  - **CKR 权重表**（客户端 × 类别，最近一轮）
- **参数热更新**：白名单参数（学习率、损失权重、生成/蒸馏步数等 15 项）可在当前轮结束后、下一轮开始前真实生效，审计记录（参数/旧值/新值/生效轮次/状态）持久化
- 训练产物按任务落盘：`runs/platform/exp_<id>/`（checkpoints / logs / metrics.jsonl / events.jsonl / final_metrics.json）
- 实验历史管理：完整配置快照（原始参数 + canonical 配置 + git commit + 运行环境）、复制为新任务、从 checkpoint 恢复、导出配置

### 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | Vue 3 + Vite + Element Plus + ECharts + Pinia |
| 后端 | FastAPI + SQLAlchemy (async) + asyncpg |
| 数据库 | PostgreSQL |
| 认证 | JWT + bcrypt |
| 实时通信 | WebSocket |
| 训练执行 | asyncio subprocess（argv 数组直传） |

### 快速启动

```bash
# 1. 创建数据库 (本地开发)
cd research-training/backend
# 按 backend/README.md 创建用户与数据库 (本地开发密码仅限本地)

# 2. 启动后端
cp .env.example .env   # 编辑数据库连接
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. 启动前端（另一个终端）
cd research-training/frontend
npm install
npm run dev
```

打开 http://localhost:5173，使用 `admin / admin123` 登录。

### 创建第一个训练任务

1. 登录后进入「训练配置」页；
2. 选择 **B4 · 完整方法** 预设（或 B1–B3 基线预设）；
3. 按需调整参数（高级参数可在「核心方法」等分组中修改）；
4. 点击「开始训练」，自动跳转监控页；
5. 监控页实时显示：状态卡（轮次/阶段/耗时）、全局指标曲线、客户端损失曲线、生成器/蒸馏曲线、CKR 权重表与实时日志；
6. 任务运行中可在监控页「参数热更新」面板修改白名单参数（下一轮开始生效），审计记录可查；
7. 任务完成后查看最佳指标，实验历史页可复制为新任务或从 checkpoint 恢复。

详细操作见 [docs/PLATFORM_GUIDE.md](docs/PLATFORM_GUIDE.md)。

### 哪些参数支持热更新（下一轮开始生效）

仅以下 15 项白名单参数支持运行中修改（在**当前轮结束后、下一轮客户端训练开始前**由训练进程应用）：

`learning_rate`（客户端学习率）、`generator_lr`、`distill_lr`、`lambda_subgraph`、`lambda_sem`、`lambda_diversity`、`lambda_disagreement`、`lambda_feature_norm`、`contrastive_temperature`、`generator_steps`、`distillation_steps`、`edge_perturb_ratio`、`knn_k`

热更新在优化器 step 边界生效（不打断当前 step），每次更新留有审计记录（参数/旧值/新值/生效轮次/时间戳/状态）。

### 哪些参数需要重新启动任务

以下参数改变模型结构、数据划分或任务语义，**运行中修改会被拒绝并提示"需要重新启动训练任务"**：

- 数据集与任务：`dataset`、`task_mode`、`normal_classes`、`anomaly_classes`、`num_clients`、`partition`、`seed` 系列
- 模型结构：`hid_dim`（输入/隐藏维度）、`diffusion_hidden`、`diffusion_steps`、`diffusion_beta_*`
- 数据划分：`rwr_subgraph_size`（RWR 缓存键）、`reliability_holdout_ratio`、`reliability_split_seed`
- 方法开关：`contrastive_mode`、`ckr_mode`、`distill_weighting`、`use_weighted_ce`、`fake_class_strategy`、`generator_init`
- 其余未列入热更新白名单的参数（完整清单见 [docs/TRAINING_PARAMETERS.md](docs/TRAINING_PARAMETERS.md)）

---

## 12. Cora 合成异常任务说明与隐私边界

### Cora 合成异常

- Cora 原始任务是**论文主题多分类**（7 类）；
- `anomaly_binary` 模式通过标签映射构造**合成异常任务**：正常类 `0,1,2,3` 映射为 0，异常类 `4,5,6` 映射为 1（可通过 `--normal_classes` / `--anomaly_classes` 调整）；
- 该映射在模型与损失初始化前完成；`multiclass` 仅为兼容模式；
- 不得将 Cora 原始主题类别直接描述为真实欺诈标签。

### 算法表述口径

- **Louvain 仅基于图拓扑划分社区**，不读取标签、不判定异常；
- 子图跨视图对比中的正负样本指：同一锚点跨视图子图对（正）/ 不同锚点子图对（负），**不代表正常/异常节点**；
- 默认 `scratch` 生成器严格命名为 **teacher-guided diffusion-style generator**：从高斯噪声与类别条件出发、在冻结教师模型反馈下训练，不是经过真实节点特征训练的标准 DDPM；
- 只有 `proxy_pretrained`（实验性）中的代理预训练阶段使用标准 DDPM 噪声预测损失；
- 伪图是**知识蒸馏代理数据**，不是真实大图或真实跨客户端边的恢复；
- 服务端不访问客户端私有子图数据（生成器/蒸馏损失只接收伪图与模型参数）；但这是工程约定，**不等于形式化差分隐私保证**。

### 平台集成说明

- 训练程序通过 `--emit_events` 输出 `[EVENT] {json}` 结构化事件（轮次/客户端指标/CKR/生成器/蒸馏/资源），后端优先解析结构化事件，自然语言正则仅作 fallback；
- 每个任务的产物目录 `runs/platform/exp_<id>/`（gitignored）：checkpoints、logs、metrics.jsonl、events.jsonl、final_metrics.json；
- 数据库存储完整配置快照（原始参数 + canonical 配置 + git commit + 运行环境），保证「数据库中的参数」与「实际执行参数」一致。

