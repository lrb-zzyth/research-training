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

#### MLP 生成器（FedTAD_ConGenerator）— 默认

| 项目 | 说明 |
|---|---|
| 输入 | 随机噪声 `z`（`[num_gen, 32]`）+ 伪标签 `c`（`[num_gen]`）|
| 结构 | 标签嵌入 → 拼接 → 3 层 Tanh MLP → Linear 输出 |
| 输出 | `[num_gen, feat_dim]` 合成节点特征 |

`feat_dim` 取决于蒸馏模式：
- `raw_distill`：`feat_dim = 原始特征维度`（如 Cora 1433 维）
- `rep_distill`：`feat_dim = hid_dim`（默认 64），使用 GCN 第二层做预测

#### 扩散生成器（ConditionalDiffusionGenerator）— 可选

基于 DDPM 的条件扩散模型，用于服务端生成合成节点特征。

| 项目 | 说明 |
|---|---|
| 输入 | 含噪特征 `x_t` + 时间步 `t` + 类别标签 `labels` |
| 结构 | 时间嵌入 + 类别嵌入 → 3 层 SiLU MLP 噪声预测网络 |
| 采样 | 完整 DDPM 反向扩散链（重参数化，支持梯度回传） |
| 输出 | `[num_gen, feat_dim]` 合成节点特征 |

通过 `--generator_type diffusion` 启用。

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

在每个客户端的子图上运行 `local_model`，计算 train/val/test 的 loss 和 accuracy，加权汇总为全局指标。

### 阶段 3：客户端本地训练

每个客户端在自己的子图上训练 GCN：

```
loss = CrossEntropyLoss(logits[train_idx], labels[train_idx])
```

**可选改进**（通过 `--use_weighted_ce` 和 `--use_contrastive` 启用）：

```
loss = weighted_ce_loss + lambda_cl × contrastive_loss
```

| 改进 | 说明 |
|---|---|
| **Weighted CE** | 基于本地 `train_mask` 标签分布计算类别权重。支持 `effective_num`（默认，`weight_c = (1-β)/(1-βⁿᶜ)`）和 `inverse`（`weight_c = N/(num_classes × n_c)`）两种方法。缺失类别权重置 0，非零权重归一化到均值约 1 |
| **Supervised Contrastive Loss**（`--contrastive_type supcon`）| 在 GCN 倒数第二层 embedding 上计算。同类节点为正样本，异类节点为负样本。无有效正样本时返回 0（避免 NaN） |
| **GRADATE Multi-Scale Contrastive Loss**（`--contrastive_type gradate`）| 自监督多尺度对比损失，包含 node-subgraph、node-node、subgraph-subgraph cross-view InfoNCE 三种损失。通过 edge perturbation 构造增强 view，RWR 子图采样。参考 GRADATE 论文结构 |

### 阶段 4：服务端蒸馏

这是 FedTAD 的核心创新，分两个子阶段：

#### 子阶段 4a：训练生成器（`it_g` loop）

1. **生成伪节点特征**：
   - MLP 模式：`z = randn([num_gen, 32])` → `generator(z, c)` → `node_logits`
   - Diffusion 模式：`generator.sample(c)` → `node_logits`
2. **构造伪图**：L2 归一化 → 内积邻接矩阵 → top-k 选边 → 对称化 + 自环
3. **模型推理**：local_model 和 global_model 在伪图上推理得到 local_pred / global_pred
4. **三种损失**：

   | 损失 | 公式 | 作用 |
   |---|---|---|
   | **语义损失** | `Σ_c n_ckr[c] × CE(local_pred[c], c[c])` | 使生成特征可分 |
   | **分歧损失** | `-Σ_c n_ckr[c] × mean(\|global_pred[c] − local_pred[c]\|)` | 对齐全局与局部模型 |
   | **多样性损失** | `exp(−mean(noise_dist × feature_dist))` | 鼓励生成特征多样化 |

5. **生成器更新**：`loss_G = lam1 × loss_sem + loss_diverg + lam2 × loss_div`

#### 子阶段 4b：训练全局模型（`it_d` loop）

固定生成器（eval 模式），在伪图上通过分歧损失更新全局模型（local_pred detach，不更新客户端模型）。

### 阶段 5：全局聚合

加权平均各客户端模型参数，加权系数为 `subgraph_size / total_size`。

---

## 7. 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--seed` | 2024 | 随机种子 |
| `--root` | `./dataset` | 数据根目录 |
| `--dataset` | Cora | 数据集名称 |
| `--gpu_id` | 0 | GPU 编号 |
| `--num_clients` | 10 | 客户端数量 |
| `--num_rounds` | 100 | 通信轮数 |
| `--num_epochs` | 3 | 客户端本地训练 epoch |
| `--partition` | Louvain | 图分区策略（Louvain / Metis） |

### 模型参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hid_dim` | 64 | GCN 隐藏层维度 |
| `--dropout` | 0.5 | Dropout 比率 |
| `--lr` | 1e-2 | 客户端学习率 |
| `--weight_decay` | 5e-4 | 权重衰减 |

### 服务端蒸馏参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--generator_type` | mlp | 生成器类型（mlp / diffusion）|
| `--num_gen` | 100 | 每轮生成的伪节点数 |
| `--glb_epochs` | 5 | 服务端蒸馏轮数 |
| `--it_g` | 1 | 生成器更新步数 |
| `--it_d` | 5 | 全局模型更新步数 |
| `--lr_g` | 1e-3 | 生成器学习率 |
| `--lr_d` | 1e-3 | 全局模型学习率 |
| `--fedtad_mode` | raw_distill | 蒸馏模式（raw_distill / rep_distill）|
| `--lam1` | 1 | 语义损失权重 |
| `--lam2` | 1 | 多样性损失权重 |
| `--topk` | 5 | kNN 构图 k 值 |

### 扩散生成器参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--diffusion_steps` | 50 | 扩散步数 |
| `--diffusion_hidden` | 256 | 隐藏层维度 |
| `--diffusion_beta_start` | 1e-4 | beta 起始值 |
| `--diffusion_beta_end` | 0.02 | beta 终止值 |

### 客户端改进参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_weighted_ce` | False | 启用 Weighted CE |
| `--class_weight_method` | effective_num | 加权方法（effective_num / inverse）|
| `--beta` | 0.999 | effective number 参数 |
| `--use_contrastive` | False | 启用对比损失（总开关）|
| `--contrastive_type` | supcon | 对比损失类型：`supcon`（监督式）或 `gradate`（GRADATE 多尺度自监督）|
| `--lambda_cl` | 0.1 | 对比损失权重 |
| `--cl_tau` | 0.5 | SupCon 温度参数 |

### GRADATE 对比学习参数（`--contrastive_type gradate` 时生效）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gradate_tau` | 0.5 | 子图-子图 InfoNCE 温度 |
| `--gradate_edge_drop_rate` | 0.2 | 增强 view 边增删比例 |
| `--gradate_beta` | 0.1 | Node-Subgraph vs Node-Node 权重平衡 |
| `--gradate_gamma` | 0.1 | Subgraph-Subgraph NCE 权重 |
| `--gradate_alpha` | 0.1 | 原始 view vs 增强 view 加权 |
| `--gradate_subgraph_size` | 4 | RWR 子图 context 节点数（不含 anchor）|
| `--gradate_negsamp_ratio_patch` | 6 | Node-Node 负采样比 |
| `--gradate_negsamp_ratio_context` | 1 | Node-Subgraph 负采样比 |

---

## 8. 训练示例

### 8.1 原始 FedTAD（MLP 生成器）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain
```

### 8.2 扩散生成器

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --generator_type diffusion --diffusion_steps 10
```

### 8.3 SupCon 监督式对比 + 扩散生成器

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --generator_type diffusion --diffusion_steps 10 \
  --use_weighted_ce --use_contrastive --lambda_cl 0.1
```

### 8.4 GRADATE 多尺度自监督对比

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --use_contrastive --contrastive_type gradate
```

### 8.5 GRADATE + 扩散生成器 + Weighted CE（全功能）

```bash
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --generator_type diffusion --diffusion_steps 10 \
  --use_weighted_ce \
  --use_contrastive --contrastive_type gradate
```

### 8.6 FedAvg 基线

```bash
python train_fedavg.py --dataset Cora --num_clients 10 --partition Louvain
```

---

## 9. 项目文件结构

```
FedTAD/
├── train_fedtad.py              # FedTAD 主训练脚本
├── train_fedavg.py              # FedAvg 基线
├── model.py                     # GCN + 两种生成器（MLP + Diffusion）
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── util/
│   ├── base_data_util.py        # 图切分（Louvain/Metis）+ 子图构建
│   ├── base_util.py             # 种子设置 + 数据加载
│   ├── fgl_dataset.py           # FGLDataset 类
│   ├── task_util.py             # accuracy / DiversityLoss / construct_graph
│   └── gradate_contrastive.py   # GRADATE 多尺度对比学习模块
├── louvain/                     # Louvain 社区发现算法实现
└── ckr/                         # CKR 矩阵缓存
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

项目还提供了一个基于 Web 的可视化训练管理平台，位于 `research-training/` 目录下。

### 功能

- 用户登录（JWT 认证）
- 训练参数配置面板
- 一键启动/中断训练
- 实时查看训练日志（WebSocket 推送）
- 实时 global_val / global_test / best_val / best_test 曲线（ECharts）
- 实验历史记录管理（PostgreSQL 持久化）

### 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | Vue 3 + Vite + Element Plus + ECharts + Pinia |
| 后端 | FastAPI + SQLAlchemy (async) + asyncpg |
| 数据库 | PostgreSQL |
| 认证 | JWT + bcrypt |
| 实时通信 | WebSocket |
| 训练执行 | asyncio subprocess |

### 快速启动

```bash
# 1. 创建数据库
createdb fedtad

# 2. 启动后端
cd research-training/backend
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. 启动前端（另一个终端）
cd research-training/frontend
npm install
npm run dev
```

打开 http://localhost:5173，使用 `admin / admin123` 登录。

详细文档请参见 `research-training/README.md`。

