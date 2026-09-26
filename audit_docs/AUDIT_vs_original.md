# 审计：本仓库实现 vs 原版 FedTAD —— 全量管线差异清单

> 2026-09-24 重做（此前版本只对照了参数与损失，**漏掉了数据侧的训练集差异**，已补齐）
> 对照对象：`references/FedTAD/`（上游 `xkLi-Allen/FedTAD` 克隆，核心算法未被改动）
> 目的：逐环节确认"任何可能影响最终数字"的差异，不静默修改算法。

---

## A. 全量差异清单（17 个环节）

| # | 环节 | FedTAD 原版 | 本仓库 | 判定 |
|---|---|---|---|---|
| 1 | 数据加载 | PyG `Planetoid`/`Coauthor` 原生 | 相同 | ✅ |
| 2 | 特征归一化 | **无** | **无**（grep 零命中） | ✅ |
| 3 | 图构建 | `to_networkx(G, to_undirected=True, remove_self_loops=True)` | **逐行相同** | ✅ |
| 4 | Louvain 社区发现 | `best_partition(graph)` | 同 + `random_state=2024` | ✅ **已验证等价**（见 C-1） |
| 5 | 过大社区切块 | `group_len_max = N//K - delta` | **逐行相同** | ✅ |
| 6 | chunk→client 均衡分配 | 按大小降序 + round-robin + 容量约束 | **逐行相同** | ✅ |
| 7 | 诱导子图 + 删跨客户端边 | `graph_nx.subgraph(node_set)` | **逐行相同** | ✅ |
| 8 | global→local 索引映射 | `node_idx_map` | **逐行相同** | ✅ |
| 9 | train/val/test 切分 | 类内分层 + `int()` 截断 | **逐行相同** | ✅ |
| **10** | **训练用哪些节点** | **`train_idx` 全部** | 原为 `fit_idx`(80%) → **已修** | 🔴 **2026-09-24 修复** |
| 11 | 模型 | 2 层 GCN | 相同（另修了原版 dropout bug） | ✅ |
| 12 | 优化器 | `Adam(lr=1e-2, wd=5e-4)` | 相同 | ✅ |
| 13 | 本地训练 | 3 epochs、全批 | 相同 | ✅ |
| 14 | FedAvg 聚合权重 | `n_k / N_global` | `n_k / Σn_k` | ✅ **等价**（Σn_k = N_global） |
| 15 | **服务端迭代次数** | 生成器 5（=glb_epochs5×it_g1）<br>蒸馏 25（=glb_epochs5×it_d5） | 搜索空间原为 `generator_steps{1,3,5}` / `distill_steps{1,3,5,10}` | 🔴 **2026-09-24 修复**（distill 加入 25） |
| 16 | **评估聚合口径** | `Σ_k (n_k/N_global) × acc_k`（按客户端**总节点数**加权） | 原只有 pooled / client_macro / client_weighted(按测试集大小) | 🔴 **2026-09-24 修复**（新增 `fedtad_official`） |
| 17 | 选轮口径 | 验证集选 best | 相同 | ✅ |

**设计差异（非错误，是本方法的创新点）**：

| 环节 | FedTAD | 本方法 | 依据 |
|---|---|---|---|
| 生成器 | MLP (`FedTAD_ConGenerator`) | 条件 DDPM | 专利权5 / 专利书 S3 |
| 散度/蒸馏损失 | L1 on raw logits（代码） | **KL on 预测分布** | 论文 Eq.10 + 专利权6 / S4.2 |
| 对比学习 | 无 | 子图跨视图 InfoNCE | 专利权3 / S2 |
| 客户端加权 CE | 无 | 有（实测负贡献，已关闭） | 专利权4 |

---

## B. 2026-09-24 修复的三项（影响所有历史结果）

### B-1 🔴 训练集：`fit_idx`(80%) → `train_idx`(100%)

**原版**（`references/FedTAD/train_fedtad.py:212`）：客户端在 `train_idx` **全部**节点上训练。
**修复前本仓库**：留 20% 作 `reliability_idx`，训练只用剩下 80%（Cora client0: 43 vs 原版 53）。

**修复方式（不改代码，只加参数）**：
```
--reliability_holdout_ratio 0
```
**为什么安全**：`reliability_idx` 只在 `elif use_dynamic_ckr:` 分支被使用
（`train_fedtad.py:2479`），`--ckr_mode static_topology` 下该保留集根本不被读取。
单元测试验证 `holdout_ratio=0` 时 `fit_idx == train_idx` 且 `reliability == ∅`。

### B-2 🔴 服务端迭代次数：`distill_steps` 搜索空间加 25

原版每轮蒸馏 **25** 次（`glb_epochs(5) × it_d(5)`）。原搜索空间上限只有 10，**永远够不到原版强度**。
已改为 `{1, 3, 5, 10, 25}`。

### B-3 🔴 评估聚合口径：新增 `fedtad_official`

原版按**客户端总节点数**加权，我们只按测试集大小加权。已新增第四个聚合层级，精确复刻原版公式。
**实测差异很小**（5 个 run 上 −0.006 ~ +0.107，最大 0.107，多数 <0.03），但口径现在是对齐的。

---

## C. 已验证为等价、无需修改的差异

### C-1 Louvain 调用：`random_state=2024` vs 全局 RNG

```python
# 原版
partition = community_louvain.best_partition(graph)
# 本仓库
partition = community_louvain.best_partition(graph, random_state=louvain_seed)  # 默认 2024
```

**等价性论证**：
- `seed_everything(2024)` 在 `train_fedtad.py:1775`，`load_dataset` 在 `:1837`
- **这 62 行区间内没有任何 RNG 调用**（已 grep 确认）
- 故全局 RNG 状态 ≡ `RandomState(2024)` 初始状态

**实测验证**：用 seed 2024 重建的子图与**作者随仓库发布的 `dataset/Cora/Client10/Louvain/data0-9.pt`
逐位一致**（`x`/`y`/`edge_index`/`train_idx`/`val_idx`/`test_idx` 全部相同，10/10）。

**附带确认**：`louvain/community/community_louvain.py` 与原版**内容完全相同**（仅 CRLF/LF 换行符差异）。

### C-2 聚合权重的分母

原版用 `dataset.global_data.x.shape[0]`（全图节点数），我们用 `Σ_k n_k`。
因客户端划分覆盖全部节点（`assert sum(...) == G.num_nodes`），两者相等 ✅

### C-3 原版的 dropout bug

原版 `model.py` GCN 中 `F.dropout(x, p=self.dropout)` **漏写 `training=self.training`**
（`F.dropout` 该参数默认 `True`），导致**评估阶段也在 dropout**，`eval()` 被架空。
本仓库已修正。实测原版 FedAvg 72.90 → 修补后 73.85。
**这是原版的 bug，不应跟。**

### C-4 原版的死参数

原版 `--num_dims`（默认 64）在代码中**从未被使用**，无影响。

---

## D. 仍然存在的、无法消除的差异（如实记录）

### D-1 非 Cora 数据集的客户端划分非作者发布版

作者仓库**只发布了 Cora 的子图**。CiteSeer / PubMed / CS / Physics 的 5/10/20 三档划分
**全部是本仓库用同一套 Louvain 逻辑自行生成的**（同 seed 2024、同 delta=20、同分配算法）。

**后果**：这四个数据集的**跨客户端档位趋势**与论文不可直接比较
（实测 CiteSeer 出现"5 档 < 10 档"，与论文的递减趋势相反；Cora 则正常递减）。

### D-2 生成器与损失的设计差异

见 A 节"设计差异"表。这些是**本方法相对 FedTAD 的改进点**，不是实现错误。

### D-3 专利与代码的一处落差

专利 S3.1/S3.2 描述的是**标准 DDPM**（对真实 x0 加噪、预测噪声）；
而联邦无数据设定下服务端拿不到 x0，代码实现的是"教师引导扩散式生成器"。
**根因是物理约束，需要修订专利表述或补一个可执行的近似前向过程。**
（注: 2026-09-24 已加联邦扩散预训练 `federated_diffusion_pretrain` 缓解此落差, 见 HANDOFF §4 问题 B。）

### D-4 专利与代码的语义差异: 边扰动比例（2026-09-25 记录, 不改专利）

专利文字写的是「以概率 p_drop 删除、以相同概率 p_add 添加」, 读起来像各 20%。
代码语义（`--edge_perturb_ratio`, 2026-09-25 已修无向性 bug 并写清 docstring/help）是:
**总扰动 = 删 ratio/2 + 补 ratio/2**（ratio=0.2 → 各 10% 的无向边）。
另: 修复前旧实现按有向条目删/加, 实际扰动强度约为文档语义的两倍, 且第二视图被削成部分有向图;
2026-09-25 修复为以无向边为单位删/加后, 代码与文档语义才真正一致。

---

## E. 复现结论

**Cora 上，本仓库的划分管线与作者发布版逐位一致。** 其余四个数据集使用同一套算法与同一 seed 生成，
但作者的原始划分未公开，故无法逐位核对。
