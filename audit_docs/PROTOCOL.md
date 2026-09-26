# PROTOCOL — Cora-10 多分类精度对照实验

> 本文件记录所有自跑实验的完整口径，保证每个数字可复现、可追溯。
> 生成时间：2026-09-21 · 仓库 commit `6ef192c`（branch `main`，工作区另有未提交改动，见 §6）

---

## 1. 任务口径

| 项 | 值 |
|---|---|
| 任务模式 | `--task_mode multiclass`（**必须显式指定**，仓库默认是 `anomaly_binary`） |
| 主指标 | test accuracy (%) ，取验证集选轮对应的那一轮 |
| 数据集 | Cora（2,708 节点 / 1,433 维 / 7 类） |
| 客户端划分 | Louvain，10 clients |
| 对应论文 | FedTAD, IJCAI 2024, arXiv:2404.14061, Table 2 |

## 2. 训练协议（与论文正文一致）

论文原文（§5.2 Hyperparameters）：
> "a two-layer GCN as our backbone. The dimension of the hidden layer is set to 64 or 128.
> The local training epoch and round are set to 3 and 100, respectively. The learning rate
> of GNN is set to 1e-2, the weight decay is set to 5e-4, and the dropout is set to 0.5."
> "For each experiment, we report the mean and variance results of **3** standardized training."

| 参数 | 取值 | 命令行 |
|---|---|---|
| 联邦轮次 | 100 | `--num_rounds 100` |
| 本地 epoch | 3 | `--num_epochs 3` |
| 隐藏维 | 64 | `--hid_dim 64` |
| 学习率 | 1e-2 | `--lr 1e-2` |
| 权重衰减 | 5e-4 | `--weight_decay 5e-4` |
| dropout | 0.5 | `--dropout 0.5` |
| 骨干 | 2 层 GCN | 固定 |

## 3. ⚠️ 关键口径：`--no-resplit_stratified`（本实验的核心决定）

仓库 `train_fedtad.py:1685-1723` 默认会**丢弃**缓存 `data*.pt` 里烤好的 train/val/test 划分，
用 `stratified_split(seed=split_seed+ci)` **重新切分**。

后果（实测，Cora-10 FedAvg，seed 0）：

| 配置 | test acc |
|---|---|
| 原版 FedTAD 代码（未打补丁） | 72.90 |
| 原版 FedTAD 代码（修 dropout bug 后，见 §5） | 73.85 |
| 本仓库 + `--no-resplit_stratified` | **74.21** |
| 本仓库（默认开启 resplit） | 77.21（5 种子均值） |

**为什么必须关掉**：resplit 后训练/测试的**节点集合**与原版不再相同（类别数量相同、具体节点不同），
且切分**随种子变化**（`seed=split_seed+ci`），导致跨种子方差从 ±0.4 级放大到 ±1.5 级。
**开启 resplit 时，本仓库数字不可与论文 Table 2 直接对照。**

**本实验全部实验组统一使用 `--no-resplit_stratified`**，此时 FedAvg=74.21 与论文 73.6±0.4 吻合。

划分 artifact 路径：
- 客户端划分：`dataset/Cora/Client10/Louvain/data0.pt … data9.pt`
  （与 `references/FedTAD/dataset/Cora/Client10/Louvain/` **逐位一致**，即论文作者随仓库发布的同一套划分）
- 各 run 的切分记录：`runs/accuracy_benchmark/<组名>/seed_<n>/split_report.json`

## 4. 种子与选轮

- **种子**：`--seed {0,1,2,3,4}`，共 5 个种子（论文用 3 个；任务书要求 ≥5）。
  该参数同时设定 partition / allocation / split / model / bootstrap 五个子种子，
  与原版 FedTAD 的单一 `--seed` 行为一致（原版 `train_fedtad.py:74-75` 先 `seed_everything` 再 `load_dataset`）。
- **选轮口径**：**验证集**选轮（`--selection_metric accuracy`），报告该轮的 test accuracy。
  **严禁用测试集选轮。** 原版代码同样使用验证集选轮（`references/FedTAD/train_fedtad.py:231`）。
- 每次运行的完整配置哈希记录在 `final_metrics.json` 的 `config_hash` 字段。

## 5. 与原版代码的差异（诚实记录）

| # | 项 | 原版 | 本仓库 | 影响 |
|---|---|---|---|---|
| 1 | 数据切分 | 缓存 `data*.pt` 内的固定切分 | 默认重切分，需 `--no-resplit_stratified` 才对齐 | §3，已通过开关消除 |
| 2 | dropout | `F.dropout(x, p=self.dropout)` **缺 `training=self.training`** → 评估时也在 dropout | 已修正 | 原版被低估约 0.95 分 |
| 3 | 服务器生成器 | `FedTAD_ConGenerator`（**纯 MLP**：噪声+类别嵌入 → 3×(Linear+Tanh+Dropout)） | `ConditionalDiffusionGenerator`（**DDPM**，T=10） | 本方法的改动点 |
| 4 | 散度损失 | **L1**：`torch.abs(global_pred − local_pred.detach())` | **KL(global‖local)**：使用显式 `Σ p_global(log p_global−log p_local)`，避免 PyTorch `kl_div` 参数顺序歧义 | 本方法的改动点 |
| 5 | 对比学习 | 无 | 子图-子图跨视图 InfoNCE（专利 S2.5 形式） | 本方法的改动点 |
| 6 | 加权 CE | 无 | 有（默认开，本实验**关闭** `--no-use_weighted_ce`） | 实测为负贡献，见 REPORT |

`references/FedTAD/` **被修改过 185 行**（新增 `--anomaly_binary`/`--task_mode`/AUC 上报 + PyTorch2.6 兼容），
但**核心算法（生成器 loss、L1 散度、CKR 加权）未被改动**——已逐行核对 diff 确认。

## 6. 代码与产物路径

- 本实验代码：`experiments/accuracy_benchmark/`（`summarize.py` 汇总、`make_curves.py` 曲线）
- 运行脚本：`runs/accuracy_benchmark/run_main_noresplit.sh`、`run_tuning.sh`
- 结果目录：`runs/accuracy_benchmark/<组名>/seed_<n>/`
  - `final_metrics.json` — 最终指标（本报告所有数字的唯一来源）
  - `metrics.jsonl` — 每轮记录（ce/cl/generator/distill loss、val_primary）
  - `events.jsonl` — 每轮 `global_test` 等事件流（曲线的数据源）
  - `split_report.json` — 该 run 的实际划分统计
  - `time.txt` — 墙钟与峰值内存

工作区未提交改动（实验基于此运行）：
- `train_fedtad.py` — 教师前向去重（每个教师只算一次，跨类别复用）
- `util/task_util.py` — `subgraph_contrastive_loss` 改为专利 S2.5 的 InfoNCE 形式（分母含正样本）

## 7. 硬件与并行约束

本机 **7 GB 内存 / 8 GB 显卡**。单个 run 峰值 RSS ≈ **2.2 GB**。
**所有实验严格串行执行**（一次一个训练进程），运行脚本内置双保险：
检测到并发进程或可用内存 < 3 GB 时立即中止。
（历史教训：6 进程并行导致 OOM，整个 WSL 被系统杀死。）
