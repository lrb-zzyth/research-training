# RETAINED_ARCHITECTURE.md — 最终保留架构

> 生成时间：2026-08-02
> 本项目最终同时保留两条主线：**核心专利算法**与**现有前后端平台**。本文说明保留的架构、模块归属与调用关系。

---

## 一、两条主线总览

```
┌─────────────────────────────────────────────────────────────────┐
│  主线 A：核心专利算法                                             │
│                                                                 │
│  train_fedtad.py（主入口）                                       │
│    ├── model.py                 GCN + ConditionalDiffusionGenerator │
│    ├── util/task_util.py        边扰动 / RWR 采样 / InfoNCE / 加权 CE │
│    ├── util/base_data_util.py   Louvain/Metis 拓扑划分（不读标签）    │
│    ├── util/fgl_dataset.py      数据集封装                          │
│    ├── util/base_util.py        加载与种子                          │
│    ├── util/data_split.py       分层重划分 / reliability holdout     │
│    ├── util/dynamic_ckr.py      CKR 跟踪器（静态核心 / 动态扩展）      │
│    ├── util/checkpoint.py       断点保存恢复                         │
│    ├── util/rwr_cache.py        RWR 子图采样缓存                     │
│    ├── util/split_artifact.py   冻结划分 artifact                   │
│    └── pretrain_diffusion.py    公开代理预训练（默认关闭，实验性）      │
│                                                                 │
│  louvain/                    vendored python-louvain（社区发现）   │
│  dataset/Cora/               raw + processed + Louvain 客户端子图   │
│  ckr/                        静态 CKR 磁盘缓存                     │
│  test_all.py / test_smoke.py / test_smoke_train.py  核心测试       │
├─────────────────────────────────────────────────────────────────┤
│  主线 B：前后端平台（research-training/）                          │
│                                                                 │
│  backend/  FastAPI + PostgreSQL + WebSocket                      │
│    ├── app/main.py              入口 + /api/health + CORS        │
│    ├── app/routers/auth.py      /api/auth/*（登录/注册/当前用户）    │
│    ├── app/routers/training.py  /api/training/*（启停/状态/WS）     │
│    ├── app/routers/experiments.py /api/experiments/*（列表/详情/日志/指标）│
│    ├── app/training.py          子进程管理 / 日志流 / 指标入库 / 状态机 │
│    ├── app/log_parser.py        stdout → 指标解析（对齐当前输出）     │
│    ├── app/websocket_manager.py 实时日志/指标/状态广播               │
│    ├── app/auth.py / config.py / database.py / models.py / schemas.py │
│    └── test_backend_smoke.py    后端平台 smoke test（38 项）         │
│                                                                 │
│  frontend/  Vue 3 + Vite + Element Plus + ECharts + Pinia       │
│    ├── src/views/         Dashboard/Login/Register/TrainingConfig/│
│    │                      TrainingMonitor/ExperimentHistory/Settings │
│    ├── src/components/    ExperimentTable / LogConsole / MetricChart │
│    ├── src/api/index.js   10 个 API 客户端函数                     │
│    ├── src/stores/        auth / settings / training              │
│    ├── src/router/index.js 路由（登录守卫）                        │
│    └── package.json / vite.config.js / index.html                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心专利算法调用路径（主线 A）

默认配置即核心完整方法 B4。完整调用路径：

```
main() (train_fedtad.py)
 ├─ [S0] 种子/横幅 + Core method 声明日志
 ├─ [S1] 数据加载 load_dataset() → util.base_util → util.fgl_dataset
 │        → util.base_data_util.data_partition → Louvain 拓扑划分（仅图拓扑，不读标签）
 ├─ [S1] 标签映射 apply_label_mapping（anomaly_binary：正常=0 异常=1，在模型/损失初始化前）
 ├─ [S1] 静态拓扑 CKR compute_ckr()（固定节点属性 + 本地图拓扑，带磁盘缓存）
 ├─ [S2] 分层重划分 stratified_split*（split_seed 控制，fit/reliability/val/test 无重叠）
 ├─ 初始化：本地 GCN × N + 全局 GCN + ConditionalDiffusionGenerator（scratch）
 │
 ├─ 每轮 for round_id in range(num_rounds):
 │   ├─ 广播全局模型 → 客户端本地训练（仅 fit_idx 参与梯度）：
 │   │    loss = 类别加权交叉熵（仅用客户端 train 标签计算权重，缺失类置 0）
 │   │         + lambda_subgraph × 子图-子图跨视图 InfoNCE
 │   │         （边扰动 → 两视图 → RWR 局部子图 → 中心特征置零 → 共享 GCN → mean readout
 │   │           → 同中心跨视图为正对 / 不同中心为负对；不读取标签）
 │   ├─ 客户端上传：state_dict 聚合（单进程模拟，服务端生成/蒸馏损失函数不接触客户端 Data）
 │   ├─ [Server] FedAvg：按客户端样本量加权平均（no_grad，不反向传播）
 │   ├─ [Server] 教师引导扩散式生成器训练（generator_steps）：
 │   │    高斯噪声 + 类别条件 → 冻结教师/全局模型反馈（语义/分歧/多样性损失）
 │   │    → 只更新生成器；输入梯度仍回传 fake_x 与生成器
 │   ├─ [Server] KNN 伪图构造：build_knn_graph(fake_x, k=--knn_k)（不读客户端真实 edge_index）
 │   └─ [Server] CKR 加权全局蒸馏（distill_steps）：
 │        生成器 no_grad 采样 → 教师预测 detach → 静态 CKR 加权教师分布
 │        → 全局模型 KL 蒸馏（只更新全局模型）
 ├─ 评估与模型选择（val/test 只做报告与选择，不进训练权重）
 └─ 最终 test：加载 best.pt → pooled / client_macro / client_weighted 指标
```

### B1–B4 方案开关（唯一正式实验集合）

| 方案 | 配置 | 说明 |
|---|---|---|
| B1 | `--distill_weighting none --contrastive_mode none --no-use_weighted_ce` | 纯 FedAvg（普通交叉熵） |
| B2 | `--distill_weighting none --contrastive_mode none` | FedAvg + Weighted CE |
| B3 | `--distill_weighting none` | FedAvg + Weighted CE + 子图对比 |
| B4 | 默认配置（`--distill_weighting static_ckr --ckr_mode static_topology`） | 完整方法 |

无 B5（动态 CKR 不进入正式核心实验；`--distill_weighting dynamic_ckr` 标注为实验性扩展，默认关闭）。

---

## 三、前后端平台架构（主线 B）

### 调用关系

```
浏览器 (Vue 3)
  │  REST /api/*（axios, Bearer JWT）+ WebSocket /api/training/ws/{id}
  ▼
FastAPI 后端
  ├─ 鉴权：JWT（python-jose + passlib/bcrypt）
  ├─ 训练任务：routers/training.py 把 parameters（canonical 键）映射为
  │   train_fedtad.py CLI flags → TrainingRunner 子进程（cwd=仓库根）
  │   → stdout 逐行：入库 training_logs + WS 广播 + log_parser 解析指标
  │   → 指标入库 training_metrics + 更新 experiments.best_round/best_val/best_test
  │   → 进程退出 → status=finished/failed/stopped
  └─ PostgreSQL：experiments / users / training_logs / training_metrics
```

### 平台 → 算法的唯一耦合点（已修复）

- `POST /api/training/start` 的 `parameters` dict → CLI flag 映射（routers/training.py）
- `log_parser.py` → 与 train_fedtad.py 当前输出格式对齐
- 平台不直接导入算法内部对象；算法不包含任何 HTTP/页面逻辑

### 默认核心配置（平台表单 = 算法默认 = CLI 默认）

```
task_mode=anomaly_binary  normal_classes=0,1,2,3  anomaly_classes=4,5,6
contrastive_mode=subgraph_cross_view  contrastive_anchor_scope=all_nodes
ckr_mode=static_topology  distill_weighting=static_ckr
fake_class_strategy=balanced  fairness_mode=none
generator_init=scratch  knn_k=5  （dynamic CKR 关闭 / fairness 关闭 / public pretraining 关闭）
```

---

## 四、扩展与遗留（默认不运行）

| 目录 | 内容 | 状态 |
|---|---|---|
| experimental/ | run_experiments.py（大型实验编排）、experiment_stats.py、benchmark_checkpointing.py、generate_conclusion_matrix.py、test_experiments.py / test_phase2.py / test_phase3.py | 实验性扩展；默认训练入口与平台均不调用；已修正路径/导入 |
| legacy/ | train_fedavg.py（旧版独立 FedAvg 基线） | 历史复现用；B1 等价实现保留在 train_fedtad.py |
| dataset/Cora/Client10/Louvain_enriched*、frozen*；Client2/Louvain_enriched* | 阶段三富集/冻结划分数据产物 | 扩展实验复现数据；核心流程不触碰 |

---

## 五、保留的完整性保障

1. **接口**：10 个 REST/WS 端点全部保留，响应字段未删未改。
2. **数据**：Cora raw/processed/Louvain 子图、ckr/ 缓存、runs/ 历史输出全部原位保留。
3. **历史兼容**：CLI 参数名未重命名（num_rounds/fake_nodes/distill_steps 等）；checkpoint 序列化格式（含 CKR tracker 状态）未改；静态 CKR 磁盘缓存键未改；`teacher_guided_scratch` 保留为 `--generator_init` 兼容别名。
4. **测试**：核心 3 个测试文件（59+26+4 项）与平台 smoke test（38 项）随仓库交付。
