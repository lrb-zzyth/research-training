# RELEASE_PLATFORM_REPORT.md — 平台化改造最终验收报告

> 生成时间：2026-08-02
> 目标：在完整保留并验证现有可视化训练平台的基础上，建立 canonical 参数契约、结构化事件、参数热更新与任务生命周期，使仓库适合上传 GitHub 并供其他用户实际运行。

---

## 1. 当前算法主线

核心专利算法（默认配置 = B4 完整方法）：

1. Louvain 社区发现（仅图拓扑）划分客户端子图；
2. 客户端计算静态拓扑类别知识可靠性 CKR（`ckr_mode=static_topology`，固定节点属性+本地拓扑，不依赖验证/测试指标）；
3. 服务器初始化并下发全局 GCN；
4. 客户端本地训练：类别加权交叉熵（仅 train 标签）+ 子图-子图跨视图对比学习（边扰动→双视图→RWR 局部子图→中心特征置零→共享 GCN→mean readout→InfoNCE；正负样本=跨视图同锚点/不同锚点，不读取标签）；
5. 客户端上传：本地模型 state_dict、静态 CKR、必要非节点级统计量（单进程模拟）；
6. FedAvg 初步聚合（按样本量加权，no_grad）；
7. 教师引导扩散式生成器（scratch，从高斯噪声+类别条件，冻结教师/全局反馈训练）；
8. KNN 伪图构造（`--knn_k`，只使用伪特征相似度，不读客户端真实 edge_index）；
9. CKR 按类别融合客户端教师知识；
10. 伪图上教师→全局 KL 蒸馏（生成器阶段只更新生成器，蒸馏阶段只更新全局模型）。

默认配置：`task_mode=anomaly_binary, normal_classes=0,1,2,3, anomaly_classes=4,5,6, contrastive_mode=subgraph_cross_view, contrastive_anchor_scope=all_nodes, ckr_mode=static_topology, distill_weighting=static_ckr, fake_class_strategy=balanced, generator_init=scratch, fairness_mode=none`。动态 CKR / fairness / 公共代理预训练均默认关闭并标注实验性。

## 2. 当前平台主线

`research-training/`：Vue 3 + Element Plus + ECharts + Pinia 前端；FastAPI + SQLAlchemy(async) + asyncpg + PostgreSQL 后端；WebSocket 实时通信；asyncio subprocess（argv 数组直传）管理训练进程。平台功能：参数配置（47 项 canonical 字段 + B1–B4 预设）、任务创建/停止/状态/生命周期、实时日志与指标可视化、CKR 展示、参数热更新与审计、实验历史与配置快照、从 checkpoint 恢复。

## 3. 完整用户操作流程

见 docs/PLATFORM_GUIDE.md §一（登录 → 配置/预设 → 创建任务 → 监控页实时图表/日志/CKR → 热更新 → 完成/失败处理 → 历史/复制/恢复）。

## 4. 前端页面和功能

| 页面 | 功能 |
|---|---|
| Dashboard | 布局壳/菜单/用户 |
| Login / Register | JWT 登录注册 |
| TrainingConfig | 47 项参数表单 + B1–B4 预设 + 重置；提交后跳转监控 |
| TrainingMonitor | 状态卡（轮次/阶段/耗时/最佳/git commit/退出码/产物目录）、全局指标图、客户端损失图、生成器/蒸馏图、CKR 权重表、实时日志、失败横幅、**参数热更新面板**（白名单选择+审计表） |
| ExperimentHistory | 实验列表（ExperimentTable）→ 详情/复制 |
| Settings | 语言切换 |

## 5. 后端任务执行链

```
POST /api/training/start
  → config_schema.validate_and_normalize (类型/范围/choices/组合/未知参数 422)
  → 建任务行 (parameters + canonical_config + git_commit + environment)
  → task_dir = runs/platform/exp_<id>/
  → _build_argv (canonical→CLI, argv 数组, 任务产物路径自动注入)
  → TrainingRunner.start(argv) → create_subprocess_exec (*argv, 无 shell)
  → _read_output: [EVENT] 结构化事件第一数据源 + 正则 fallback
      → training_metrics 入库 + experiments.best_*/last_round 更新
      → WS 广播 log/metrics/status
  → _wait_process: status=finished/failed/stopped + exit_code + error_tail
  → POST /api/training/update: 白名单校验 → pending 文件 + parameter_updates 审计
  → 训练进程轮次边界应用 → parameter_update_applied 事件 → 审计置 applied
```

## 6. Canonical 参数 schema

`research-training/backend/app/config_schema.py`：71 个字段，每字段含 `type/default/min/max/choices/group/hot_update_policy/requires_restart/effective_stage/description`。单一命名体系（前端字段 == canonical == CLI 参数名，4 个别名：federated_rounds→num_rounds、learning_rate→lr、fake_node_count→fake_nodes、distillation_steps→distill_steps）。

## 7. 前端—后端—训练代码参数映射

- 前端字段 ⊆ canonical schema（契约测试 1a 验证，47 项）；
- canonical → CLI：COMMAND_FIELDS（全部存在于 train_fedtad.py --help，契约测试 3a 验证）；
- 无"接受但永不生效"字段（契约测试 4a 验证）；
- 完整清单：docs/TRAINING_PARAMETERS.md（由 schema 自动生成）。

## 8. 可热更新参数列表（round_boundary，下一轮生效）

`learning_rate, generator_lr, distill_lr, lambda_subgraph, lambda_sem, lambda_diversity, lambda_disagreement, lambda_feature_norm, contrastive_temperature, generator_steps, distillation_steps, edge_perturb_ratio, knn_k`（共 13 个 canonical 名；契约测试 6 验证后端白名单与训练进程白名单一致）。

## 9. 必须重启参数列表

数据集/任务（dataset、task_mode、normal_classes、anomaly_classes、num_clients、partition、seed 系列）、模型结构（hid_dim、diffusion_hidden、diffusion_steps、diffusion_beta_*）、数据划分（rwr_subgraph_size、reliability_*）、方法开关（contrastive_mode、ckr_mode、distill_weighting、use_weighted_ce、fake_class_strategy、generator_init）等其余全部参数。修改时前端/后端提示"该参数需要重新启动训练任务"（API 400）。

## 10. 参数生效时机

- `before_task`：任务启动时生效（命令行构建阶段）；
- `next_round`：训练进程在每轮 `[Round N]` 开始前检查 `pending_updates.json`，在优化器 step 之间应用（LR 通过 optimizer param_groups 真实更新），绝对不在 step 中途修改。

## 11. 实时指标事件结构

训练进程输出 `[EVENT] {json}` 结构化事件（`--emit_events`），示例：

```json
{"event": "client_training_metric", "task_id": "5", "round": 0,
 "stage": "client_training", "client_id": 0, "ce_loss": 0.702758,
 "cl_loss": 0.123, "total_loss": 0.715, "train_samples": 812,
 "class_counts": [640, 172], "zero_anomaly": false}
```

事件类型：task_started / round_started / client_training_metric / ckr_update / global_metric / generator_metric / distillation_metric / checkpoint_saved / resource_usage / parameter_update_applied / task_finished / task_failed。后端 `log_parser.event_to_metrics` 优先解析结构化事件，自然语言正则仅作 fallback。

## 12. 前端实际展示指标

global_val/global_test/best_val/best_test（曲线）；ce_loss/cl_loss/total_loss（客户端曲线，均值）；generator_loss/semantic_loss/diversity_loss/distillation_loss/fake_x_std（曲线）；ckr_c{ci}c{cls}（CKR 表）；round_time_sec/generator_peak_gpu_mb（状态卡）；无数据项显示为空（unavailable），不伪造 0。

## 13. 修复的接口断裂

| 断裂 | 修复 |
|---|---|
| 16 个 CLI flag 不存在 → 默认表单必失败 | 前端字段重写 + 后端 canonical 映射 + argv 数组直传 |
| log_parser 正则零匹配 → 指标/图表恒空 | [EVENT] 结构化事件第一数据源 + 新正则 fallback |
| 训练结束不打印 best_test | 补充 `best_test(...)`（train_fedtad.py） |
| 任务不落盘任何产物 | 平台自动注入 checkpoint_dir/log_dir/metrics_jsonl/events_jsonl/final_metrics_json 到 task_dir |
| 布尔 False 静默忽略 | `--no-*` 形式（BooleanOptionalAction） |
| 历史实验新列为 NULL → 列表接口 500 | 迁移回填默认值 + schema 字段 Optional |
| 命令字符串 split 注入风险 | create_subprocess_exec(*argv) 直接传参 |
| 未知参数进入命令行 | 422 拒绝 |

## 14. 删除的文件

见 docs/archive/DELETED_FILES_MANIFEST.md（gradate_contrastive.py、ParameterForm.vue、.env.bak 等 + 8 个文件迁移 experimental//legacy/）。

## 15. 保留的非算法文件及原因

全部前端/后端/数据库/WS/日志解析/配置/部署文件（平台主线）；experimental/（扩展实验系统，历史复现价值）；legacy/（旧 FedAvg 基线）；docs/archive/（历史审计与实验报告）；pretrain_diffusion.py（核心脚本模块级导入，默认关闭）；louvain/（vendored 社区发现库）；runs/（gitignored 实验产物）。

## 16. 最终目录树

```
FedTAD/
├── train_fedtad.py  model.py  pretrain_diffusion.py  requirements.txt
├── util/            # 核心工具 (10 模块)
├── research-training/
│   ├── backend/app/      # config_schema / routers / training / log_parser / models
│   ├── backend/test_backend_smoke.py  backend/test_param_contract.py
│   └── frontend/src/     # views / components / api / stores / router
├── experimental/    legacy/    docs/    docs/archive/
├── louvain/  dataset/  ckr/  runs/(gitignored)
└── README.md  RELEASE_PLATFORM_REPORT.md
```

## 17. 核心算法测试结果

| 测试 | 结果 |
|---|---|
| test_all.py（59 项单元） | **59/59 通过** |
| test_smoke.py（26 项冒烟） | **26/26 通过** |
| test_smoke_train.py（4 组真实训练） | **A/B/C/D 全部通过** |

## 18. 后端测试结果

| 测试 | 结果 |
|---|---|
| test_backend_smoke.py（真实服务端到端） | **57/57 通过**（健康/鉴权/参数转换/配置快照/B1 任务完成链/B4 任务+热更新+checkpoint 落盘/失败路径 422+failed+error_tail/日志指标接口/历史兼容） |
| test_param_contract.py（参数契约） | **16/16 通过** |

## 19. 前端生产构建结果

`npm run build` **成功**（6.1s；仅 chunk 体积提示，无错误、无缺失模块）。

## 20. B1 平台端到端结果

经平台 API 创建 B1 型任务（anomaly_binary、无对比、无蒸馏、1 轮）：创建 200 → canonical 快照入库 → 训练 finished → 日志含 Core method 声明 → global_val/best_val 指标入库 → best_round/best_val 写入实验记录（smoke 6a–6e 通过）。

## 21. B4 平台端到端结果

经平台 API 创建 B4 完整方法任务（3 轮、对比+CKR+生成器+蒸馏）：finished；客户端指标（ce_loss/total_loss）、CKR 指标（ckr_c*）、生成器/蒸馏指标（generator_loss/distillation_loss/fake_graph_edges）、资源指标（round_time_sec/peak_gpu_mb）全部入库；best checkpoint 落盘 `task_dir/checkpoints/best.pt`（smoke 6B-a…6B-j 通过）。

## 22. 参数调整端到端结果

B4 训练中（round 1 开始后）提交 `learning_rate: 0.01→0.005`：请求 accepted（pending）→ 训练进程在下一轮边界应用 → `parameter_update_applied` 事件 → 审计记录 applied + effective_round≥2；非白名单参数（hid_dim）被 400 拒绝并提示重启（smoke 6B-b…6B-e 通过）。验证了优化器 LR 真实变化（训练侧断言 + 审计记录一致）。

## 23. 数据库和 checkpoint 验证

- PostgreSQL：experiments（含 canonical_config/git_commit/environment/task_dir/last_round/exit_code/error_tail）、training_logs、training_metrics（server/client/ckr 源）、parameter_updates（pending→applied 审计）；迁移对历史记录回填默认值，列表接口不再 500；
- checkpoint：best.pt 由平台自动落盘到 `runs/platform/exp_<id>/checkpoints/`，`--resume_checkpoint` 可恢复；events.jsonl/metrics.jsonl/final_metrics.json 完整。

## 24. 敏感信息扫描结果

- git 跟踪文件扫描 `password|secret|token|api_key|BEGIN PRIVATE KEY|/home/|postgresql://`：命中均为功能代码（auth 逻辑、JWT 拦截器）或本地开发文档（backend/README.md 明确标注"仅为本地开发密码"）；
- `.env` 未跟踪（gitignore `**/.env`）；`.env.example` 已改为占位值（`change-me`）；config.py 代码默认值改为占位；
- 已清除：根 `__pycache__`、`.env.bak`、legacy/train_fedavg.py 中的绝对路径 `/home/ai2/work/fedtad`；
- `.claude/` 已加入 gitignore；
- 已知说明：backend/README.md 中的本地开发密码（fedtad_password）为本地 quickstart 文档，显式标注非生产使用。

## 25. 仍存在的限制

1. 无头环境未做真实浏览器验证（以 API 全链路 + 前端生产构建代替）；
2. 单进程联邦模拟（隐私边界在生成/蒸馏损失层成立，非形式化差分隐私）；
3. WebSocket 无鉴权（既有设计）；
4. 热更新白名单仅覆盖经审计安全的 13 个参数，其余需重启；
5. 前端 stage 展示由日志关键词推导（展示层，非伪造指标）；
6. 扩展层（experimental/）需要 GPU 与较长时间才能完整运行。

## 26. GitHub 发布前最后检查清单

- [x] README 重写（研究问题/算法/平台能力/参数/热更新/重启参数/B1-B4/安装/启动/首个任务/CLI/恢复/目录/限制/隐私/Cora 合成异常）
- [x] docs/：ARCHITECTURE / API / TRAINING_PARAMETERS / PLATFORM_GUIDE / REPRODUCTION
- [x] .env 与凭据不入库；.env.example 占位值
- [x] 无本地绝对路径、无机器信息
- [x] 无构建产物/缓存/checkpoint/runs 入库（gitignore 覆盖）
- [x] 核心算法 + 平台 + 扩展测试全部通过
- [x] B1 / B4 / 热更新平台端到端验证通过
- [x] 前端生产构建通过
- [x] 历史审计报告归档到 docs/archive/（可选删除）
- [ ] 提交前 `git status` 复查：确认 untracked 中无 `.env`/凭据/大文件（本次验证时 runs/、ckr/、dataset .pt 均为 gitignored）
