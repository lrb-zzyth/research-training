# ARCHITECTURE.md — 系统架构

## 一、两条主线

1. **核心算法主线**：`train_fedtad.py` + `model.py` + `util/`（客户端训练、FedAvg、子图-子图跨视图对比、静态拓扑 CKR、教师引导扩散式伪节点生成、KNN 伪图、CKR 加权知识蒸馏、评估与 checkpoint）。
2. **可视化训练平台主线**：`research-training/`（Vue 3 前端 + FastAPI 后端 + PostgreSQL），用户配置参数、创建任务、实时查看进度与指标、热更新参数、管理结果。

## 二、分层

```
┌─────────────────────────────────────────────────────────────┐
│ 前端 (Vue 3 + Element Plus + ECharts + Pinia)                │
│  TrainingConfig (47 项 canonical 参数 + B1-B4 预设)          │
│  TrainingMonitor (状态卡 / 图表 / CKR 表 / 热更新面板 / 日志)  │
│  ExperimentHistory (历史 / 复制 / 恢复)                       │
└──────────────┬──────────────────────────────────────────────┘
               │ REST /api/* (JWT) + WebSocket /api/training/ws/{id}
┌──────────────▼──────────────────────────────────────────────┐
│ 后端 (FastAPI, research-training/backend/app/)               │
│  config_schema.py   canonical 参数 schema + 校验 (唯一权威)    │
│  routers/training.py  start (argv 数组直传) / stop / status   │
│                      / update (热更新) / updates (审计)        │
│  routers/experiments.py  列表 / 详情 / 日志 / 指标 / 配置快照   │
│  training.py        TrainingRunner: 子进程管理、事件采集、      │
│                     生命周期状态机、错误尾部保存                │
│  log_parser.py      [EVENT] 结构化事件 (第一数据源) + 正则 fallback │
│  models.py          experiments / training_logs /             │
│                     training_metrics / parameter_updates      │
└──────────────┬──────────────────────────────────────────────┘
               │ argv 数组 (无 shell) + stdout 流 + 任务产物目录
┌──────────────▼──────────────────────────────────────────────┐
│ 训练进程 (train_fedtad.py, cwd=仓库根)                        │
│  --emit_events → [EVENT] {json} 结构化事件到 stdout           │
│  --param_update_dir → 每轮边界应用 pending_updates.json       │
│  --task_id / --checkpoint_dir / --log_dir / --events_jsonl   │
│  runs/platform/exp_<id>/: checkpoints / logs / metrics.jsonl │
│  / events.jsonl / final_metrics.json                         │
└─────────────────────────────────────────────────────────────┘
```

## 三、参数契约（单一命名体系）

```
Frontend field (TrainingConfig.vue defaults)
  → canonical field (config_schema.CANONICAL_SCHEMA, 校验+默认值补全)
  → CLI argument (train_fedtad.py, 同名; 别名: federated_rounds→num_rounds,
     learning_rate→lr, fake_node_count→fake_nodes, distillation_steps→distill_steps)
```

- 未知参数 → 422（不静默忽略、不进入命令行）；
- canonical 快照入库（`experiments.canonical_config`），与 `command` 列共同保证「数据库参数 == 实际执行参数」；
- 热更新白名单 15 项（round_boundary），其余参数 requires_restart=True。

## 四、实时指标链

```
train_fedtad.py stdout
  → [EVENT] {json} 行
  → backend training.py::_read_output
  → log_parser.event_to_metrics
  → training_metrics 表 (source: server / client_<id> / ckr)
  → experiments.best_val/best_test/best_round/last_round 更新
  → WebSocket {type:'metrics'} 广播
  → 前端 store → MetricChart / CKR 表
```

事件类型：task_started / round_started / client_training_metric / ckr_update /
global_metric / generator_metric / distillation_metric / checkpoint_saved /
resource_usage / parameter_update_applied / task_finished / task_failed。

## 五、任务生命周期

```
pending → running → finished | failed | stopped
```

- `running`：start_time 写入，子进程以 argv 直传启动；
- 每轮 `round_started` 事件更新 `last_round`；
- 结束：exit_code 入库；failed 时保存最近 200 行日志尾部到 `error_tail`；
- 停止：SIGTERM 整个进程组（5s 后 SIGKILL），不留孤儿；不删除已生成 checkpoint；
- 恢复：`resume_checkpoint` 参数从 checkpoint 续训（参数校验通过后直传训练进程）。

## 六、存储

| 内容 | 位置 |
|---|---|
| 任务/用户/日志/指标/参数审计 | PostgreSQL（experiments / users / training_logs / training_metrics / parameter_updates） |
| 训练产物（checkpoint/指标/事件/日志文件） | `runs/platform/exp_<id>/`（gitignored） |
| 静态 CKR 缓存 | `ckr/*.pt`（gitignored） |
| 数据集 | `dataset/Cora/{raw,processed,Client*/Louvain}` |
