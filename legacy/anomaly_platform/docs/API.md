# API.md — 平台 API 文档

Base URL: `http://<host>:8000/api`。除 WebSocket 外均需 `Authorization: Bearer <token>`（`/api/auth/login` 获取）。

## 认证

| 方法 | 路径 | 请求 | 响应 |
|---|---|---|---|
| POST | /auth/register | `{username, password, confirm_password}` | `{id, username, created_at}` |
| POST | /auth/login | `{username, password}` | `{access_token, token_type}` |
| GET | /auth/me | — | `{id, username, created_at}` |

## 训练

### POST /training/start — 创建并启动训练任务

请求：`{name, description?, parameters: {canonical 字段...}}`

- 参数经 `config_schema.validate_and_normalize` 严格校验：类型/范围/choices/组合/未知参数；
- 未知参数或非法值 → **422** `{detail: {errors: [...], hint: ...}}`；
- 响应：`ExperimentResponse`（见下）；
- 任务产物自动落盘 `runs/platform/exp_<id>/`（checkpoint / logs / metrics.jsonl / events.jsonl / final_metrics.json）。

### POST /training/stop — 停止当前训练

响应：`{message}`；无运行任务时 400。

### GET /training/status

响应：`{is_running, experiment_id}`。

### POST /training/update — 参数热更新（白名单，下一轮生效）

请求：`{parameter, value}`

- 白名单（round_boundary）：`learning_rate, generator_lr, distill_lr, lambda_subgraph, lambda_sem, lambda_diversity, lambda_disagreement, lambda_feature_norm, contrastive_temperature, generator_steps, distillation_steps, edge_perturb_ratio, knn_k`；
- 非白名单参数 → **400**（"该参数需要重新启动训练任务"）；类型/范围非法 → 422；
- 响应：`ParameterUpdateResponse`（status=pending）；训练进程在下一轮边界应用后状态变 applied 并记录生效轮次。

### GET /training/updates?experiment_id=N

响应：`list[ParameterUpdateResponse]`（参数/旧值/新值/生效轮次/状态/时间戳，审计记录）。

### WS /training/ws/{experiment_id}

服务端推送 JSON：
- `{type:"log", data: "原始日志行"}`
- `{type:"metrics", data: [{metric_name, metric_value, source, round}]}`
- `{type:"status", status, message}`

## 实验

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /experiments/?limit=&offset= | 实验列表（倒序） |
| GET | /experiments/{id} | 实验详情 |
| GET | /experiments/{id}/logs?limit= | 日志分页（历史回放） |
| GET | /experiments/{id}/metrics | 全部指标行 |
| GET | /experiments/{id}/config | 完整配置快照（原始参数 + canonical + git commit + 环境 + resume 提示） |

## 响应字段（ExperimentResponse）

`id, name, description, status, command, parameters, canonical_config, git_commit, environment, task_dir, last_round, exit_code, error_tail, start_time, end_time, best_round, best_val, best_test, created_by, created_at`

## 指标命名（training_metrics.metric_name）

| source | metric_name 示例 | 含义 |
|---|---|---|
| server | global_val / global_test / best_val / best_test / current_round | 全局指标（每轮） |
| client_0 | ce_loss / cl_loss / total_loss / train_samples / zero_anomaly | 客户端本地训练（每轮每客户端） |
| server | generator_loss / semantic_loss / diversity_loss / fake_x_std / … | 生成器状态（每轮） |
| server | distillation_loss / distill_grad_norm / fake_graph_nodes / fake_graph_edges / fake_graph_avg_degree / fake_graph_components | 蒸馏与伪图（每轮） |
| ckr | ckr_c{ci}c{cls} | CKR 服务端权重（客户端×类别，每轮） |
| server | checkpoint_saved | checkpoint 保存标记 |
| server | round_time_sec / generator_peak_gpu_mb / rwr_cache_hit_rate | 资源（每轮） |

## 健康检查

GET /health → `{status: "ok"}`
