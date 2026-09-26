# API_COMPATIBILITY_REPORT.md — 前后端接口兼容报告

> 生成时间：2026-08-02
> 范围：research-training 平台（FastAPI 后端 + Vue 3 前端）与核心算法 train_fedtad.py 之间的全部接口。

---

## 一、接口总览（10 个端点全部保留，无删除）

| HTTP 方法 | 路径 | 请求参数/body | 响应字段 | 前端调用位置 | 后端处理函数 | 状态 |
|---|---|---|---|---|---|---|
| POST | /api/auth/login | `{username, password}` | `{access_token, token_type}` | stores/auth.js:12 → Login.vue:45 | routers/auth.py:13-20 | 有效，未修改 |
| POST | /api/auth/register | `{username, password, confirm_password}` | `{id, username, created_at}` | api/index.js:36-42 → Register.vue:84 | routers/auth.py:23-46 | 有效，未修改 |
| GET | /api/auth/me | Header Bearer token | `{id, username, created_at}` | stores/auth.js:16,29 | routers/auth.py:49-55 | 有效，未修改 |
| POST | /api/training/start | `{name, description, parameters: dict}` | `ExperimentResponse`（id/name/description/status/command/parameters/start_time/end_time/best_round/best_val/best_test/created_by/created_at） | stores/training.js:37 → TrainingConfig.vue:259 | routers/training.py:15+ | **已修复**（参数映射重写，见 §二） |
| POST | /api/training/stop | 无 | `{message}` | TrainingMonitor.vue:107 | routers/training.py:112-117 | 有效，未修改 |
| GET | /api/training/status | 无 | `{is_running, experiment_id}` | stores/training.js:54（当前无 view 调用，保留） | routers/training.py:120-125 | 有效，未修改 |
| WS | /api/training/ws/{experiment_id} | — | `{type:"log"/"metrics"/"status", ...}` | TrainingMonitor.vue:60-85 | routers/training.py:128-138 + websocket_manager.py | 有效，未修改 |
| GET | /api/experiments/ | `limit, offset` | `list[ExperimentResponse]` | ExperimentTable.vue:63 | routers/experiments.py:17-31 | 有效，未修改 |
| GET | /api/experiments/{id} | path | `ExperimentResponse` | TrainingMonitor.vue:89 | routers/experiments.py:34-46 | 有效，未修改 |
| GET | /api/experiments/{id}/logs | `limit, offset` | `list[TrainingLogResponse]` | api/index.js:70-72（历史日志回放链路，当前无 view 调用，保留） | routers/experiments.py:49-65 | 有效，未修改 |
| GET | /api/experiments/{id}/metrics | 无 | `list[TrainingMetricResponse]` | TrainingMonitor.vue:93 | routers/experiments.py:68-80 | 有效，**恢复数据供给**（见 §三） |
| GET | /api/health | 无 | `{status:"ok"}` | 前端未调用（保留供运维） | main.py:68-70 | 有效，未修改 |

---

## 二、已修复的接口：POST /api/training/start 参数映射

### 问题（修改前）

后端 `param_to_flag` 将前端 parameters 键直接映射为 CLI flag，其中 **16 个键在 train_fedtad.py 中不存在**（`use_contrastive`、`lambda_cl`、`cl_tau`、`subgraph_size`、`edge_drop_rate`、`restart_prob`、`glb_epochs`、`it_g`、`it_d`、`lr_g`、`lr_d`、`fedtad_mode`、`num_gen`、`lam1`、`lam2`、`topk`）。前端默认表单会发送全部这些键 → argparse `unrecognized arguments` 退出码 2 → 任务 failed。**用默认配置点击"开始训练"必失败。**

### 修复（修改后，routers/training.py:32-130）

1. 映射表全部替换为 train_fedtad.py 真实存在的 CLI flag；
2. 支持 canonical 配置名别名（平台展示名 → CLI 名）：
   - `federated_rounds` → `--num_rounds`
   - `learning_rate` → `--lr`
   - `local_epochs` → `--local_epochs`
   - `fake_node_count` → `--fake_nodes`
   - `distillation_steps` → `--distill_steps`
   - `knn_k` → `--knn_k`（train_fedtad.py 新增参数，默认 5）
   - 旧键名（`num_rounds`、`lr`、`fake_nodes`、`distill_steps`）仍然有效，**向后兼容**；
3. 布尔参数处理：`use_weighted_ce`、`resplit_stratified` 为 BooleanOptionalAction，False 时正确传 `--no-*`（修复原实现 False 被静默忽略的缺陷）；
4. 不在映射表中的键被忽略（前向兼容，避免未来新增 UI 字段再次破坏命令行）。

### 前端同步修改（TrainingConfig.vue）

- 16 个失效键从表单删除（不留指向不存在参数的按钮/字段）；
- 表单改用 canonical 键名：`federated_rounds`、`learning_rate`、`fake_node_count`、`knn_k`、`generator_steps`、`distillation_steps`、`model_seed`、`partition_seed`、`split_seed`；
- 新增核心配置展示：`task_mode`、`normal_classes`、`anomaly_classes`、`ckr_mode`、`distill_weighting`、`fake_class_strategy`、`contrastive_mode`、`contrastive_anchor_scope`、`edge_perturb_ratio`、`rwr_restart_prob`、`rwr_subgraph_size`、`contrastive_batch_size`、`contrastive_temperature`、`lambda_subgraph`；
- 默认配置 = B4 完整方法（anomaly_binary / static_topology / static_ckr / subgraph_cross_view / balanced / scratch）。

---

## 三、已修复的接口数据供给：指标解析（log_parser.py）

### 问题（修改前）

`log_parser.py` 的正则匹配旧专利版输出（`[S5] 全局模型 val_f1=...` 等），而 train_fedtad.py 实际输出为 `[Global] pooled_pr_auc=...` / `训练结束. 最佳 round=...` → **metrics 表、experiments.best_val/best_test/best_round 列、前端图表全部永远为空**。

### 修复（修改后）

1. `log_parser.py` 重写为匹配当前输出：
   - `[Global] <metric>=<val> (best=<best> @ round <N>)` → 每轮写入 `global_val`、`best_val`、`current_round`（server 源）；
   - `训练结束. 最佳 round=N, best_val(...)=X, best_test(...)=Y` → 写入最终 `best_val`、`best_test`、`current_round`；
2. `training.py::_read_output` 维护 `[Round N]` 头解析的当前轮次并传入 `parse_line`；
3. train_fedtad.py 训练结束打印补充 `best_test(...)`（此前只打印 best_val），使平台 best_test 列/图表有数据来源。

### 数据流（修复后）

```
train_fedtad.py stdout
  → _read_output 逐行读取
  → parse_line(line, current_round)
  → training_metrics 表 (round, metric_name, metric_value, source='server')
  → experiments.best_val/best_test/best_round 列（metric_name 匹配时更新）
  → WebSocket {type:'metrics'} 广播
  → TrainingMonitor.vue store → MetricChart.vue（global 模式画 global_val/best_val，best 模式画 best_val/best_test）
```

---

## 四、后端保留的字段与前端消费对照（未修改）

| 响应字段 | 前端消费位置 |
|---|---|
| ExperimentResponse.id / name / status | ExperimentTable.vue:4-15、TrainingMonitor.vue:13-22 |
| start_time / end_time | ExperimentTable.vue:4-15 |
| best_round / best_val / best_test | ExperimentTable.vue:4-15、TrainingMonitor.vue:13-22（修改前因解析失效恒为空，修改后恢复） |
| command / parameters / description / created_by / created_at | 前端当前不展示但**保留**（历史记录完整性与 API 稳定性） |
| TrainingMetricResponse.{id, experiment_id, round, metric_name, metric_value, source, created_at} | TrainingMonitor.vue:93-95 + store.metricsByRound |
| TrainingLogResponse | api/index.js 的 getExperimentLogs（保留历史日志回放能力） |

---

## 五、新增的 CLI 参数（train_fedtad.py，平台间接依赖）

| 参数 | 默认值 | 说明 | 兼容性 |
|---|---|---|---|
| `--knn_k` | 5 | KNN 伪图近邻数（此前硬编码 5） | 默认值与原行为完全一致，无破坏 |

---

## 六、默认配置变更（平台与算法对齐）

| 参数 | 修改前默认 | 修改后默认 | 影响 |
|---|---|---|---|
| --task_mode | multiclass | anomaly_binary | 平台默认提交即核心异常检测模式 |
| --normal_classes / --anomaly_classes | 必填空 | 0,1,2,3 / 4,5,6 | 默认可运行 |
| --ckr_mode | hybrid_dynamic | static_topology | 核心静态 CKR |
| --distill_weighting | dynamic_ckr | static_ckr | B4 完整方法为默认 |
| --generator_init | teacher_guided_scratch | scratch（旧值为别名） | 等价 public_pretraining=false |

平台参数 `parameters` 字典的键名不重命名旧记录（历史实验记录以 JSON 原样存储可读）；新增 canonical 键为纯增量。

---

## 七、未废弃的接口与后续建议

1. **无接口被删除**；`GET /api/training/status` 与 `GET /api/experiments/{id}/logs` 当前无 view 直接调用，但属于任务状态查询与历史日志回放能力，按保留原则保留。
2. 建议（不在本次范围）：TrainingMonitor 刷新后通过 `getExperimentLogs` 回放历史日志，可补全"刷新页面后日志丢失"的体验缺口。
