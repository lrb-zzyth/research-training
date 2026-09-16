# PLATFORM_GUIDE.md — 可视化训练平台使用指南

## 一、完整用户流程

```
打开前端 (http://localhost:5173)
  ↓ 注册/登录（默认管理员 admin / admin123）
  ↓ 进入「训练配置」页
  ↓ 选择 B1–B4 预设模板 或 手动配置 47 项 canonical 参数
  ↓ 点击「开始训练」→ 后端校验 → 创建任务 → 启动训练进程
  ↓ 自动跳转「训练监控」页
  ↓ 实时看到：状态卡（轮次/阶段/耗时）、全局指标曲线、
  ↓         客户端损失曲线、生成器/蒸馏曲线、CKR 权重表、实时日志
  ↓ 运行中可在「参数热更新」面板修改白名单参数（下一轮生效，有审计记录）
  ↓ 任务完成/失败/停止后状态正确更新，失败原因可见
  ↓ 「实验历史」页查看/复制/恢复历史任务
```

## 二、B1–B4 正式方案

| 预设 | 组合 | 说明 |
|---|---|---|
| B1 | `use_weighted_ce=false, contrastive_mode=none, distill_weighting=none` | FedAvg（普通交叉熵） |
| B2 | `use_weighted_ce=true, contrastive_mode=none, distill_weighting=none` | + 类别加权交叉熵 |
| B3 | `use_weighted_ce=true, contrastive_mode=subgraph_cross_view, distill_weighting=none` | + 子图跨视图对比 |
| B4 | `use_weighted_ce=true, contrastive_mode=subgraph_cross_view, distill_weighting=static_ckr, ckr_mode=static_topology` | 完整方法（默认） |

预设只组合既有字段，不引入新参数；高级设置中可查看/修改全部参数。

## 三、实时可视化说明

| 区域 | 数据来源 | 说明 |
|---|---|---|
| 状态卡 | 事件 + 数据库 | 当前轮次/总轮次、阶段（由日志关键词推导）、已运行时间、最佳指标、git commit、退出码、产物目录 |
| 全局指标图 | server 指标 | global_val / global_test / best_val / best_test |
| 客户端损失图 | client_<id> 指标 | ce_loss / cl_loss / total_loss（客户端均值） |
| 生成器/蒸馏图 | server 指标 | generator_loss / semantic_loss / diversity_loss / distillation_loss / fake_x_std |
| CKR 表 | ckr 指标 | 最近一轮 客户端×类别 服务端权重 |
| 日志控制台 | WebSocket | 实时行流；失败时红色横幅显示错误尾部 |

未产生数据的项显示为空（unavailable 语义），不伪造 0。

## 四、参数热更新

1. 训练运行中，监控页「参数热更新」选择白名单参数（如 `learning_rate`）并输入新值；
2. 后端校验 → 写入 pending 文件 + 审计记录（status=pending）；
3. 训练进程在**下一轮开始**时应用（优化器 LR 真实更新），并发 `parameter_update_applied` 事件；
4. 审计记录变为 applied 并记录生效轮次；「参数更新历史」表可查全部记录；
5. 非白名单参数（如 `hid_dim`、`dataset`）→ 400 提示"该参数需要重新启动训练任务"。

## 五、任务生命周期与失败处理

- 状态：pending → running → finished / failed / stopped；
- 失败任务：exit_code、日志尾部（error_tail）、最后有效轮次均可查询；
- 停止：SIGTERM 进程组，不删除已生成 checkpoint；
- 恢复：新任务配置中设置 `resume_checkpoint`（历史任务产物目录下的 best.pt / last.pt）。

## 六、实验历史与复现

- 每个任务保存：原始参数 + canonical 配置 + git commit + 运行环境 + 产物目录；
- `GET /api/experiments/{id}/config` 导出完整快照（复制为新任务时作为参数源）；
- 产物目录含 events.jsonl / metrics.jsonl / final_metrics.json / checkpoints，可完整复现。

## 七、故障排查

| 现象 | 排查 |
|---|---|
| 开始训练 422 | 参数校验失败：查看 detail.errors（未知参数/越界/组合错误） |
| 任务 failed | 监控页红色横幅显示错误尾部；`GET /api/experiments/{id}` 查看 exit_code |
| 图表为空 | 检查任务是否完成至少一轮；指标由 [EVENT] 事件驱动 |
| 热更新被拒 | 确认参数在白名单（页面提示需要重启的请新建任务） |
| 数据库连接失败 | 检查 backend/.env 的 DATABASE_URL 与 PostgreSQL 状态 |
