# TRAINING_PARAMETERS.md — 训练参数清单

> 本文档由 `research-training/backend/app/config_schema.py` 自动生成, 与代码保持一致。
> 单一命名体系: **前端字段 == canonical 字段 == CLI 参数名** (train_fedtad.py 支持 canonical 别名)。

## 命名体系

```text
Frontend field (TrainingConfig.vue)
  → Canonical config field (config_schema.py, 唯一权威)
  → CLI argument (train_fedtad.py, 同名或别名)
```

canonical 别名 (前端可用别名, 规范化后只保留主键):
- `federated_rounds` → `num_rounds`
- `learning_rate` → `lr`
- `fake_node_count` → `fake_nodes`
- `distillation_steps` → `distill_steps`

## 参数组

### 基础实验参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `anomaly_classes` | str | 4,5,6 | - | start 仅 | before_task | 是 | anomaly_binary 异常类 ID (逗号分隔) |
| `dataset` | str | Cora | Cora | start 仅 | before_task | 是 | 数据集名称 (核心实验仅 Cora) |
| `federated_rounds` | int | None | [1, 10000] | start 仅 | before_task | 是 | 联邦通信轮数 (canonical 名, 等价 num_rounds) |
| `gpu_id` | int | 0 | [0, 7] | start 仅 | before_task | 是 | GPU 编号 |
| `local_epochs` | int | 0 | [0, 100] | start 仅 | before_task | 是 | 本地 epoch 覆盖 (0=使用 num_epochs) |
| `model_seed` | int | -1 | [-1, None] | start 仅 | before_task | 是 | 模型种子 (-1=跟随 seed) |
| `normal_classes` | str | 0,1,2,3 | - | start 仅 | before_task | 是 | anomaly_binary 正常类 ID (逗号分隔) |
| `num_clients` | int | 10 | [2, 100] | start 仅 | before_task | 是 | 客户端数量 |
| `num_epochs` | int | 3 | [1, 100] | start 仅 | before_task | 是 | 客户端本地训练 epoch |
| `num_rounds` | int | 100 | [1, 10000] | start 仅 | before_task | 是 | 联邦通信轮数 (canonical 别名 federated_rounds) |
| `part_delta` | int | 20 | [0, 1000] | start 仅 | before_task | 是 | 分区松弛量 |
| `partition` | str | Louvain | Louvain, Metis | start 仅 | before_task | 是 | 分区策略 (仅读取图拓扑) |
| `partition_seed` | int | -1 | [-1, None] | start 仅 | before_task | 是 | 划分种子 (-1=跟随 seed) |
| `root` | str | ./dataset | - | start 仅 | before_task | 是 | 数据根目录 |
| `seed` | int | 2024 | - | start 仅 | before_task | 是 | 随机种子 |
| `split_seed` | int | -1 | [-1, None] | start 仅 | before_task | 是 | 切分种子 (-1=跟随 seed) |
| `task_mode` | str | anomaly_binary | anomaly_binary, multiclass | start 仅 | before_task | 是 | 任务模式: anomaly_binary (正常=0 异常=1) / multiclass (兼容) |

### 模型参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `dropout` | float | 0.5 | [0.0, 1.0] | start 仅 | before_task | 是 | Dropout 比率 |
| `hid_dim` | int | 64 | [4, 2048] | start 仅 | before_task | 是 | GCN 隐藏层维度 (模型结构, 重启生效) |
| `learning_rate` | float | None | [1e-06, 10.0] | round_boundary | next_round | 否 | 客户端学习率 (canonical 名, 等价 lr) |
| `lr` | float | 0.01 | [1e-06, 10.0] | round_boundary | next_round | 否 | 客户端学习率 (canonical 别名 learning_rate, 轮次边界热更新) |
| `weight_decay` | float | 0.0005 | [0.0, 1.0] | start 仅 | before_task | 是 | 权重衰减 |

### 客户端分类参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `beta` | float | 0.999 | [0.0, 1.0] | start 仅 | before_task | 是 | effective_num 参数 |
| `class_weight_method` | str | inverse | inverse, effective_num | start 仅 | before_task | 是 | 类别权重方法 |
| `use_weighted_ce` | bool | True | - | start 仅 | before_task | 是 | 类别加权交叉熵 (仅用客户端 train 标签) |

### 子图跨视图对比参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `contrastive_anchor_scope` | str | all_nodes | all_nodes, train_nodes | start 仅 | before_task | 是 | 锚点作用域 |
| `contrastive_batch_size` | int | 64 | [1, 8192] | start 仅 | before_task | 是 | 对比锚点批大小 |
| `contrastive_mode` | str | subgraph_cross_view | subgraph_cross_view, none | start 仅 | before_task | 是 | 对比学习模式 |
| `contrastive_temperature` | float | 0.5 | [0.001, 10.0] | round_boundary | next_round | 否 | InfoNCE 温度 (轮次边界热更新) |
| `edge_perturb_ratio` | float | 0.2 | [0.0, 1.0] | round_boundary | next_round | 否 | 边扰动比例 (轮次边界热更新) |
| `lambda_subgraph` | float | 0.1 | [0.0, 100.0] | round_boundary | next_round | 否 | 对比损失权重 (轮次边界热更新) |
| `rwr_cache` | str | enabled | enabled, disabled | start 仅 | before_task | 是 | RWR 子图结构缓存 |
| `rwr_restart_prob` | float | 0.5 | [0.0, 1.0] | start 仅 | before_task | 是 | RWR 重启概率 |
| `rwr_subgraph_size` | int | 5 | [2, 100] | start 仅 | before_task | 是 | RWR 子图采样大小 (缓存键, 重启生效) |

### CKR 参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `ckr_mode` | str | static_topology | static_topology, hybrid_dynamic, dynamic_only, performance_only | start 仅 | before_task | 是 | CKR 模式 (static_topology 为核心) |
| `distill_weighting` | str | static_ckr | none, equal, static_ckr, dynamic_ckr | start 仅 | before_task | 是 | 蒸馏权重来源 (B1-B4 核心开关) |
| `dynamic_ckr_alpha` | float | 0.5 | [0.0, 1.0] | start 仅 | before_task | 是 | 静态/动态混合系数 (实验性) |
| `dynamic_ckr_ema_decay` | float | 0.8 | [0.0, 1.0] | start 仅 | before_task | 是 | CKR EMA 衰减 (实验性) |
| `dynamic_ckr_metric` | str | f1 | f1, recall, confidence | start 仅 | before_task | 是 | 动态 CKR 指标 (实验性) |
| `reliability_holdout_ratio` | float | 0.2 | [0.0, 0.9] | start 仅 | before_task | 是 | 可靠性 holdout 比例 (实验性) |
| `reliability_min_support` | int | 3 | [1, 100000] | start 仅 | before_task | 是 | 最小 support (实验性) |
| `reliability_split_seed` | int | 42 | - | start 仅 | before_task | 是 | 可靠性划分种子 (实验性) |
| `static_ckr_scaling` | str | max | max, minmax, rank | start 仅 | before_task | 是 | 静态 CKR 缩放 |

### 生成器参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `diffusion_beta_end` | float | 0.02 | [0.001, 1.0] | start 仅 | before_task | 是 | 噪声调度 beta 终止 (重启生效) |
| `diffusion_beta_start` | float | 0.0001 | [1e-06, 0.1] | start 仅 | before_task | 是 | 噪声调度 beta 起始 (重启生效) |
| `diffusion_hidden` | int | 256 | [8, 8192] | start 仅 | before_task | 是 | 生成器隐藏维度 (重启生效) |
| `diffusion_steps` | int | 10 | [1, 1000] | start 仅 | before_task | 是 | 扩散采样步数 (生成器结构, 重启生效) |
| `fake_class_strategy` | str | balanced | balanced, prior, reliability | start 仅 | before_task | 是 | 伪节点类别策略 |
| `fake_node_count` | int | 100 | [2, 100000] | start 仅 | before_task | 是 | 每轮伪节点数 (canonical 别名 fake_nodes) |
| `fake_nodes` | int | None | [2, 100000] | start 仅 | before_task | 是 | 每轮伪节点数 (canonical 名) |
| `generator_backprop_mode` | str | checkpointed | full, checkpointed, truncated | start 仅 | before_task | 是 | 生成器反向传播模式 |
| `generator_init` | str | scratch | scratch, teacher_guided_scratch, proxy_pretrained | start 仅 | before_task | 是 | 生成器初始化 (scratch=核心) |
| `generator_lr` | float | 0.001 | [1e-06, 10.0] | round_boundary | next_round | 否 | 生成器学习率 (轮次边界热更新) |
| `generator_output_bound` | str | tanh | tanh, clamp, none | start 仅 | before_task | 是 | 生成器输出约束 |
| `generator_steps` | int | 1 | [1, 100] | round_boundary | next_round | 否 | 生成器更新步数 (轮次边界热更新) |
| `generator_truncate_interval` | int | 2 | [1, 1000] | start 仅 | before_task | 是 | truncated 模式 detach 间隔 |
| `generator_warmup_rounds` | int | 3 | [0, 1000] | start 仅 | before_task | 是 | 生成器热身轮数 |
| `knn_k` | int | 5 | [1, 1000] | round_boundary | next_round | 否 | KNN 伪图近邻数 (轮次边界热更新) |
| `lambda_disagreement` | float | 0.1 | [0.0, 100.0] | round_boundary | next_round | 否 | 分歧损失权重 (轮次边界热更新) |
| `lambda_diversity` | float | 0.1 | [0.0, 100.0] | round_boundary | next_round | 否 | 多样性损失权重 (轮次边界热更新) |
| `lambda_feature_norm` | float | 0.001 | [0.0, 100.0] | round_boundary | next_round | 否 | 特征范数权重 (轮次边界热更新) |
| `lambda_sem` | float | 1.0 | [0.0, 100.0] | round_boundary | next_round | 否 | 语义损失权重 (轮次边界热更新) |

### 蒸馏参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `distill_lr` | float | 0.001 | [1e-06, 10.0] | round_boundary | next_round | 否 | 蒸馏学习率 (轮次边界热更新) |
| `distill_steps` | int | 5 | [1, 1000] | round_boundary | next_round | 否 | 蒸馏步数 (canonical 别名 distillation_steps) |
| `distill_temperature` | float | 1.0 | [0.01, 10.0] | start 仅 | before_task | 是 | 蒸馏温度 |
| `distillation_steps` | int | None | [1, 1000] | round_boundary | next_round | 否 | 蒸馏步数 (canonical 名) |

### checkpoint 与任务参数

| canonical 参数 | 类型 | 默认值 | 合法范围/choices | 热更新策略 | 生效时机 | 需重启 | 说明 |
|---|---|---|---|---|---|---|---|
| `allow_anomaly_majority` | bool | False | - | start 仅 | before_task | 是 | 允许异常为多数类 |
| `auc_threshold` | float | 1.0 | [0.0, 100.0] | start 仅 | before_task | 是 | 低资源客户端提前终止阈值 |
| `checkpoint_dir` | str |  | - | start 仅 | before_task | 是 | checkpoint 保存目录 (平台默认落盘到任务目录) |
| `f1_threshold` | float | 0.1 | [0.0, 100.0] | start 仅 | before_task | 是 | 主指标连续三轮提升阈值 (终止) |
| `resplit_stratified` | bool | True | - | start 仅 | before_task | 是 | 标签映射后分层重划分 |
| `resume_checkpoint` | str |  | - | start 仅 | before_task | 是 | 从 checkpoint 恢复训练 |
| `save_last_checkpoint` | bool | True | - | start 仅 | before_task | 是 | 每轮保存 last.pt |
| `selection_metric` | str |  | , macro_f1, pooled_pr_auc, accuracy | start 仅 | before_task | 是 | 模型选择指标 (空=按任务模式默认) |

## 进入训练命令行的字段 (COMMAND_FIELDS)

共 70 个字段; 别名键在规范化后删除, 只传主键。

## 热更新语义

- `round_boundary`: 训练进程在**每轮开始**检查 `pending_updates.json`, 在轮次边界应用 (优化器 step 之间);
- 应用后写入 `applied_updates.jsonl` 并发 `parameter_update_applied` 事件, 后端落审计记录;
- 非白名单参数由后端 API 直接拒绝 (400: 该参数需要重新启动训练任务);
- 修改模型结构/数据划分/任务语义的参数必须在任务启动前设置。
