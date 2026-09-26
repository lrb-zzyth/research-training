"""
Canonical training configuration schema.

单一参数命名体系: 前端字段 == canonical 字段 == CLI 参数名 (train_fedtad.py)。
后端负责:
  1. 类型/范围/choices/组合校验;
  2. 默认值补全 (生成 canonical_config 快照);
  3. 热更新白名单判定 (round_boundary vs restart_required)。

未知参数一律 422 拒绝, 不得拼接进命令行。
"""
from typing import Any

# ---------------------------------------------------------------------------
#  canonical schema: 字段 -> 元数据
# ---------------------------------------------------------------------------
# hot_update_policy:
#   'start'          仅任务启动前可配置 (修改需重启任务)
#   'round_boundary' 可在当前通信轮结束后、下一轮开始前生效 (训练进程每轮检查)
# requires_restart:  True = 修改该参数必须重新启动训练任务
# effective_stage:  'before_task' | 'next_round'
CANONICAL_SCHEMA: dict[str, dict] = {
    # ---- 基础实验参数 ----
    "dataset": {"type": "str", "default": "Cora", "choices": ["Cora"],
                "group": "basic", "hot_update_policy": "start",
                "requires_restart": True, "effective_stage": "before_task",
                "description": "数据集名称 (核心实验仅 Cora)"},
    "task_mode": {"type": "str", "default": "anomaly_binary",
                  "choices": ["anomaly_binary", "multiclass"],
                  "group": "basic", "hot_update_policy": "start",
                  "requires_restart": True, "effective_stage": "before_task",
                  "description": "任务模式: anomaly_binary (正常=0 异常=1) / multiclass (兼容)"},
    "normal_classes": {"type": "str", "default": "0,1,2,3", "group": "basic",
                       "hot_update_policy": "start", "requires_restart": True,
                       "effective_stage": "before_task",
                       "description": "anomaly_binary 正常类 ID (逗号分隔)"},
    "anomaly_classes": {"type": "str", "default": "4,5,6", "group": "basic",
                        "hot_update_policy": "start", "requires_restart": True,
                        "effective_stage": "before_task",
                        "description": "anomaly_binary 异常类 ID (逗号分隔)"},
    "num_clients": {"type": "int", "default": 10, "min": 2, "max": 100,
                    "group": "basic", "hot_update_policy": "start",
                    "requires_restart": True, "effective_stage": "before_task",
                    "description": "客户端数量"},
    "num_rounds": {"type": "int", "default": 100, "min": 1, "max": 10000,
                   "group": "basic", "hot_update_policy": "start",
                   "requires_restart": True, "effective_stage": "before_task",
                   "description": "联邦通信轮数 (canonical 别名 federated_rounds)"},
    "federated_rounds": {"type": "int", "default": None, "min": 1, "max": 10000,
                         "group": "basic", "hot_update_policy": "start",
                         "requires_restart": True, "effective_stage": "before_task",
                         "description": "联邦通信轮数 (canonical 名, 等价 num_rounds)"},
    "num_epochs": {"type": "int", "default": 3, "min": 1, "max": 100,
                   "group": "basic", "hot_update_policy": "start",
                   "requires_restart": True, "effective_stage": "before_task",
                   "description": "客户端本地训练 epoch"},
    "local_epochs": {"type": "int", "default": 0, "min": 0, "max": 100,
                     "group": "basic", "hot_update_policy": "start",
                     "requires_restart": True, "effective_stage": "before_task",
                     "description": "本地 epoch 覆盖 (0=使用 num_epochs)"},
    "seed": {"type": "int", "default": 2024, "group": "basic",
             "hot_update_policy": "start", "requires_restart": True,
             "effective_stage": "before_task", "description": "随机种子"},
    "model_seed": {"type": "int", "default": -1, "min": -1, "group": "basic",
                   "hot_update_policy": "start", "requires_restart": True,
                   "effective_stage": "before_task", "description": "模型种子 (-1=跟随 seed)"},
    "partition_seed": {"type": "int", "default": -1, "min": -1, "group": "basic",
                       "hot_update_policy": "start", "requires_restart": True,
                       "effective_stage": "before_task", "description": "划分种子 (-1=跟随 seed)"},
    "split_seed": {"type": "int", "default": -1, "min": -1, "group": "basic",
                   "hot_update_policy": "start", "requires_restart": True,
                   "effective_stage": "before_task", "description": "切分种子 (-1=跟随 seed)"},
    "gpu_id": {"type": "int", "default": 0, "min": 0, "max": 7, "group": "basic",
               "hot_update_policy": "start", "requires_restart": True,
               "effective_stage": "before_task", "description": "GPU 编号"},
    "root": {"type": "str", "default": "./dataset", "group": "basic",
             "hot_update_policy": "start", "requires_restart": True,
             "effective_stage": "before_task", "description": "数据根目录"},
    "partition": {"type": "str", "default": "Louvain", "choices": ["Louvain", "Metis"],
                  "group": "basic", "hot_update_policy": "start",
                  "requires_restart": True, "effective_stage": "before_task",
                  "description": "分区策略 (仅读取图拓扑)"},
    "part_delta": {"type": "int", "default": 20, "min": 0, "max": 1000,
                   "group": "basic", "hot_update_policy": "start",
                   "requires_restart": True, "effective_stage": "before_task",
                   "description": "分区松弛量"},

    # ---- 模型参数 ----
    "hid_dim": {"type": "int", "default": 64, "min": 4, "max": 2048,
                "group": "model", "hot_update_policy": "start",
                "requires_restart": True, "effective_stage": "before_task",
                "description": "GCN 隐藏层维度 (模型结构, 重启生效)"},
    "dropout": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                "group": "model", "hot_update_policy": "start",
                "requires_restart": True, "effective_stage": "before_task",
                "description": "Dropout 比率"},
    "lr": {"type": "float", "default": 0.01, "min": 1e-6, "max": 10.0,
           "group": "model", "hot_update_policy": "round_boundary",
           "requires_restart": False, "effective_stage": "next_round",
           "description": "客户端学习率 (canonical 别名 learning_rate, 轮次边界热更新)"},
    "learning_rate": {"type": "float", "default": None, "min": 1e-6, "max": 10.0,
                      "group": "model", "hot_update_policy": "round_boundary",
                      "requires_restart": False, "effective_stage": "next_round",
                      "description": "客户端学习率 (canonical 名, 等价 lr)"},
    "weight_decay": {"type": "float", "default": 5e-4, "min": 0.0, "max": 1.0,
                     "group": "model", "hot_update_policy": "start",
                     "requires_restart": True, "effective_stage": "before_task",
                     "description": "权重衰减"},

    # ---- 客户端分类参数 ----
    "use_weighted_ce": {"type": "bool", "default": None, "group": "classification",
                        "hot_update_policy": "start", "requires_restart": True,
                        "effective_stage": "before_task",
                        "description": "类别加权交叉熵。默认 None = 不传该参数, 由训练端按 task_mode 自适应 (anomaly_binary=True, multiclass=False; 实测 multiclass 下加权 CE 为 -2.10 的负贡献)"},
    "class_weight_method": {"type": "str", "default": "inverse",
                            "choices": ["inverse", "effective_num"],
                            "group": "classification", "hot_update_policy": "start",
                            "requires_restart": True, "effective_stage": "before_task",
                            "description": "类别权重方法"},
    "beta": {"type": "float", "default": 0.999, "min": 0.0, "max": 1.0,
             "group": "classification", "hot_update_policy": "start",
             "requires_restart": True, "effective_stage": "before_task",
             "description": "effective_num 参数"},

    # ---- 子图跨视图对比参数 ----
    "contrastive_mode": {"type": "str", "default": "subgraph_cross_view",
                         "choices": ["subgraph_cross_view", "none"],
                         "group": "contrastive", "hot_update_policy": "start",
                         "requires_restart": True, "effective_stage": "before_task",
                         "description": "对比学习模式"},
    "contrastive_anchor_scope": {"type": "str", "default": "all_nodes",
                                 "choices": ["all_nodes", "train_nodes"],
                                 "group": "contrastive", "hot_update_policy": "start",
                                 "requires_restart": True, "effective_stage": "before_task",
                                 "description": "锚点作用域"},
    "contrastive_batch_size": {"type": "int", "default": 64, "min": 1, "max": 8192,
                               "group": "contrastive", "hot_update_policy": "start",
                               "requires_restart": True, "effective_stage": "before_task",
                               "description": "对比锚点批大小"},
    "contrastive_temperature": {"type": "float", "default": 0.2, "min": 1e-3, "max": 10.0,
                                "group": "contrastive", "hot_update_policy": "round_boundary",
                                "requires_restart": False, "effective_stage": "next_round",
                                "description": "InfoNCE 温度 (轮次边界热更新)"},
    "lambda_subgraph": {"type": "float", "default": 0.1, "min": 0.0, "max": 100.0,
                        "group": "contrastive", "hot_update_policy": "round_boundary",
                        "requires_restart": False, "effective_stage": "next_round",
                        "description": "对比损失权重 (轮次边界热更新)"},
    "edge_perturb_ratio": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0,
                           "group": "contrastive", "hot_update_policy": "round_boundary",
                           "requires_restart": False, "effective_stage": "next_round",
                           "description": "边扰动比例 (轮次边界热更新)"},
    "rwr_restart_prob": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                         "group": "contrastive", "hot_update_policy": "start",
                         "requires_restart": True, "effective_stage": "before_task",
                         "description": "RWR 重启概率"},
    "rwr_subgraph_size": {"type": "int", "default": 5, "min": 2, "max": 100,
                          "group": "contrastive", "hot_update_policy": "start",
                          "requires_restart": True, "effective_stage": "before_task",
                          "description": "RWR 子图采样大小 (缓存键, 重启生效)"},
    "rwr_cache": {"type": "str", "default": "enabled", "choices": ["enabled", "disabled"],
                  "group": "contrastive", "hot_update_policy": "start",
                  "requires_restart": True, "effective_stage": "before_task",
                  "description": "RWR 子图结构缓存"},

    # ---- CKR 参数 ----
    "ckr_mode": {"type": "str", "default": "static_topology",
                 "choices": ["static_topology", "hybrid_dynamic", "dynamic_only",
                             "performance_only"],
                 "group": "ckr", "hot_update_policy": "start",
                 "requires_restart": True, "effective_stage": "before_task",
                 "description": "CKR 模式 (static_topology 为核心)"},
    "static_ckr_scaling": {"type": "str", "default": "max",
                           "choices": ["max", "minmax", "rank"],
                           "group": "ckr", "hot_update_policy": "start",
                           "requires_restart": True, "effective_stage": "before_task",
                           "description": "静态 CKR 缩放"},
    "dynamic_ckr_metric": {"type": "str", "default": "f1",
                           "choices": ["f1", "recall", "confidence"],
                           "group": "ckr", "hot_update_policy": "start",
                           "requires_restart": True, "effective_stage": "before_task",
                           "description": "动态 CKR 指标 (实验性)"},
    "dynamic_ckr_alpha": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                          "group": "ckr", "hot_update_policy": "start",
                          "requires_restart": True, "effective_stage": "before_task",
                          "description": "静态/动态混合系数 (实验性)"},
    "dynamic_ckr_ema_decay": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0,
                              "group": "ckr", "hot_update_policy": "start",
                              "requires_restart": True, "effective_stage": "before_task",
                              "description": "CKR EMA 衰减 (实验性)"},
    "reliability_holdout_ratio": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.9,
                                  "group": "ckr", "hot_update_policy": "start",
                                  "requires_restart": True, "effective_stage": "before_task",
                                  "description": "可靠性 holdout 比例 (实验性)"},
    "reliability_min_support": {"type": "int", "default": 3, "min": 1, "max": 100000,
                                "group": "ckr", "hot_update_policy": "start",
                                "requires_restart": True, "effective_stage": "before_task",
                                "description": "最小 support (实验性)"},
    "reliability_split_seed": {"type": "int", "default": 42, "group": "ckr",
                               "hot_update_policy": "start", "requires_restart": True,
                               "effective_stage": "before_task",
                               "description": "可靠性划分种子 (实验性)"},
    "distill_weighting": {"type": "str", "default": "static_ckr",
                          "choices": ["none", "equal", "static_ckr", "dynamic_ckr"],
                          "group": "ckr", "hot_update_policy": "start",
                          "requires_restart": True, "effective_stage": "before_task",
                          "description": "蒸馏权重来源 (B1-B4 核心开关)"},

    # ---- 生成器参数 ----
    "generator_init": {"type": "str", "default": "scratch",
                       "choices": ["scratch", "teacher_guided_scratch", "proxy_pretrained"],
                       "group": "generator", "hot_update_policy": "start",
                       "requires_restart": True, "effective_stage": "before_task",
                       "description": "生成器初始化 (scratch=核心)"},
    "diffusion_steps": {"type": "int", "default": 20, "min": 1, "max": 1000,
                        "group": "generator", "hot_update_policy": "start",
                        "requires_restart": True, "effective_stage": "before_task",
                        "description": "扩散采样步数 (生成器结构, 重启生效)"},
    "diffusion_hidden": {"type": "int", "default": 256, "min": 8, "max": 8192,
                         "group": "generator", "hot_update_policy": "start",
                         "requires_restart": True, "effective_stage": "before_task",
                         "description": "生成器隐藏维度 (重启生效)"},
    "diffusion_beta_start": {"type": "float", "default": 1e-4, "min": 1e-6, "max": 0.1,
                             "group": "generator", "hot_update_policy": "start",
                             "requires_restart": True, "effective_stage": "before_task",
                             "description": "噪声调度 beta 起始 (重启生效)"},
    "diffusion_beta_end": {"type": "float", "default": 0.5, "min": 1e-3, "max": 1.0,
                           "group": "generator", "hot_update_policy": "start",
                           "requires_restart": True, "effective_stage": "before_task",
                           "description": "噪声调度 beta 终止 (重启生效)"},
    "generator_output_bound": {"type": "str", "default": "tanh",
                               "choices": ["tanh", "clamp", "none"],
                               "group": "generator", "hot_update_policy": "start",
                               "requires_restart": True, "effective_stage": "before_task",
                               "description": "生成器输出约束"},
    "generator_backprop_mode": {"type": "str", "default": "checkpointed",
                                "choices": ["full", "checkpointed", "truncated"],
                                "group": "generator", "hot_update_policy": "start",
                                "requires_restart": True, "effective_stage": "before_task",
                                "description": "生成器反向传播模式"},
    "generator_truncate_interval": {"type": "int", "default": 2, "min": 1, "max": 1000,
                                    "group": "generator", "hot_update_policy": "start",
                                    "requires_restart": True,
                                    "effective_stage": "before_task",
                                    "description": "truncated 模式 detach 间隔"},
    "fake_node_count": {"type": "int", "default": 100, "min": 2, "max": 100000,
                        "group": "generator", "hot_update_policy": "start",
                        "requires_restart": True, "effective_stage": "before_task",
                        "description": "每轮伪节点数 (canonical 别名 fake_nodes)"},
    "fake_nodes": {"type": "int", "default": None, "min": 2, "max": 100000,
                   "group": "generator", "hot_update_policy": "start",
                   "requires_restart": True, "effective_stage": "before_task",
                   "description": "每轮伪节点数 (canonical 名)"},
    "fake_class_strategy": {"type": "str", "default": "balanced",
                            "choices": ["balanced", "prior", "reliability"],
                            "group": "generator", "hot_update_policy": "start",
                            "requires_restart": True, "effective_stage": "before_task",
                            "description": "伪节点类别策略"},
    "generator_warmup_rounds": {"type": "int", "default": 3, "min": 0, "max": 1000,
                                "group": "generator", "hot_update_policy": "start",
                                "requires_restart": True, "effective_stage": "before_task",
                                "description": "生成器热身轮数"},
    "generator_steps": {"type": "int", "default": 1, "min": 1, "max": 100,
                        "group": "generator", "hot_update_policy": "round_boundary",
                        "requires_restart": False, "effective_stage": "next_round",
                        "description": "生成器更新步数 (轮次边界热更新)"},
    "generator_lr": {"type": "float", "default": 1e-3, "min": 1e-6, "max": 10.0,
                     "group": "generator", "hot_update_policy": "round_boundary",
                     "requires_restart": False, "effective_stage": "next_round",
                     "description": "生成器学习率 (轮次边界热更新)"},
    "lambda_sem": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0,
                   "group": "generator", "hot_update_policy": "round_boundary",
                   "requires_restart": False, "effective_stage": "next_round",
                   "description": "语义损失权重 (轮次边界热更新)"},
    "lambda_disagreement": {"type": "float", "default": 0.1, "min": 0.0, "max": 100.0,
                            "group": "generator", "hot_update_policy": "round_boundary",
                            "requires_restart": False, "effective_stage": "next_round",
                            "description": "分歧损失权重 (轮次边界热更新)"},
    "lambda_diversity": {"type": "float", "default": 0.1, "min": 0.0, "max": 100.0,
                         "group": "generator", "hot_update_policy": "round_boundary",
                         "requires_restart": False, "effective_stage": "next_round",
                         "description": "多样性损失权重 (轮次边界热更新)"},
    "lambda_feature_norm": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0,
                            "group": "generator", "hot_update_policy": "round_boundary",
                            "requires_restart": False, "effective_stage": "next_round",
                            "description": "特征范数权重 (轮次边界热更新)"},
    "knn_k": {"type": "int", "default": 5, "min": 1, "max": 1000,
              "group": "generator", "hot_update_policy": "round_boundary",
              "requires_restart": False, "effective_stage": "next_round",
              "description": "KNN 伪图近邻数 (轮次边界热更新)"},

    # ---- 蒸馏参数 ----
    "distill_steps": {"type": "int", "default": 5, "min": 1, "max": 1000,
                      "group": "distillation", "hot_update_policy": "round_boundary",
                      "requires_restart": False, "effective_stage": "next_round",
                      "description": "蒸馏步数 (canonical 别名 distillation_steps)"},
    "distillation_steps": {"type": "int", "default": None, "min": 1, "max": 1000,
                           "group": "distillation", "hot_update_policy": "round_boundary",
                           "requires_restart": False, "effective_stage": "next_round",
                           "description": "蒸馏步数 (canonical 名)"},
    "distill_lr": {"type": "float", "default": 1e-3, "min": 1e-6, "max": 10.0,
                   "group": "distillation", "hot_update_policy": "round_boundary",
                   "requires_restart": False, "effective_stage": "next_round",
                   "description": "蒸馏学习率 (轮次边界热更新)"},
    "distill_temperature": {"type": "float", "default": 1.0, "min": 0.01, "max": 10.0,
                            "group": "distillation", "hot_update_policy": "start",
                            "requires_restart": True, "effective_stage": "before_task",
                            "description": "蒸馏温度"},

    # ---- checkpoint / 任务参数 ----
    "checkpoint_dir": {"type": "str", "default": "", "group": "task",
                       "hot_update_policy": "start", "requires_restart": True,
                       "effective_stage": "before_task",
                       "description": "checkpoint 保存目录 (平台默认落盘到任务目录)"},
    "resume_checkpoint": {"type": "str", "default": "", "group": "task",
                          "hot_update_policy": "start", "requires_restart": True,
                          "effective_stage": "before_task",
                          "description": "从 checkpoint 恢复训练"},
    "save_last_checkpoint": {"type": "bool", "default": True, "group": "task",
                             "hot_update_policy": "start", "requires_restart": True,
                             "effective_stage": "before_task",
                             "description": "每轮保存 last.pt"},
    "selection_metric": {"type": "str", "default": "",
                         "choices": ["", "macro_f1", "pooled_pr_auc", "accuracy"],
                         "group": "task", "hot_update_policy": "start",
                         "requires_restart": True, "effective_stage": "before_task",
                         "description": "模型选择指标 (空=按任务模式默认)"},
    "f1_threshold": {"type": "float", "default": -1e6, "min": -1e6, "max": 100.0,
                     "group": "task", "hot_update_policy": "start",
                     "requires_restart": True, "effective_stage": "before_task",
                     "description": "双重终止第一条件: 主指标连续三轮提升 < 该值则终止。"
                                    "默认 -1e6 = **关闭**(条件永不成立)。"
                                    "原因: 论文协议为固定 100 轮, 且全部自跑实验均跑满 100 轮"
                                    "(federated_mode='fedavg' 时该早停代码是激活的, 收敛后连续 3 轮"
                                    "增幅 <0.1% 即 break, 约 20~30 轮就会提前结束, 与论文不可比)。"
                                    "需启用时设为 0.1 等正值"},
    "auc_threshold": {"type": "float", "default": -1e6, "min": -1e6, "max": 100.0,
                      "group": "task", "hot_update_policy": "start",
                      "requires_restart": True, "effective_stage": "before_task",
                      "description": "双重终止第二条件: 低资源客户端指标连续两轮增幅 < 该值则终止。"
                                     "默认 -1e6 = **关闭**, 理由同上。需启用时设为 1.0 等正值"},
    "allow_anomaly_majority": {"type": "bool", "default": False, "group": "task",
                               "hot_update_policy": "start", "requires_restart": True,
                               "effective_stage": "before_task",
                               "description": "允许异常为多数类"},
    "resplit_stratified": {"type": "bool", "default": None, "group": "task",
                           "hot_update_policy": "start", "requires_restart": True,
                           "effective_stage": "before_task",
                           "description": "标签映射后分层重划分。⚠️ 本平台默认 task_mode="
                                          "anomaly_binary 时为 True; 切到 multiclass 时训练端会"
                                          "自动置 False(2026-09-21 实测: 重切分会丢弃缓存的作者划分, "
                                          "使 Cora-10 FedAvg 从 74.2 虚高到 77.2、跨种子 std 从 ±0.3 "
                                          "放大到 ±1.5, 数字不可与论文对照)"},
    # ---- 蒸馏/扩散新增 (2026-09-21 算法主线更新后同步) ----
    "distill_loss_type": {"type": "str", "default": "kl", "choices": ["kl", "l1"],
                          "group": "distillation", "hot_update_policy": "start",
                          "requires_restart": True, "effective_stage": "before_task",
                          "description": "蒸馏/分歧损失形式。kl = 本方法设计(专利权6 / 专利书 S4.2, "
                                         "亦与 FedTAD 论文 Eq.10 一致); l1 = 上游参考代码的写法。"
                                         "二者梯度性质不同, 非等价变形"},
    "federated_diffusion_pretrain": {"type": "bool", "default": True,
                                     "group": "distillation",
                                     "hot_update_policy": "start",
                                     "requires_restart": True,
                                     "effective_stage": "before_task",
                                     "description": "联邦扩散预训练 (专利 S3.1/S3.2 + 专利权1)。"
                                                    "各客户端本地用自己的真实特征做前向加噪并训练噪声预测, "
                                                    "只上传去噪网络参数, 原始特征不出域"},
    "diffusion_pretrain_rounds": {"type": "int", "default": 10, "min": 1, "max": 200,
                                  "group": "distillation", "hot_update_policy": "start",
                                  "requires_restart": True, "effective_stage": "before_task",
                                  "description": "联邦扩散预训练的联邦轮数"},
    "diffusion_pretrain_epochs": {"type": "int", "default": 30, "min": 1, "max": 500,
                                  "group": "distillation", "hot_update_policy": "start",
                                  "requires_restart": True, "effective_stage": "before_task",
                                  "description": "扩散预训练每客户端本地 epoch 数"},
    "diffusion_pretrain_batch": {"type": "int", "default": 256, "min": 2, "max": 8192,
                                 "group": "distillation", "hot_update_policy": "start",
                                 "requires_restart": True, "effective_stage": "before_task",
                                 "description": "扩散预训练批大小"},
    "diffusion_pretrain_lr": {"type": "float", "default": 1e-3, "min": 1e-6, "max": 1.0,
                              "group": "distillation", "hot_update_policy": "start",
                              "requires_restart": True, "effective_stage": "before_task",
                              "description": "扩散预训练学习率"},
    "feature_stats_align": {"type": "bool", "default": False,
                            "group": "distillation", "hot_update_policy": "start",
                            "requires_restart": True, "effective_stage": "before_task",
                            "description": "[实验性] 特征统计特性对齐。客户端上传统计量(非原始特征), "
                                           "服务端把伪特征重标定到真实统计特性。实测能精确对齐 σ "
                                           "但会令生成器梯度塌缩, 默认关闭"},
}

# 平台内部字段 (不进入训练命令行)
INTERNAL_FIELDS = {"name", "description"}

# canonical 别名 -> 主键 (规范快照只保留主键)
ALIAS_TO_PRIMARY = {
    "federated_rounds": "num_rounds",
    "learning_rate": "lr",
    "fake_node_count": "fake_nodes",
    "distillation_steps": "distill_steps",
}

# 热更新白名单 (与 train_fedtad.py HOT_UPDATABLE_PARAMS 保持一致)
HOT_UPDATABLE = {k: v for k, v in CANONICAL_SCHEMA.items()
                 if v["hot_update_policy"] == "round_boundary"}


# ---------------------------------------------------------------------------
#  校验与规范化
# ---------------------------------------------------------------------------
def _coerce(meta: dict, value):
    vtype = meta["type"]
    if vtype == "int":
        return int(value)
    if vtype == "float":
        return float(value)
    if vtype == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        raise ValueError(f"not a boolean: {value!r}")
    return str(value)


def validate_and_normalize(params: dict) -> tuple[dict, list[str]]:
    """校验用户参数并生成规范化 canonical config (含默认值补全)。

    返回 (canonical_config, errors)。errors 非空时调用方应拒绝任务创建。
    未知参数一律报错, 不静默忽略, 不进入命令行。
    """
    if not isinstance(params, dict):
        return {}, ["parameters must be a dict"]

    errors: list[str] = []
    normalized: dict = {}
    unknown = sorted(set(params) - set(CANONICAL_SCHEMA) - INTERNAL_FIELDS)
    for key in unknown:
        errors.append(f"unsupported parameter: {key}")

    for key, meta in CANONICAL_SCHEMA.items():
        if key not in params or params[key] is None or params[key] == "":
            continue
        value = params[key]
        try:
            coerced = _coerce(meta, value)
        except (TypeError, ValueError) as e:
            errors.append(f"{key}: invalid value {value!r} ({e})")
            continue
        # 范围校验
        if "min" in meta and coerced < meta["min"]:
            errors.append(f"{key}: {coerced} < min {meta['min']}")
            continue
        if "max" in meta and coerced > meta["max"]:
            errors.append(f"{key}: {coerced} > max {meta['max']}")
            continue
        # choices 校验
        if "choices" in meta and coerced not in meta["choices"]:
            errors.append(f"{key}: {coerced!r} not in choices {meta['choices']}")
            continue
        normalized[key] = coerced

    # 组合校验
    tm = normalized.get("task_mode", CANONICAL_SCHEMA["task_mode"]["default"])
    if tm == "anomaly_binary":
        nc = normalized.get("normal_classes",
                            CANONICAL_SCHEMA["normal_classes"]["default"])
        ac = normalized.get("anomaly_classes",
                            CANONICAL_SCHEMA["anomaly_classes"]["default"])
        if not str(nc).strip() or not str(ac).strip():
            errors.append("anomaly_binary 模式必须提供 normal_classes 与 anomaly_classes")
    knn = normalized.get("knn_k", CANONICAL_SCHEMA["knn_k"]["default"])
    fake = (normalized.get("fake_node_count") or normalized.get("fake_nodes")
            or CANONICAL_SCHEMA["fake_node_count"]["default"])
    if knn >= fake:
        errors.append(f"knn_k ({knn}) 必须小于伪节点数 ({fake})")

    # 默认值补全 (未提供字段使用 schema 默认; 别名成对时取已提供者)
    for key, meta in CANONICAL_SCHEMA.items():
        if key in normalized:
            continue
        default = meta["default"]
        if default is not None:
            normalized[key] = default

    # 别名归一化: canonical 名优先, 旧名回退 (规范快照只保留主键)
    if "federated_rounds" in normalized:
        normalized["num_rounds"] = normalized["federated_rounds"]
        del normalized["federated_rounds"]
    if "learning_rate" in normalized:
        normalized["lr"] = normalized["learning_rate"]
        del normalized["learning_rate"]
    if "fake_node_count" in normalized:
        normalized["fake_nodes"] = normalized["fake_node_count"]
        del normalized["fake_node_count"]
    if "distillation_steps" in normalized:
        normalized["distill_steps"] = normalized["distillation_steps"]
        del normalized["distillation_steps"]

    return normalized, errors


# 进入训练命令行的字段 (排除别名重复与纯别名键)
COMMAND_FIELDS = [
    "dataset", "task_mode", "normal_classes", "anomaly_classes",
    "num_clients", "num_rounds", "num_epochs", "local_epochs",
    "seed", "model_seed", "partition_seed", "split_seed", "gpu_id",
    "root", "partition", "part_delta",
    "hid_dim", "dropout", "lr", "weight_decay",
    "use_weighted_ce", "class_weight_method", "beta",
    "contrastive_mode", "contrastive_anchor_scope", "contrastive_batch_size",
    "contrastive_temperature", "lambda_subgraph", "edge_perturb_ratio",
    "rwr_restart_prob", "rwr_subgraph_size", "rwr_cache",
    "ckr_mode", "static_ckr_scaling", "dynamic_ckr_metric", "dynamic_ckr_alpha",
    "dynamic_ckr_ema_decay", "reliability_holdout_ratio", "reliability_min_support",
    "reliability_split_seed", "distill_weighting",
    "generator_init", "diffusion_steps", "diffusion_hidden",
    "diffusion_beta_start", "diffusion_beta_end", "generator_output_bound",
    "generator_backprop_mode", "generator_truncate_interval",
    "fake_nodes", "fake_class_strategy", "generator_warmup_rounds",
    "generator_steps", "generator_lr", "lambda_sem", "lambda_disagreement",
    "lambda_diversity", "lambda_feature_norm", "knn_k",
    "distill_steps", "distill_lr", "distill_temperature", "distill_loss_type",
    "federated_diffusion_pretrain", "diffusion_pretrain_rounds",
    "diffusion_pretrain_epochs", "diffusion_pretrain_batch",
    "diffusion_pretrain_lr", "feature_stats_align",
    "checkpoint_dir", "resume_checkpoint", "save_last_checkpoint",
    "selection_metric", "f1_threshold", "auc_threshold",
    "allow_anomaly_majority", "resplit_stratified",
]

# BooleanOptionalAction 参数 (False 时传 --no- 形式)
BOOLEAN_OPTIONAL = {"use_weighted_ce", "save_last_checkpoint", "resplit_stratified",
                    "federated_diffusion_pretrain", "feature_stats_align"}
