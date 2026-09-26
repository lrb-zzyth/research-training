# CLEANUP_AUDIT.md — 清理前仓库审计

> 生成时间：2026-08-02
> 审计方式：4 个并行探索代理全局扫描 + 逐一 grep 验证（Python/JS import、API URL、路由注册、CLI 参数、配置文件、shell/Docker、测试引用、文件名字符串）。
> 主线定义：**A = 核心专利算法**，**B = 现有前后端平台**。

---

## 一、总体结论

1. **核心算法文件**（train_fedtad.py 2570 行 / model.py 278 行 / util/* 11 个模块）结构完整，客户端训练 → FedAvg → 生成器训练 → KNN 伪图 → CKR 融合蒸馏的调用链全部存在，**生成器阶段与全局蒸馏阶段的参数冻结/detach 分离符合专利流程**。
2. **平台前后端 API 自洽**（10 个 REST/WS 端点全部存在），但**与训练脚本已完全脱节**：
   - 后端 `param_to_flag` 映射的 37 个 CLI flag 中有 **16 个在 train_fedtad.py 中不存在**（`--use_contrastive`、`--lambda_cl`、`--cl_tau`、`--subgraph_size`、`--edge_drop_rate`、`--restart_prob`、`--glb_epochs`、`--it_g`、`--it_d`、`--lr_g`、`--lr_d`、`--fedtad_mode`、`--num_gen`、`--lam1`、`--lam2`、`--topk`）→ **用默认表单点"开始训练"必然 argparse 报错、任务 failed**。
   - `log_parser.py` 的 `[S5] val_f1=...` 等正则与当前输出格式（`[Global] pooled_pr_auc=...` / `[Round N]`）完全不匹配 → **指标表、best_val/best_test 列、前端图表永远为空**。
3. **默认配置偏离核心专利**：`task_mode=multiclass`（应 anomaly_binary）、`ckr_mode=hybrid_dynamic`（应 static_topology）、`distill_weighting=dynamic_ckr`（B4 应为 static_ckr），且**无 "Core method" 声明日志**。
4. **扩展层清晰可分离**：run_experiments.py（1894 行，独立大型实验编排系统，仅以 subprocess + JSON 产物与主线耦合）、阶段二/三测试、结论矩阵脚本均不进入核心调用链。
5. **死代码**：`util/gradate_contrastive.py`（416 行，全仓库零引用且与 task_util.py 重复）、`ParameterForm.vue`（全仓库零引用且参数命名已过时）。

---

## 二、逐文件审计表

### 2.1 仓库根目录

| 文件 | 当前作用 | 所属主线 | 被谁调用 | 建议处理 | 理由 | 删除风险 | 影响前端 | 影响后端 | 影响算法 | 影响历史数据 |
|---|---|---|---|---|---|---|---|---|---|---|
| train_fedtad.py | 核心训练主脚本（S1 静态 CKR → S2 本地训练 → FedAvg → 生成器 → KNN 伪图 → CKR 蒸馏） | A | 平台 routers/training.py 子进程；test_all/smoke/smoke_train；run_experiments（subprocess） | **保留 + 改默认值** | 核心专利唯一正式入口 | — | 需要同步修正平台 flag 映射 | 无 | 默认值改为 static 主线（见 §三） | 需保持既有 CLI 参数名（num_rounds/fake_nodes/distill_steps 等），不删参数 |
| model.py | GCN（共享编码器）+ ConditionalDiffusionGenerator（教师引导扩散生成器，别名 TeacherGuidedDiffusionGenerator） | A | train_fedtad.py / test_all / test_smoke / pretrain_diffusion / benchmark_checkpointing | **保留** | 核心模型定义 | — | 否 | 否 | 否 | 否 |
| train_fedavg.py | 旧版独立 FedAvg 基线脚本（无生成器/蒸馏，硬编码旧绝对路径 `/home/ai2/work/fedtad/dataset`，含已废弃参数 glb_epochs/it_g/it_d/lr_g/lr_d/fedtad_mode/num_gen/lam1/lam2/topk/num_dims） | A（历史基线） | 仅 README.md:315,325 文档引用 | **移入 legacy/** | B1 已可由 train_fedtad.py `--distill_weighting none` 复现，本文件是被正式实现替代的旧入口，有历史复现价值故不删除 | 低（README 同步更新即可） | 否 | 否 | 否（B1 等价实现保留在 train_fedtad.py） | 否 |
| pretrain_diffusion.py | 公开代理数据 DDPM 预训练 + FeatureAdapter + load_proxy_pretrained_generator | A（可选路径） | train_fedtad.py:72（模块级导入）；test_all Test 52/53；test_smoke_train Smoke D；run_experiments --prepare-proxy | **保留**（默认关闭） | 核心脚本模块级导入，`--generator_init` 的可选值；默认 scratch 不使用 | 低 | 否 | 否 | 删除会破坏 train_fedtad.py 导入 | 代理 checkpoint（runs/assets/proxy_diffusion.pt）为扩展产物 |
| run_experiments.py | 1894 行独立大型实验编排系统（SUITES：阶段一 13 个 + 阶段二 7 个 + 阶段三 5 个；含 D0-D8 因果链、fairness、dynamic CKR 网格、partition-cluster bootstrap、统计汇总/图表） | 扩展 | test_experiments/test_phase2/test_phase3 导入；自身 subprocess 调 train_fedtad.py | **移入 experimental/** | 大型统计审计系统，偏离核心专利，默认训练入口不调用；有历史复现价值 | 中（ROOT/sys.path 需修正；runs/ 相对路径依赖） | 否 | 否 | 否 | runs/ 历史产物保留原位 |
| benchmark_checkpointing.py | 生成器 full vs checkpointed 反向传播显存/时间基准 | 扩展 | run_experiments（checkpoint_memory_scaling）；test_experiments/test_phase2/test_phase3 | **移入 experimental/** | 仅扩展层使用 | 低 | 否 | 否 | 否 | 否 |
| generate_conclusion_matrix.py | 规则式"自动结论判断"系统（读 runs/ CSV 生成 conclusion_matrix.csv） | 扩展 | 仅 PHASE3_EXPERIMENT_REPORT.md 提及 | **移入 experimental/** | 自动结论判断系统（禁止项），无代码引用 | 低 | 否 | 否 | 否 | 否 |
| test_all.py | 59 项核心单元测试（CKR/对比/生成器/蒸馏/checkpoint/RWR 缓存/代理 DDPM） | A | 手动运行 `python test_all.py` | **保留** | 核心算法必要测试 | — | 否 | 否 | 否 | 否 |
| test_smoke.py | 26 项核心冒烟（梯度隔离/隐私边界/标签映射/RWR/InfoNCE 数值） | A | 手动运行 | **保留** | 核心算法必要测试 | — | 否 | 否 | 否 | 否 |
| test_smoke_train.py | 4 组真实训练冒烟（multiclass+动态CKR+断点续训 / anomaly_binary / static CKR 消融 / 代理预训练） | A（含扩展子测试） | 手动运行 | **保留** | 覆盖真实端到端训练链路，含静态 CKR 消融 | — | 否 | 否 | 否 | 否 |
| test_experiments.py | 10 项扩展层测试（suite 参数合法性、多种子确定性、统计数学、失败状态） | 扩展 | 手动运行 | **移入 experimental/** | 测试对象是 run_experiments 扩展系统 | 中（import run_experiments/experiment_stats 需修） | 否 | 否 | 否 | 写入 runs/_t_* scratch |
| test_phase2.py | 25 项阶段二扩展测试（动态 CKR 记录、10-seed 聚合、cache 语义、suite 计划数） | 扩展 | 手动运行 | **移入 experimental/** | 测试对象是阶段二扩展 | 中 | 否 | 否 | 否 | 写入 runs/_t_* scratch |
| test_phase3.py | 32 项阶段三扩展测试（frozen split、D0-D8、fairness 只读 val、bootstrap 数学） | 扩展 | 手动运行 | **移入 experimental/** | 测试对象是阶段三扩展 | 中 | 否 | 否 | 否 | 写入 runs/_t_* scratch |
| README.md | 项目文档 | A/B | — | **保留 + 修正过时命令**（§8 中 `--generator_type`/`--use_contrastive`/`--contrastive_type`/`--fedtad_mode` 等均不存在于当前脚本） | 文档与实现必须一致 | — | 否 | 否 | 否 | 否 |
| DELIVERY/EXPERIMENT/PHASE3 报告 | 历史实验报告 | 文档 | — | **保留** | 历史记录，含扩展实验复现命令 | — | 否 | 否 | 否 | 否 |
| 专利申请书_终版.docx / interview_project_introduction.tex | 专利申请书/访谈介绍 | A | — | **保留** | 专利文档 | — | 否 | 否 | 否 | 否 |
| update.sh | 提交推送脚本（含失效绝对路径 /home/ai2/work/fedtad） | 平台运维 | — | **保留并修正路径** | 部署脚本属于平台保留项，修正路径 | 低 | 否 | 否 | 否 | 否 |
| __pycache__/ | Python 字节码缓存 | 垃圾 | — | **删除** | 缓存 | 无 | 否 | 否 | 否 | 否 |
| requirements.txt | **缺失**（README §9 引用了它） | A/B | — | **补建** | 记录实际依赖（torch/pyg/sklearn/networkx） | — | 否 | 否 | 否 | 否 |

### 2.2 util/ 目录

| 模块 | 公开符号 | 作用 | 所属主线 | 被谁调用 | 建议处理 | 理由 | 删除风险 | 影响前端/后端 | 影响算法 | 影响历史数据 |
|---|---|---|---|---|---|---|---|---|---|---|
| task_util.py | cal_topo_emb/edge_perturbation/rwr_subgraph_sampling/subgraph_contrastive_loss/compute_class_weights/multiclass_metrics/binary_anomaly_metrics/DiversityLoss 等 | 核心训练与评估工具（边扰动、RWR 采样、InfoNCE、类别加权、指标） | A | train_fedtad.py / train_fedavg.py / test_all / test_smoke | **保留** | 核心链路最重依赖 | — | 否 | 否 | 否 |
| base_util.py | load_dataset/weight_init/seed_everything | 数据加载与种子封装 | A | train_fedtad / train_fedavg / test_phase3 | **保留** | 核心入口依赖 | — | 否 | 否 | 否 |
| fgl_dataset.py | FGLDataset/DatasetBlockedError | PyG 数据集封装（process 阶段执行 data_partition） | A | util/base_util.py | **保留** | 核心数据链路 | — | 否 | 否 | 否 |
| base_data_util.py | louvain_partition/data_partition/construct_subgraph_dict_from_node_dict/enrich_anomaly_support 等 | Louvain/Metis 拓扑划分 + 客户端子图构造（**已确认不读取标签**，仅用图拓扑） | A | util/fgl_dataset.py；test_phase2 | **保留** | 专利流程第 1 步（Louvain 社区发现） | — | 否 | 否 | 否 |
| data_split.py | stratified_split/stratified_reliability_split/anomaly_holdout_boost_split/split_report | 标签映射后分层重划分（含 min-support） | A | train_fedtad / 4 个测试 | **保留** | 核心数据链路 | — | 否 | 否 | 否 |
| checkpoint.py | save/load_checkpoint/find_*_checkpoint/validate_meta | 断点保存恢复（含 CKR tracker 序列化） | A | train_fedtad / test_all | **保留** | 历史 checkpoint 兼容依赖 | — | 否 | 否 | 历史 best.pt 加载依赖它 |
| rwr_cache.py | RWRSubgraphCache | RWR 子图采样缓存 | A | train_fedtad / test_all / test_phase2 / test_phase3 | **保留** | 核心训练使用 | — | 否 | 否 | 否 |
| split_artifact.py | data_identity_hash/save_split_artifact/load_artifact/build_subgraphs_from_artifact/initialization_hash | 冻结划分 artifact（数据身份哈希） | A | train_fedtad（惰性导入）；test_phase3 | **保留** | train_fedtad 阶段三 frozen-split 功能 | — | 否 | 否 | frozen artifact 加载依赖 |
| dynamic_ckr.py | DynamicCKRTracker/compute_per_class_reliability_metrics/scale_static_ckr | 动态 CKR 跟踪器（静态先验 + 动态指标 + EMA 混合） | A（静态部分）/ 扩展（动态部分） | train_fedtad（tracker 初始化与静态权重路径也用它）；test_all/test_phase2/test_phase3 | **保留，默认关闭** | 静态路径（tracker.static_ckr_scaled）与 checkpoint 序列化依赖它；动态路径默认关闭并标注 experimental，不进入核心默认配置 | 中（若删除需改 checkpoint 兼容与蒸馏权重分支） | 否 | 默认流程不再使用动态更新 | 历史 checkpoint 含 tracker 状态 |
| experiment_stats.py | summarize/paired_diff/paired_bootstrap_ci/holm_correction/cliffs_delta/sensitivity_analysis/blocked_paired_analysis | 扩展层统计审计工具 | 扩展 | run_experiments.py（4 处函数级导入）；test_experiments/test_phase2/test_phase3 | **移入 experimental/** | 核心算法与平台均不引用 | 低（4 个导入点同步改） | 否 | 否 | 否 |
| gradate_contrastive.py | GradateContrastiveModule/AvgReadout/ContextualDiscriminator/PatchDiscriminator | GRADATE 多尺度对比模块 | 扩展 | **全仓库零引用**（grep 验证） | **删除** | 死代码；且 rwr_subgraph_sampling/edge_perturbation/邻接表构建与 task_util.py 重复，正式实现为 task_util 版本 | 无 | 否 | 否 | 否 | 否 |

### 2.3 research-training/backend（全部保留，部分重构）

| 文件 | 作用 | 建议处理 | 理由 | 删除风险 |
|---|---|---|---|---|
| app/main.py | FastAPI 入口 + /api/health + CORS + 静态挂载 | **保留** | 平台入口 | — |
| app/config.py | 环境配置（DATABASE_URL/PROJECT_ROOT/CORS） | **保留** | 平台配置 | — |
| app/database.py / models.py / schemas.py | ORM 模型与请求/响应 schema | **保留** | 任务状态/日志/指标持久化核心 | — |
| app/auth.py + routers/auth.py | JWT 登录注册 | **保留** | 平台鉴权 | — |
| app/routers/training.py | 训练启停/状态/WS | **保留 + 重写 param_to_flag** | 16 个 CLI flag 不存在导致训练必然失败；改为 canonical→CLI 映射 + BooleanOptionalAction 处理 + 未知键跳过 | 低（同步更新前端表单键名） |
| app/training.py | 子进程管理/日志流/指标入库/状态机 | **保留** | 任务管理核心 | — |
| app/websocket_manager.py | WS 广播 | **保留** | 实时进度通信 | — |
| app/log_parser.py | 输出行→指标解析 | **保留 + 重写正则** | 现正则与当前输出格式零匹配，指标/图表/best_* 列永远为空 | 低（新正则按 train_fedtad.py 实际输出编写） |
| requirements.txt | 依赖清单 | **保留** | 平台依赖 | — |
| .env / .env.example | 环境配置 | **保留** | 平台配置（.env gitignored，.env.example 为模板） | — |
| .env.bak | 备份文件 | **删除** | 过期备份，与 .env 内容相同，无引用 | 无 |
| README.md | 平台文档 | **保留** | 启动说明 | — |

### 2.4 research-training/frontend（全部保留，部分重构）

| 文件 | 作用 | 建议处理 | 理由 | 删除风险 |
|---|---|---|---|---|
| 全部 views（Dashboard/Login/Register/TrainingConfig/TrainingMonitor/ExperimentHistory/Settings） | 页面 | **保留** | 平台页面 | — |
| components/ExperimentTable.vue / LogConsole.vue / MetricChart.vue | 表格/日志/图表 | **保留** | 结果展示核心 | — |
| components/ParameterForm.vue | 训练参数表单 | **删除** | 全仓库零 import 引用（死组件）；参数命名（use_contrastive/cl_tau/topk 等）已过时，正式表单在 TrainingConfig.vue 内 | 无 |
| api/index.js | API 客户端（10 个函数） | **保留** | 接口层；getExperimentLogs 当前无调用但保留（历史日志回放链路） | — |
| stores/auth.js / settings.js / training.js | Pinia 状态 | **保留** | 平台状态 | — |
| router/index.js / main.js / App.vue / utils/i18n.js | 路由/入口/布局/国际化 | **保留** | 平台骨架 | — |
| package.json / vite.config.js / index.html | 构建配置 | **保留** | 前端构建 | — |
| views/TrainingConfig.vue | 训练配置表单 | **保留 + 修正参数键与核心配置** | 16 个失效键 + 缺失 task_mode/ckr_mode/fake_class_strategy/fake_node_count/knn_k/generator_steps/distillation_steps/各 seed 等核心配置 | 低（键名与后端映射同步更新） |

### 2.5 数据与产物

| 路径 | 作用 | 建议处理 | 理由 |
|---|---|---|---|
| dataset/Cora/{raw,processed} | 原始+预处理全局图 | **保留** | 核心数据 |
| dataset/Cora/Client10/Louvain/、Client2/Louvain/ | 核心 Louvain 子图（data0-9.pt / data0-1.pt） | **保留** | 核心数据（默认训练使用） |
| dataset/Cora/Client10/{Louvain_enriched, Louvain_frozen_*, Louvain_enriched_frozen_*}、Client2/Louvain_enriched* | 阶段三富集/冻结划分产物（partition_moves.json/node_assignments.json） | **保留原位**（隔离语义） | 扩展层（run_experiments 阶段三 suite）实验复现数据；默认核心流程不触碰；避免大文件移动风险 |
| ckr/*.pt | 静态 CKR 磁盘缓存（gitignored） | **保留** | 训练缓存，加速重复运行 |
| runs/ | 扩展实验输出（gitignored，47 个目录含 _t_* 测试 scratch） | **保留** | 历史实验结果与复现依据 |
| louvain/ | vendored python-louvain 社区发现库 | **保留** | 核心依赖（Louvain 划分） |
| references/ | 外部参考代码（gitignored） | **保留** | 外部参考，不进入构建 |

---

## 三、核心算法需恢复的默认配置（阶段 C）

| 参数 | 当前默认 | 目标默认 | 说明 |
|---|---|---|---|
| --task_mode | multiclass | **anomaly_binary** | 正常=0 异常=1 主模式；multiclass 仅作兼容 |
| --normal_classes / --anomaly_classes | ''（必填） | **'0,1,2,3' / '4,5,6'** | 默认映射，文档注明 |
| --ckr_mode | hybrid_dynamic | **static_topology** | 静态拓扑 CKR 为主流程 |
| --distill_weighting | dynamic_ckr | **static_ckr** | B4 完整方法；dynamic_ckr 标注 experimental |
| --generator_init | teacher_guided_scratch | **scratch**（保留旧值为别名） | 等价 public_pretraining=false |
| --fairness_mode | none | none（不变） | 公平性扩展关闭 |
| --contrastive_mode | subgraph_cross_view | 不变 | 子图-子图跨视图对比 |
| --contrastive_anchor_scope | all_nodes | 不变 | 全节点锚点 |
| --fake_class_strategy | balanced | 不变 | 平衡类别伪标签 |
| --use_weighted_ce | True | 不变 | 类别加权交叉熵 |
| KNN k | 硬编码 5 | **新增 --knn_k 默认 5** | 平台配置项 |
| 日志 | 无 Core method 声明 | **新增声明块** | 输出核心方法 + 运行状态 |
| 最终打印 | 无 best_test | **补 best_test** | 平台 best_test 列/图表依赖 |

---

## 四、删除/隔离前引用检查记录（均已 grep 验证）

| 目标 | 检查结果 |
|---|---|
| util/gradate_contrastive.py | 全仓库（排除 .git/node_modules/louvain/references）仅自身文件命中；无 import、无字符串引用 |
| ParameterForm.vue | research-training/frontend/src 全目录 grep 无任何 import；未挂载任何 view/路由 |
| .env.bak | 无任何代码引用；内容与 .env 逐字节相同 |
| 根 __pycache__ | gitignored 字节码缓存 |
| train_fedtad.py 中 accuracy/DiversityLoss 导入 | 仅导入行命中，文件内无调用（accuracy 仅作 dict 键字符串） |
| validate_manifest（train_fedtad.py:444） | 被 test_phase3.py:387 导入 → **保留**（随 test 移入 experimental） |
| run_experiments.py 移动 | ROOT=os.path.dirname(__file__) → 移动后需改为仓库根；subprocess 均 cwd=ROOT；util.experiment_stats 函数级导入需修 |
| benchmark_checkpointing.py 移动 | import model → 需加仓库根 sys.path |
| test_experiments/phase2/phase3 移动 | import run_experiments / util.* / train_fedtad → 需加仓库根与 experimental 目录 sys.path |
| dataset enriched/frozen 目录 | 由 run_experiments 阶段三 suite 以 --build_split_artifact/--frozen_split 名称引用（CLI 传参）；无 Python import |

---

## 五、平台接口兼容清单（详见 API_COMPATIBILITY_REPORT.md）

- 保留全部 10 个 REST/WS 端点与响应字段（id/name/status/command/parameters/best_round/best_val/best_test/logs/metrics）。
- 修复 `POST /api/training/start` 的 parameters→CLI 映射（16 个失效键 → 真实参数；新增 canonical 名映射：federated_rounds→num_rounds、learning_rate→lr、fake_node_count→fake_nodes、distillation_steps→distill_steps、knn_k→knn_k、local_epochs→local_epochs）。
- 重写 log_parser 正则以匹配当前输出（[Round N] / [Global] metric=val (best=.. @ round N) / 训练结束. 最佳 round=N, best_val(...)=..., best_test(...)=...），恢复指标入库、best_* 列与前端图表。
- 前端 TrainingConfig.vue 键名与后端映射同步；新增核心配置字段。
- 删除前端无效按钮/字段（use_contrastive 等 16 个失效项），不留指向不存在参数的入口。

---

## 六、不允许修改的保留边界（冻结清单）

1. 核心算法：train_fedtad.py 全部算法函数、model.py、util/{task_util,base_util,fgl_dataset,base_data_util,data_split,checkpoint,rwr_cache,split_artifact,dynamic_ckr}.py、louvain/。
2. 平台：research-training/ 全部（仅 §2.3/§2.4 列出的修复点可动）。
3. 接口：10 个 API 端点、全部响应字段、WS 消息格式。
4. 数据：dataset/Cora/{raw,processed,Client10/Louvain,Client2/Louvain}、ckr/、runs/。
5. 历史兼容：checkpoint 序列化格式（含 tracker）、CLI 参数名（num_rounds/fake_nodes/distill_steps 等不重命名）、静态 CKR 磁盘缓存键。
