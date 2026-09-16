# DELETED_FILES_MANIFEST.md — 删除文件清单

> 生成时间：2026-08-02
> 删除原则：仅删除满足"不属于核心算法 / 不属于前后端平台 / 无任何引用 / 不影响历史数据 / 有正式替代实现"的文件。每项删除前均执行全局 grep 引用检查。

---

## 1. util/gradate_contrastive.py（416 行）

| 项目 | 内容 |
|---|---|
| 删除路径 | util/gradate_contrastive.py |
| 删除原因 | 完全死代码 + 重复实现：全仓库（排除 .git/node_modules/louvain/references）grep 仅自身文件命中；`rwr_subgraph_sampling`、`edge_perturbation`、邻接表构建与 util/task_util.py 三处重复 |
| 删除前引用检查 | Python import：无；动态 import：无；API/路由：无；shell/配置：无；测试：无（test_all/test_smoke/test_smoke_train 均不引用）；文件名字符串：无 |
| 替代实现 | util/task_util.py 的同名函数（带 seed 参数、被 train_fedtad.py 核心引用） |
| 删除后验证 | train_fedtad.py --help 正常；test_all.py 59 项通过；test_smoke.py 26 项通过；仓库 grep 无残留引用 |

## 2. research-training/frontend/src/components/ParameterForm.vue

| 项目 | 内容 |
|---|---|
| 删除路径 | research-training/frontend/src/components/ParameterForm.vue |
| 删除原因 | 零引用死组件：frontend/src 全目录 grep 无任何 import；未挂载任何 view/路由；参数命名（use_contrastive/cl_tau/topk 等）已过时（指向不存在的 CLI 参数） |
| 删除前引用检查 | Vue import：无；JS import：无；动态 import：无；路由注册：无 |
| 替代实现 | TrainingConfig.vue 内联表单（已同步修正为当前核心参数） |
| 删除后验证 | npm run build 成功（无缺失模块错误）；grep "ParameterForm" 无残留 |

## 3. research-training/backend/.env.bak

| 项目 | 内容 |
|---|---|
| 删除路径 | research-training/backend/.env.bak |
| 删除原因 | 过期备份文件；与 .env 逐字节相同；无任何代码/脚本引用 |
| 删除前引用检查 | grep 全仓库：无引用 |
| 替代实现 | .env（正式配置，gitignored）+ .env.example（模板） |
| 删除后验证 | 后端 smoke test 38 项通过（服务正常启动） |

## 4. /__pycache__/（根目录字节码缓存）

| 项目 | 内容 |
|---|---|
| 删除路径 | ./__pycache__/ |
| 删除原因 | Python 字节码缓存（gitignored，`*.py[cod]`），运行时自动重建 |
| 删除前引用检查 | 无需检查（编译器产物） |
| 替代实现 | 无（自动重建） |
| 删除后验证 | 全部测试通过（缓存已按需重建） |

---

## 5. 移动而非删除（git 视角下的删除，实际已迁移）

| 原路径 | 新路径 | 原因 | 迁移后验证 |
|---|---|---|---|
| train_fedavg.py | legacy/train_fedavg.py | 旧独立 FedAvg 基线被 train_fedtad.py B1 正式替代，保留历史复现价值 | sys.path 修复后 parse 通过；README 更新 |
| run_experiments.py | experimental/run_experiments.py | 大型实验编排系统（扩展层） | ROOT/sys.path 修复；test_experiments/phase2/phase3 从新位置全部通过（10/25/32 项） |
| util/experiment_stats.py | experimental/experiment_stats.py | 仅扩展层使用的统计审计工具 | 5 处导入点修复；扩展测试全通过 |
| benchmark_checkpointing.py | experimental/benchmark_checkpointing.py | 仅扩展层使用的基准工具 | sys.path 修复；test_phase2/phase3 回归通过 |
| generate_conclusion_matrix.py | experimental/generate_conclusion_matrix.py | 自动结论判断系统（扩展） | 无仓库内依赖，未改 |
| test_experiments.py | experimental/test_experiments.py | 扩展层测试 | ROOT/sys.path/benchmark 路径修复；10/10 通过 |
| test_phase2.py | experimental/test_phase2.py | 阶段二扩展测试 | 同上；25/25 通过 |
| test_phase3.py | experimental/test_phase3.py | 阶段三扩展测试 | 同上；32/32 通过 |

---

## 6. 代码内删除（非文件级）

| 位置 | 内容 | 删除后验证 |
|---|---|---|
| train_fedtad.py | 未使用 import `accuracy`、`DiversityLoss` | test_all/test_smoke/smoke_train 全通过 |
| research-training/backend/app/routers/training.py | 16 个失效 flag 映射 + 布尔 False 静默忽略缺陷（重写映射表） | backend smoke 38 项通过（4a-4g 转换检查） |
| research-training/backend/app/log_parser.py | 旧 `[S5]`/`[S2]` 正则（零匹配）（重写为当前输出格式） | backend smoke 6d（指标入库）通过 |
| frontend TrainingConfig.vue | 16 个失效表单字段（改为当前核心配置 47 项） | 前端构建通过；47 键与后端映射交叉校验一致 |

---

## 7. 统计

- 文件级删除：3 个文件 + 1 个缓存目录
- 文件级迁移：8 个文件（experimental/ 7 个 + legacy/ 1 个）
- 代码级删除/重写：4 处（train_fedtad.py 死导入、param_to_flag、log_parser、前端表单）
- 删除后全部测试通过：核心 89 项（59+26+4）、扩展 67 项（10+25+32）、后端 38 项、前端构建 1 项、算法 CLI 冒烟 2 组
