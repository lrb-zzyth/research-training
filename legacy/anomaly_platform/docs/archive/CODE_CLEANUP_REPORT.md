# CODE_CLEANUP_REPORT.md — 清理执行报告

> 生成时间：2026-08-02
> 前置文档：CLEANUP_AUDIT.md（清理前审计）、API_COMPATIBILITY_REPORT.md（接口兼容）、RETAINED_ARCHITECTURE.md（保留架构）、DELETED_FILES_MANIFEST.md（删除清单）。

---

## 1. 修改前偏离主线的内容

| 偏离项 | 位置 | 严重性 |
|---|---|---|
| 默认配置偏离核心专利：`task_mode=multiclass`、`ckr_mode=hybrid_dynamic`、`distill_weighting=dynamic_ckr`（B5 为默认） | train_fedtad.py | 高（默认训练跑的是动态 CKR 扩展） |
| 平台 16 个 CLI flag 不存在 → 默认表单"开始训练"必失败 | research-training/backend/app/routers/training.py | 高（平台核心功能失效） |
| log_parser 正则与当前输出零匹配 → 指标/图表/best_* 列永远为空 | research-training/backend/app/log_parser.py | 高（平台展示失效） |
| 无 "Core method" 声明日志 | train_fedtad.py | 中（规范要求） |
| 前端表单仍用旧专利版参数命名（use_contrastive/lambda_cl/cl_tau/topk 等） | frontend/src/views/TrainingConfig.vue | 高（与后端共同导致必失败） |
| README 文档了不存在参数（--generator_type/--contrastive_type gradate/node_node 等） | README.md | 中（文档与实现脱节） |
| 死代码：gradate_contrastive.py（416 行零引用，与 task_util 重复） | util/ | 低 |
| 死组件：ParameterForm.vue（零引用，参数过时） | frontend/src/components/ | 低 |
| update.sh 硬编码失效绝对路径 /home/ai2/work/fedtad | update.sh | 低 |
| 根目录缺 requirements.txt（README 引用它） | — | 低 |
| KNN k 硬编码 5，无 CLI/平台入口 | train_fedtad.py | 低 |

## 2. 已删除文件

见 DELETED_FILES_MANIFEST.md（4 个文件 + 缓存目录）。

## 3. 已删除函数/类/导入

| 位置 | 内容 | 原因 |
|---|---|---|
| train_fedtad.py L44,47 | 未使用的 import `accuracy`、`DiversityLoss`（grep 验证文件内无调用） | 死代码 |
| 前端表单 16 个失效字段 | use_contrastive/lambda_cl/cl_tau/subgraph_size/edge_drop_rate/restart_prob/glb_epochs/it_g/it_d/lr_g/lr_d/fedtad_mode/num_gen/lam1/lam2/topk | 指向不存在的 CLI 参数 |
| 后端 param_to_flag 旧映射 | 上述 16 个失效映射 + 布尔 False 静默忽略缺陷 | 重写为 canonical→CLI 映射 |

## 4. 已隔离到 experimental/ 或 legacy/ 的内容

| 内容 | 去向 | 迁移修复 |
|---|---|---|
| run_experiments.py（1894 行实验编排系统） | experimental/ | ROOT 改为仓库根（上一级）；4 处 `util.experiment_stats` → `experiment_stats`；benchmark runner 路径改本目录 |
| experiment_stats.py | experimental/ | 5 处导入点同步修改 |
| benchmark_checkpointing.py | experimental/ | 加 `sys.path`（仓库根）与 `import sys` |
| generate_conclusion_matrix.py（自动结论判断系统） | experimental/ | 无仓库内依赖，未改 |
| test_experiments.py / test_phase2.py / test_phase3.py（扩展层测试） | experimental/ | ROOT 改仓库根 + sys.path；`util.experiment_stats` → `experiment_stats`；benchmark 子进程路径改本目录 |
| train_fedavg.py（旧独立 FedAvg 基线） | legacy/ | 加 `sys.path`（仓库根）；README 命令更新 |
| dataset/Cora/Client10/Louvain_enriched*、Louvain_frozen_*、Client2/Louvain_enriched* | 保留原位 | 扩展实验复现数据，核心流程不触碰（避免大文件移动风险） |
| 动态 CKR（distill_weighting=dynamic_ckr / ckr_mode 非 static） | 代码保留、默认关闭 | 帮助文本标注"实验性扩展"；平台下拉中标注；不属于 B1-B4 |

## 5. 因前后端依赖而保留的内容

- 全部 10 个 REST/WS 端点与全部响应字段（含前端未展示的 description/command/parameters/created_by/created_at）；
- `GET /api/training/status`、`GET /api/experiments/{id}/logs`（当前无 view 调用，属状态查询/日志回放能力，保留）；
- `getExperimentLogs` API 客户端函数；
- 历史实验记录中旧参数键的 JSON 原样存储（只读展示，无破坏）；
- `--teacher_guided_scratch` 作为 `--generator_init` 兼容别名保留（experimental/run_experiments.py 仍使用）；
- `validate_manifest` 函数保留（experimental/test_phase3.py 引用）；
- ckr/ 缓存、runs/ 输出、enriched/frozen 数据集保留。

## 6. 合并的重复实现

| 重复对 | 处理 |
|---|---|
| util/gradate_contrastive.py vs util/task_util.py（rwr_subgraph_sampling / edge_perturbation / 邻接表构建各两份） | 删除 gradate_contrastive.py；正式实现为 task_util.py（带 seed 参数、被核心引用） |
| 前端 ParameterForm.vue vs TrainingConfig.vue 内联表单 | 删除 ParameterForm.vue（零引用）；保留 TrainingConfig.vue |

## 7. 修改后的目录结构

```
FedTAD/
├── train_fedtad.py              # 核心训练主脚本（B1-B4 默认 B4）
├── model.py                     # GCN + ConditionalDiffusionGenerator
├── pretrain_diffusion.py        # 公开代理预训练（默认关闭，实验性）
├── requirements.txt             # 新增：核心算法依赖
├── test_all.py                  # 59 项核心单元测试
├── test_smoke.py                # 26 项核心冒烟
├── test_smoke_train.py          # 4 组真实训练冒烟
├── util/                        # 核心工具（11 模块 → 10 模块）
├── research-training/
│   ├── backend/                 # FastAPI 平台（+ test_backend_smoke.py 38 项）
│   └── frontend/                # Vue 3 平台
├── experimental/                # 扩展层（7 个文件，默认不运行）
├── legacy/                      # train_fedavg.py
├── louvain/  dataset/  ckr/  runs/  references/
├── *.md（README + 审计/架构/接口/清理/删除清单报告）
├── 专利申请书_终版.docx
└── interview_project_introduction.tex
```

## 8. 核心算法完整调用路径

见 RETAINED_ARCHITECTURE.md §二。

## 9. 前端启动方式

```bash
cd research-training/frontend
npm install        # 依赖已安装
npm run dev        # 开发模式 http://localhost:5173（/api 代理到 :8000）
npm run build      # 生产构建（已验证通过）
```

## 10. 后端启动方式

```bash
cd research-training/backend
createdb fedtad    # 首次（或使用现有 fedtad 库）
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 默认管理员: admin / admin123（auth.py 自动创建）
```

## 11. 前后端联合运行方式

1. 启动后端（§10）；
2. 启动前端（§9 dev 或 build + preview）；
3. 浏览器打开 http://localhost:5173 → 登录 → 训练配置页 → 开始训练；
4. 训练监控页实时显示日志（WS）、指标曲线（global_val/best_val/best_test）与状态；
5. 实验历史页可查看 best_round/best_val/best_test。

## 12. API 测试结果

`research-training/backend/test_backend_smoke.py`：**38 通过 / 0 失败**（服务启动、鉴权、参数转换、任务创建、状态查询、完成链路、指标入库、失败路径、日志/指标接口、响应字段完整性）。

## 13. 前端构建结果

`npm run build` 成功（6.94s，仅有 chunk 体积提示，无错误无缺失模块）。

## 14. 后端测试结果

同上 §12（后端平台 smoke test 即后端测试集）。

## 15. 算法测试结果

| 测试 | 结果 |
|---|---|
| test_all.py（59 项单元） | ALL TESTS PASSED ✓ |
| test_smoke.py（26 项冒烟） | ALL TESTS PASSED ✓ |
| test_smoke_train.py（4 组真实训练） | A/B/C/D 全部 PASSED ✓ |
| experimental/test_experiments.py | ALL 10 PASSED ✓ |
| experimental/test_phase2.py | ALL 25 PASSED ✓ |
| experimental/test_phase3.py | ALL 32 PASSED ✓ |

## 16. smoke test 结果（阶段 G）

| 项目 | 配置 | 结果 |
|---|---|---|
| B1 算法最小实验 | 2 客户端 / 2 轮 / 1 epoch / anomaly_binary / 无对比无蒸馏 | ✓ exit 0，test pr_auc=79.19，全部指标有限 |
| B4 算法最小实验 | 默认核心配置（2 客户端 / 2 轮 / 1 epoch / 静态 CKR / 蒸馏） | ✓ exit 0，test pr_auc=89.10，L_D 有限 |
| 后端任务创建与查询 | 经 API 创建 B1 型任务 → finished；指标入库；失败路径 failed | ✓（backend smoke 38 项） |
| 前端生产构建 | npm run build | ✓ |
| 前后端联合流程 | API 全链路（登录→建任务→状态→日志→指标→失败） | ✓（浏览器不可用，见 §18） |

## 17. 仍存在但暂未安全删除的内容

| 内容 | 原因 |
|---|---|
| 动态 CKR 扩展代码（util/dynamic_ckr.py + train_fedtad.py 动态分支 + 相关参数） | 有历史复现价值（阶段二/三实验与 checkpoint 兼容依赖），采用"默认关闭 + 标注实验性"隔离；彻底移除会破坏 checkpoint 序列化兼容与 3 个扩展测试 |
| runs/ 下 47 个实验输出目录（含 _t_* 测试 scratch） | gitignored 历史产物，保留供复现；如需清理可删除 `runs/_t_*` 测试残留 |
| dataset/Cora/Client10/{Louvain_enriched*, Louvain_frozen_*}、Client2/Louvain_enriched* | 扩展实验复现数据 |
| pretrain_diffusion.py | 核心脚本模块级导入（可选路径），默认关闭 |
| 历史报告 .md（EXPERIMENT_REPORT*.md 等） | 历史记录文档 |
| 平台未使用但保留的 API（getExperimentLogs/checkStatus/health） | 能力保留原则 |

## 18. 当前真实限制

1. **未完成真实浏览器验证**：本环境为无头 WSL（无图形界面/浏览器），前后端联合验证采用"后端真实服务 + API 全链路 + 前端生产构建"替代（规范允许的方式）。前端页面渲染未做浏览器级验证。
2. **平台训练不落盘模型文件**：train_fedtad.py 的 checkpoint_dir 等参数后端默认不传（落盘路径参数默认关闭，属平台既有设计）；模型指标经 PostgreSQL 与日志可查询。
3. **单进程联邦模拟**：服务端与客户端在同一进程共享内存（模拟联邦，非真实跨进程通信）；隐私边界在生成器/蒸馏损失函数层面成立（不接收客户端 Data），静态 CKR 计算读取子图属模拟环境实现方式。
4. **平台 WebSocket 无鉴权**：既有设计，本次未改动（不在清理范围）。
5. **扩展层测试依赖 GPU/网络策略**：test_phase3 的 external_blocked 用例按设计预期网络失败；扩展测试需在仓库根目录运行 `python experimental/test_*.py`。
