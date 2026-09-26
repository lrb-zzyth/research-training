> **SUPERSEDED:** This historical audit predates the final v3 KL-direction/RWR/projection corrections. See `FINAL_SERVER_READY_AUDIT.md`.

# FedTAD 改进项目独立审计报告（2026-09-26）

## 结论

**当前原始 review package：NOT READY FOR SERVER MIGRATION。**

本轮已对核心训练链、DDPM、radius projection、formal campaign、RNG、test isolation、checkpoint/optimizer、迁移脚本和依赖复现进行静态审计，并对证据明确的 correctness / protocol 问题做最小修复。

**审计修订版：代码层已消除本轮发现的 P0/P1/P2 工程与实验协议 blocker，但仍不能给出“算法机制最终无风险”的 READY 结论。** 原因有二：

1. 上传包没有专利申请书原文和 FedTAD 论文 PDF，只包含二手 audit_docs，无法完成 source-of-truth 级专利/论文逐条核验；
2. 上传包没有 Cora-5 study.db / trial logs / trial_health.json 等原始实验产物，无法对 raw-radius runaway 做真实 gradient attribution，也无法证明 lambda_sem>=0.01 已从机制上消除内部 DDPM/raw trajectory drift。

因此本报告不新增 raw-flow regularizer、不改 anchor、不改 KL/DDPM/CKR/InfoNCE 数学。

## 已确认的重要数学结论

### Radius projection

当前主线：

`x_proj = r_target * z / sqrt(sum(z^2)+eps)`

在正常有限数值范围内，对正标量 a 有 `Proj(a z) ≈ Proj(z)`。合成 float64 autograd 验证：

- `Proj(z)` vs `Proj(10z)` 最大差约 `3e-14`；
- projected loss 对 z 的径向梯度内积约 0；
- `L_norm = mean(x_proj^2)` 在固定 target radius 下梯度范数约 `1e-15`。

所以 radius projection 开启后，不能再把 raw runaway 简化解释为“generator 直接放大 raw amplitude 来提高 projected KL”。更准确的风险是：**projected objective 对径向尺度近似不敏感，但参数更新仍可能让 DDPM 内部/raw reverse trajectory 发生数值漂移，最终进入 float32 溢出区。**

这也意味着当前 `lambda_feature_norm * L_norm` 在 radius 主线下数学上基本是常数项，不能承担 raw-flow 稳定器职责。本轮未擅自删除/替换该项。

## P0/P1/P2 发现与修复

### P0/P1-01：RNG v2 文档声称“server dedicated generator”，代码实际上没有

原代码 Stage2 的 `differentiable_sample()` / final `sample()` 使用全局 torch RNG；`sample_fake_labels()` 的随机分支也使用全局 RNG。客户端 round/client reseed 确实隔离了客户端随机流，但“服务端专用 generator”这一协议声明并未真实实现。

修复：

- 新增 `make_server_rng(device, base_seed, round_id, stream_id)`；
- fake labels、每个 generator step 的 x_T / reverse noise、final distillation sampling 使用独立确定性 stream；
- `differentiable_sample()` 新增 `initial_noise` / `noise_generator`；
- 为避免 `torch.utils.checkpoint` backward 重计算时自定义 Generator 状态已推进导致噪声不一致，reverse noise 先物化并在 forward/backward 重计算中复用；
- 合成测试确认 full vs checkpointed(segments=1/2/3/6) 输出与梯度逐位一致。

影响：这是 RNG/protocol correctness 修复，会改变正式 v2 的随机轨迹。因此旧 code hash `290e25c26aa8` 不再适用于修订版正式战役。

### P2-02：formal campaign 声称严格串行，但代码/脚本默认并发

原始：

- `formal_campaign.py --n_jobs` 默认 2；
- `run_formal_cell.sh` 默认 2；
- `generate_stage1_all.sh` 默认并行 4；
- `launch_all_cells.sh` 会同时 nohup 15 个 cell。

这与单 RTX 4090、严格串行协议直接冲突，并存在 GPU OOM / 多 trial 竞态风险。

修复：formal n_jobs 固定 1，非 1 fail-fast；Stage1 固定 P=1；all-cells 脚本改为顺序执行。

### P2-03：formal protocol hash / resume guard 覆盖不完整

原 `code_hash()` 只 hash `train_fedtad.py/model.py/util/task_util.py/util/checkpoint.py`，**不包含 formal_campaign.py 本身**。因此 health checker、fixed flags、runner 语义发生变化时，旧 study 可能继续 resume。

原 `verify_protocol()` 也只比较少数字段。

修复：code hash 覆盖训练、数据/CKR、checkpoint、formal runner/health；resume 除纯预算字段外逐项比较 protocol。

### P2-04：正式 final seeds 没有“test 只在最后算一次”的路径

原 `--tuning_mode` 调参期 test=0 次，正确；但非 tuning 训练会每轮计算 test。这样正式 3-seed 决赛虽然 checkpoint 仍按 validation 选择，却会在 100 轮中持续暴露 test。

修复：新增 `--final_test_only`：每轮只算 validation，训练结束加载 val-best checkpoint 后 test 只算一次；新增 `formal_final_eval.py` 和 `scripts/run_formal_final.sh`。

### P1/P2-05：formal health v2 声称检查参数/梯度/raw output nonfinite，但主要依赖日志正则

原 health checker 能抓很多日志中的 `nan/inf/raw_r=inf`，但没有在训练进程内部系统性验证参数、梯度、raw/projected feature。

修复：正式 guard 下新增 `formal_finite_guard()`，在 generator pre-backward/post-backward/post-step 和 global distill 对应阶段直接检查 tensor/grad/parameter；发现 NaN/Inf 立即 fail-fast。

### P2-06：formal runner 忽略训练子进程非零退出码

原 `subprocess.run(..., check=False)` 后不检查 returncode，只看 final_metrics 文件。

修复：非零退出直接 health FAIL，禁止进入候选。

### P3-07：`--checkpoint_segments` CLI 定义但 Stage2 调用未传入

原参数是 dead flag，实际始终使用 model 默认 `checkpoint_segments=1`。

修复：真实传入 `differentiable_sample()`。

### P2-08：CKR cache 未绑定 data identity

原 CKR cache key 只有 dataset/partition/client/task/resplit/seed；若同名 data*.pt 或 split 内容变化，可能静默复用旧 CKR。

修复：cache key 加 `data_identity_hash` 前缀。

### P0-09：review package 缺少本地模块

上传包没有：

- `louvain/` vendored tree；
- `pretrain_diffusion.py`。

`pretrain_diffusion.py` 只服务可选 `proxy_pretrained`，已改为 lazy import，不再阻塞正式 Stage1→Stage2 主线。

`louvain/` 对重建 partition 是必要依赖。本轮把 util 中 Louvain 改为 lazy import，因此已有 cached/frozen `data*.pt` 可以加载；但要重建 partition 必须从完整项目补回原 `louvain/`，**不建议用未知第三方版本静默替换**，否则客户端划分可能改变。

### P0-10：服务器依赖文件缺少 PyTorch/PyG wheel index

`requirements_frozen.txt` / `environment.yml` 锁定了 `torch==2.7.1+cu128` 和 `torch_scatter/torch_sparse/... +pt27cu128`，但原文件没有对应 wheel index/find-links。官方 PyTorch 的 cu128 安装需要 PyTorch wheel index；PyG 官方也要求从 `data.pyg.org` 对应 Torch/CUDA wheel 页面安装扩展。

修复：依赖文件加入 PyTorch cu128 extra index 与 PyG torch-2.7.0+cu128 find-links，并更新服务器指南。

## 未自动修改的算法级问题

### A. raw-radius runaway 根因仍需真实 trial 证据

静态数学已经否定“projected KL 可通过纯径向放大直接获利”的简单解释，但没有 study.db/log/checkpoint，无法测：

- semantic/disagreement/diversity/anchor 的真实 generator gradient norm；
- 哪一层先出现 parameter drift；
- lambda_sem>=0.01 是否真的仍会 runaway；
- stronger anchor vs raw-flow regularizer 谁更有效。

所以本轮**没有**把 anchor 0.01 改成 0.03，也**没有**加入 raw regularizer。

### B. `L_norm` 在 radius projection 主线下基本无梯度

这是数学上确认的 dead stabilizer，但删除/置零属于训练目标变更，需结合专利原文和最小机制实验再决定。

### C. global Adam 跨 FedAvg overwrite 保留 moments

local Adam 的 stale-momentum 问题已通过 `reset_each_round` 解决；global optimizer 仍跨轮持久化，而 global parameters 每轮会先被 FedAvg overwrite。它是否也构成 stale-momentum 问题，当前没有因果实验。标记 OPEN QUESTION，不自动修改。

### D. `fake_class_strategy=reliability/prior` 不是 formal 主线

当前 formal 使用 balanced。可选 reliability 分支把 `normalized_ckr.sum(dim=0)` 作为类别概率；由于 normalized CKR 每类跨客户端归一后列和约为 1，这一策略实际上接近 uniform，语义名与实现不匹配。非正式主线，未修改。

## 迁移/正式战役状态

原始上传包：**NOT READY FOR SERVER MIGRATION**。

审计修订版在代码/协议层可以进入服务器 **environment smoke + Cora-5 Stage1 + 极短 Stage2 smoke**，但在以下条件满足前，不建议直接启动 15-cell formal campaign：

1. 补回完整项目的 `louvain/`（至少保证 partition 可重建/复核）；
2. 上传专利申请书和 FedTAD 原论文 PDF，完成 source-of-truth 最终核验；
3. 若已有新的 lambda_sem>=0.01 runaway 证据，提供对应 study.db/log，先完成 gradient attribution；
4. 服务器安装环境后跑全部 regression tests；
5. 先只生成/验证 Cora-5 Stage1，再跑 1 个短 smoke，不直接启动 100-round Optuna。

## 本轮测试

- `python -m compileall -q .`：PASS
- `tests/test_checkpoint_state_mgmt.py`：3 PASS
- radius projection scale-invariance synthetic autograd：PASS
- radius-projected L_norm ~ zero-gradient synthetic test：PASS
- dedicated server RNG full vs checkpointed segments 1/2/3/6 exact output+gradient：PASS
- formal guard / full protocol compare / health regex synthetic harness：PASS
- 全量 pytest：NOT RUN（当前审计容器缺 `torch_geometric`，且无网络安装；不能伪报 PASS）

## 修订版 code hash

见 `server_migration_manifest.json`。该 hash 与旧 `290e25c26aa8` 不兼容；不要把旧 formal v2 study 直接 resume 到修订版。
