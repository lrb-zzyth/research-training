# REPRODUCTION.md — 复现指南

## 一、环境

- Python 3.10+、PyTorch 2.x、PyTorch Geometric 2.4+（本项目在 torch 2.7.1 / pyg 2.6.1 / RTX 5060 验证）
- `pip install -r requirements.txt`
- 平台：PostgreSQL + `cd research-training/backend && pip install -r requirements.txt`
- 前端：`cd research-training/frontend && npm install`

## 二、命令行复现（核心算法）

默认配置即 B4 完整方法：

```bash
# B4 完整方法（默认）：Weighted CE + 子图对比 + 静态 CKR + 教师引导扩散式伪图蒸馏
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain

# B1 纯 FedAvg
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --distill_weighting none --contrastive_mode none --no-use_weighted_ce

# B2 / B3
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --distill_weighting none --contrastive_mode none
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --distill_weighting none

# 显式完整配置
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --task_mode anomaly_binary --normal_classes 0,1,2,3 --anomaly_classes 4,5,6 \
  --contrastive_mode subgraph_cross_view --ckr_mode static_topology \
  --distill_weighting static_ckr --fake_class_strategy balanced \
  --generator_init scratch
```

结构化事件与热更新（平台等价方式）：

```bash
mkdir -p /tmp/hot && echo '[{"parameter":"learning_rate","value":0.005}]' > /tmp/hot/pending_updates.json
python train_fedtad.py --dataset Cora --num_clients 2 --num_rounds 3 --num_epochs 1 \
  --emit_events --events_jsonl /tmp/hot/events.jsonl --param_update_dir /tmp/hot \
  --checkpoint_dir /tmp/hot/ckpt
```

## 三、平台复现（推荐）

```bash
# 1. 数据库
cd research-training/backend
createdb fedtad            # 或按 backend/README.md 创建用户与库
cp .env.example .env       # 编辑 DATABASE_URL

# 2. 后端
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 前端
cd research-training/frontend
npm install && npm run dev   # http://localhost:5173, admin / admin123
```

创建任务 → 监控 → 热更新 → 历史，完整流程见 [PLATFORM_GUIDE.md](PLATFORM_GUIDE.md)。

## 四、checkpoint 恢复

```bash
# 命令行：从任务产物恢复
python train_fedtad.py --dataset Cora --num_clients 10 --partition Louvain \
  --resume_checkpoint runs/platform/exp_<id>/checkpoints/best.pt \
  --checkpoint_dir runs/platform/exp_<id>/checkpoints --num_rounds 150
```

平台：新任务配置中把 `resume_checkpoint` 指向历史任务产物目录下的 best.pt / last.pt。

## 五、测试

```bash
# 核心算法
python test_all.py              # 59 项单元测试
python test_smoke.py            # 26 项冒烟
python test_smoke_train.py      # 4 组真实训练冒烟 (GPU)

# 平台
cd research-training/backend
python test_param_contract.py   # 16 项参数契约测试
python test_backend_smoke.py    # 57 项端到端 smoke (含 B1/B4 任务 + 热更新)

# 前端
cd research-training/frontend
npm run build

# 扩展层 (实验性, 默认不运行)
cd "$(git rev-parse --show-toplevel)" && python experimental/test_experiments.py
python experimental/test_phase2.py && python experimental/test_phase3.py
```

## 六、已知限制

- 联邦为单进程模拟（无真实跨进程通信）；隐私边界在生成/蒸馏损失函数层面成立，非形式化差分隐私；
- 平台训练默认不传 `--checkpoint_dir` 时产物落盘 `runs/platform/exp_<id>/checkpoints`；
- 无浏览器环境下可用 `test_backend_smoke.py` + 前端生产构建代替浏览器联合验证；
- 扩展层（experimental/）为历史实验系统，不进入默认训练入口。
