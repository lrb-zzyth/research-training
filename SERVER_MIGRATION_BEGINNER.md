# 服务器迁移新手手册（从零开始）

> 写给从未租过 Linux GPU 服务器的用户。所有命令可直接复制。
> 本手册只讲"怎么把项目搬上服务器跑起来"，不含 DevOps 高级内容。

## 0. 建议的服务器规格

| 项目 | 建议 |
|---|---|
| GPU | 16-24GB VRAM 单卡即可（本项目不是纯 GPU 计算密集型，不需要为 H100/A100 付高溢价） |
| 内存 | ≥32GB，推荐 64GB |
| CPU | ≥8 核，单核性能比核数更重要 |
| 硬盘 | ≥100GB NVMe |
| 系统 | Ubuntu 20.04 / 22.04（与本地 WSL2 环境最接近） |

## 1. 租完服务器你会拿到什么

租用平台（AutoDL/恒源云/矩池云等）付款后，页面会给你 4 样东西：
**IP 地址**（如 `120.46.12.34`）、**SSH 端口**（如 `22` 或 `12345`）、**用户名**（通常是 `root`）、**密码**（或让你自己设置）。

## 2. IP 是什么

IP 就是服务器在网络上的门牌号，形如 `120.46.12.34`。

## 3. 用户名是什么

登录服务器用的账号名，通常是 `root`。

## 4. 密码 / SSH key 是什么

密码：平台给你的登录密码（`ssh` 时会让你输入）。
SSH key：更安全的免密登录方式（可选，新手先用密码即可）。

## 5. Windows 怎么打开终端

1. 按 `Win + R`，输入 `cmd` 回车（或用 PowerShell）；
2. 或直接点击"Windows Terminal"。

## 6. ssh 命令怎么写

```bash
ssh root@120.46.12.34 -p 12345
```

把 `120.46.12.34` 换成你的 IP，`12345` 换成你的端口（默认 22 可省略 `-p 22`）。

## 7. 第一次看到 yes/no 怎么选

```
The authenticity of host ... Are you sure you want to continue connecting (yes/no)?
```

输入 `yes` 回车。这只是首次确认对方主机指纹，正常。

## 8. 怎么确认自己已经登录服务器

登录成功后，命令行前缀会变成类似 `root@ubuntu:~#`。敲一下 `hostname` 回车，能打印出服务器名字就说明已经在服务器上了。

## 9. pwd / ls / cd 是什么

```bash
pwd      # 显示当前所在目录
ls       # 列出当前目录文件
ls -la   # 列出全部（含隐藏文件）
cd 目录名   # 进入目录
cd ..    # 回上一级
```

## 10. 怎么上传项目

在**本地 Windows** 终端（不是服务器上）执行：

```bash
# 先在本机把项目打包（排除 dataset 和 runs 产物）
# 本机（WSL2 或 Git Bash）:
cd /home/lrb/federated/FedTAD
tar czf FedTAD_code.tar.gz \
  --exclude='dataset' --exclude='runs' --exclude='ckr' \
  --exclude='references' --exclude='presentation' --exclude='*.pt' \
  train_fedtad.py model.py util/ experiments/ scripts/ tests/ *.py *.md

# 上传到服务器（在 Windows 终端执行, 把 IP/端口换掉）:
scp -P 12345 FedTAD_code.tar.gz root@120.46.12.34:/root/
```

## 11. 怎么上传 dataset

在服务器上解压代码后：

```bash
cd /root && tar xzf FedTAD_code.tar.gz
mkdir -p dataset
```

然后本地打包 dataset（只打包划分好的部分，raw 数据服务器上下载不便时直接传）：

```bash
# 本地:
cd /home/lrb/federated/FedTAD
tar czf FedTAD_dataset.tar.gz dataset/Cora dataset/CiteSeer dataset/PubMed dataset/CS dataset/Physics
# 传到服务器:
scp -P 12345 FedTAD_dataset.tar.gz root@120.46.12.34:/root/
# 服务器上解压:
cd /root && tar xzf FedTAD_dataset.tar.gz
```

## 12. 怎么上传 Stage1 checkpoint

每个 cell 的 Stage1 checkpoint（`stage1_generator.pt`）必须在服务器上**重新生成**（Stage1 是联邦预训练，必须在正式协议下运行；见第 18 步）。不要直接复用本机 checkpoint。

## 13. 怎么创建 conda 环境

```bash
# 安装 miniconda（若平台镜像没预装）:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh   # 一路 yes, 最后 source ~/.bashrc
# 用项目导出的环境文件直接复现本地已验证环境:
cd /root
conda env create -f environment.yml -n fedtad5060
conda activate fedtad5060
```

## 14. 怎么安装依赖

`environment.yml` / `requirements_frozen.txt` 已加入 PyTorch cu128 官方 wheel index
与 PyG 对应 wheel 页面。PyTorch/PyG 的 CUDA 扩展不能只靠默认 PyPI 索引安装。
若 `conda env create` 个别 pip 包失败，可在激活环境后重试：

```bash
pip install -r requirements_frozen.txt
```

另外，正式项目必须包含原仓库 vendored 的 `louvain/` 目录；不要用未知版本的
第三方 Louvain 实现静默替换，否则客户端 partition 可能改变。

## 15. 怎么检查 GPU

```bash
nvidia-smi
```

能显示显卡名字和显存就 OK。若报 "command not found"，先联系平台客服装驱动。

## 16. 怎么跑 server_smoke_test.sh

```bash
cd /root
export STAGE1_PATH=/root/runs/formal_campaign_v3/Cora/clients5/stage1_generator.pt
bash scripts/server_smoke_test.sh
```

全部 PASS 才能继续。

## 17. 怎么开 tmux

```bash
tmux new -s fedtad
```

## 18. 怎么启动第一个 formal cell（含 Stage1）

```bash
# 先跑该 cell 的 Stage1 (20 联邦预训练轮, pretrain_diagnostic_only):
mkdir -p runs/formal_campaign_v3/Cora/clients5/stage1
python train_fedtad.py --root ./dataset --dataset Cora --num_clients 5 \
  --partition Louvain --num_rounds 1 --num_epochs 3 --hid_dim 64 --dropout 0.5 \
  --lr 1e-2 --weight_decay 5e-4 --task_mode multiclass \
  --selection_metric accuracy --f1_threshold=-1e6 --auc_threshold=-1e6 \
  --seed 2024 --reliability_holdout_ratio 0 \
  --diffusion_skip_mode timestep_scalar --diffusion_pretrain_rounds 20 \
  --pretrain_diagnostic_only \
  --checkpoint_dir runs/formal_campaign_v3/Cora/clients5/stage1 \
  > runs/formal_campaign_v3/Cora/clients5/stage1/stage1_training.log 2>&1

# 复制 checkpoint 到正式路径:
cp runs/formal_campaign_v3/Cora/clients5/stage1/pretrained_generator.pt \
   runs/formal_campaign_v3/Cora/clients5/stage1_generator.pt

# 检查 Stage1 health (raw radius 不得 nonfinite/数量级爆炸):
tail -20 runs/formal_campaign_v3/Cora/clients5/stage1/stage1_training.log

# 启动正式 Optuna (v2 搜索空间, 12 healthy trials, 串行):
bash scripts/run_formal_cell.sh Cora 5 12
```

## 19. 怎么退出 tmux 但不停止训练

在 tmux 里按 `Ctrl+B`，松开，再按 `D`。训练会继续跑。

## 20. 怎么重新连接

```bash
ssh root@120.46.12.34 -p 12345
tmux attach -t fedtad
```

## 21. 怎么看日志

```bash
tail -f runs/formal_campaign_v3/Cora/clients5/campaign.log
# 退出 tail: Ctrl+C
```

## 22. 怎么看 GPU/RAM

```bash
nvidia-smi            # GPU
free -h               # 内存
top                   # 进程 (按 q 退出)
```

## 23. 怎么安全停止训练

```bash
# 先找到训练进程:
pgrep -af "train_fedtad.py --root"
# 按 PID 结束 (用上面显示的数字):
kill <PID>
# 不要把 python 环境/数据集目录直接删除, 也不要 kill -9 非必要进程
```

## 24. 怎么下载最终结果

在**本地 Windows 终端**执行：

```bash
scp -P 12345 root@120.46.12.34:/root/runs/formal_campaign_v3/Cora/clients5/study_summary.csv .
scp -P 12345 root@120.46.12.34:/root/runs/formal_campaign_v3/Cora/clients5/best_params.json .
```

## 25. 怎么备份 study.db

```bash
# 服务器上:
cp runs/formal_campaign_v3/Cora/clients5/study.db \
   runs/formal_campaign_v3/Cora/clients5/study_backup_$(date +%Y%m%d_%H%M%S).db
# 或下载到本地:
scp -P 12345 root@120.46.12.34:/root/runs/formal_campaign_v3/Cora/clients5/study.db .
```

---

## 断点恢复原则（重要）

- 一切结果都在 `study.db`（Optuna SQLite）+ `trial_logs/` + `protocol.json`；
- 训练中断后直接重新执行 `bash scripts/run_formal_cell.sh Cora 5 12`，runner 会核对
  protocol hash / data identity / 搜索空间，一致则安全续跑（已完成 trial 不重跑）；
- 若 protocol 不一致会直接报错退出（fail-fast），不要手动绕过。

## 25. 调参完成后怎么跑正式 3-seed 决赛

只有该 cell 已生成 `best_params.json` 后才运行：

```bash
bash scripts/run_formal_final.sh Cora 5
```

该脚本固定串行运行 2024/2025/2026 三个 seed。每个 seed 的 100 轮训练只看
validation 选 best checkpoint；test 在训练结束、重新加载 val-best checkpoint 后只计算一次。
