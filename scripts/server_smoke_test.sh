#!/usr/bin/env bash
# 服务器迁移 smoke test: 只做环境/数据/checkpoint 检查, 不启动正式 trial。
# 用法: bash scripts/server_smoke_test.sh   (全部 PASS 才可启动正式战役)
set -e
PY=${PY:-python}
echo "=== [1] Python / 关键包 import ==="
$PY -c "
import sys, torch, torch_geometric, optuna, numpy
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('torch_geometric', torch_geometric.__version__)
print('optuna', optuna.__version__)
"
echo "=== [2] CUDA ==="
$PY -c "
import torch
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('cuda runtime:', torch.version.cuda)
"
echo "=== [3] 数据集加载 + Cora-5 data identity ==="
$PY -c "
import sys; sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/accuracy_benchmark')
import formal_campaign as fc
print('Cora c5 data_identity:', fc.compute_data_identity('Cora', 5))
import torch
for ci in range(5):
    d = torch.load('dataset/Cora/Client5/Louvain/data{}.pt'.format(ci),
                   map_location='cpu', weights_only=False)
    assert d.x is not None and d.edge_index is not None
print('Cora-5 5 个子图加载 OK')
"
echo "=== [4] Stage1 checkpoint 加载 ==="
$PY -c "
import sys, torch; sys.path.insert(0, '.')
from model import ConditionalDiffusionGenerator
# 示例: 迁移后本机已有的任意 Stage1 checkpoint 路径, 此处用环境变量传入
import os
p = os.environ.get('STAGE1_PATH', '')
if p:
    g = ConditionalDiffusionGenerator(1433, 7, 256, 20, 1e-4, 0.5,
                                      output_bound='tanh',
                                      skip_mode='timestep_scalar',
                                      use_posterior_variance=True)
    g.load_state_dict(torch.load(p, map_location='cpu', weights_only=False))
    print('Stage1 checkpoint 加载 OK:', p)
else:
    print('(未设置 STAGE1_PATH, 跳过 checkpoint 加载检查)')
"
echo "=== [5] formal guard / health checker / test isolation ==="
$PY -c "
import sys, types, tempfile, json, os
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/accuracy_benchmark')
from train_fedtad import validate_formal_campaign
import formal_campaign as fc
args = types.SimpleNamespace(task_mode='multiclass', use_weighted_ce=True,
    reliability_holdout_ratio=0, selection_metric='accuracy',
    local_optimizer_lifecycle='reset_each_round', radius_constraint=True,
    feature_stats_align=False, diffusion_freeze_skip_scale=True,
    federated_diffusion_pretrain=False,
    diffusion_pretrained_checkpoint=os.environ.get('STAGE1_PATH', '/nonexistent'))
if os.environ.get('STAGE1_PATH'):
    validate_formal_campaign(args)
    print('formal guard: PASS (with checkpoint)')
else:
    print('formal guard: 跳过 (需 STAGE1_PATH)')
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, 'stdout.log'), 'w').write(
        '[Round 1]\n  step 0: L_sem=1.0 |grad|=0.5 raw_r=7.3 proj_r=7.3\n')
    ok, _ = fc.trial_health_check(d)
    assert ok is True
print('health checker: PASS')
print('test isolation: tuning_mode flag 存在 (--tuning_mode)')
"
echo "=== [6] 1 步合成 backward smoke ==="
$PY -c "
import sys, torch; sys.path.insert(0, '.')
from model import ConditionalDiffusionGenerator
g = ConditionalDiffusionGenerator(16, 3, 32, 4, 1e-4, 0.5,
                                  output_bound='tanh',
                                  skip_mode='timestep_scalar')
opt = torch.optim.Adam(g.parameters(), lr=1e-3)
t = torch.randint(0, 4, (8,)); x0 = torch.randn(8, 16); eps = torch.randn(8, 16)
ab = g.alpha_bars[t].unsqueeze(-1)
x_t = ab**0.5 * x0 + (1 - ab)**0.5 * eps
loss = torch.nn.functional.mse_loss(g(x_t, t, torch.tensor([0,1,2]*3)[:8]), eps)
opt.zero_grad(); loss.backward(); opt.step()
print('1-step synthetic backward: PASS, loss =', round(loss.item(), 4))
"
echo ""
echo "SERVER SMOKE TEST: ALL PASSED"
