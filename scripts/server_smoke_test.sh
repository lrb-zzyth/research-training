#!/usr/bin/env bash
# Final v3 server preflight.  No formal Optuna trial is launched here.
set -euo pipefail
PY=${PY:-python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "=== [1] Python / PyTorch / PyG imports ==="
$PY - <<'PY'
import sys, torch, torch_geometric, optuna, numpy
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('torch_geometric', torch_geometric.__version__)
print('optuna', optuna.__version__)
print('numpy', numpy.__version__)
for name in ('torch_scatter','torch_sparse','torch_cluster'):
    mod = __import__(name)
    print(name, getattr(mod, '__version__', 'OK'))
assert torch.__version__.startswith('2.7.1'), torch.__version__
assert torch_geometric.__version__.startswith('2.6.1'), torch_geometric.__version__
PY

echo "=== [2] CUDA / GPU ==="
$PY - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('gpu:', torch.cuda.get_device_name(0))
print('cuda runtime:', torch.version.cuda)
print('capability:', torch.cuda.get_device_capability(0))
# RTX 4090 is sm_89; allow another CUDA-capable GPU for portability but report it.
x=torch.randn(512,512,device='cuda')
y=x@x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('CUDA matmul: PASS')
PY

echo "=== [3] Final-v3 pure math regressions ==="
$PY tests/test_final_v3_math.py

echo "=== [4] Core regression tests ==="
for t in \
  tests/test_generator_state_mgmt.py \
  tests/test_checkpoint_state_mgmt.py \
  tests/test_resume_freeze_policy.py \
  tests/test_stage2_smoke.py \
  tests/test_paired_rng.py \
  tests/test_formal_campaign.py; do
  echo "-- $t"
  $PY "$t"
done

echo "=== [5] Syntax/bytecode compile ==="
$PY -m compileall -q train_fedtad.py model.py util experiments/accuracy_benchmark tests

echo "=== [6] Formal v3 identity / strict serial ==="
$PY - <<'PY'
import sys
sys.path.insert(0, 'experiments/accuracy_benchmark')
import formal_campaign as fc
assert fc.FORMAL_VERSION == 'v3', fc.FORMAL_VERSION
assert fc.SEARCH_SPACE['lambda_sem']['choices'] == [0.01, 0.1, 1.0]
print('formal version:', fc.FORMAL_VERSION)
print('code hash:', fc.code_hash())
print('formal OUT:', fc.OUT)
PY

echo "=== [7] Optional Cora-5 cached-data identity ==="
if [ -d dataset/Cora/Client5/Louvain ]; then
  $PY - <<'PY'
import sys, torch
sys.path.insert(0, 'experiments/accuracy_benchmark')
import formal_campaign as fc
print('Cora c5 data_identity:', fc.compute_data_identity('Cora', 5))
for ci in range(5):
    p=f'dataset/Cora/Client5/Louvain/data{ci}.pt'
    d=torch.load(p,map_location='cpu',weights_only=False)
    assert d.x is not None and d.edge_index is not None
print('Cora-5 cached partitions: PASS')
PY
else
  echo "Cora-5 cached partition not present; dataset check SKIPPED"
fi

echo "=== [8] Optional Stage1 checkpoint load ==="
if [ -n "${STAGE1_PATH:-}" ]; then
  $PY - <<'PY'
import os, torch
from model import ConditionalDiffusionGenerator
p=os.environ['STAGE1_PATH']
assert os.path.exists(p), p
obj=torch.load(p,map_location='cpu',weights_only=False)
state=obj.get('generator_state_dict',obj) if isinstance(obj,dict) else obj
g=ConditionalDiffusionGenerator(1433,7,256,20,1e-4,0.5,
    output_bound='tanh',skip_mode='timestep_scalar',use_posterior_variance=True)
g.load_state_dict(state,strict=True)
print('Stage1 checkpoint strict load: PASS',p)
PY
else
  echo "STAGE1_PATH not set; checkpoint check SKIPPED"
fi

echo ""
echo "SERVER V3 SMOKE TEST: ALL REQUIRED CHECKS PASSED"
