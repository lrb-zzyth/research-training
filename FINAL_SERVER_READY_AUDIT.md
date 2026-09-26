# FINAL SERVER-READY AUDIT — Formal Campaign v3

Date: 2026-09-26

## Verdict

**SERVER-READY CANDIDATE, with mandatory server smoke test before any long formal run.**

The training math now matches the project source-of-truth for the two places that were still wrong in the uploaded package:

1. **FedTAD Eq.(10) / patent S4.2 KL direction:** `KL(global || local)`.
2. **Patent S2.2 local RWR semantics:** fallback never injects nodes from another connected component.

Formal campaign identity is bumped to **v3**.  Do **not** resume or merge any v1/v2 Optuna study into v3.

Current formal code hash: **f38f7fe74ab5**.

## Confirmed fixes in v3

### P0 — KL direction corrected

The uploaded code used `F.kl_div(log_softmax(global), softmax(local))`, which computes `KL(local || global)` under PyTorch's input/target convention.  FedTAD Eq.(10) and patent S4.2 require `KL(global || local)`.

v3 introduces `util/kl_utils.py::categorical_kl_from_logits()` and uses the explicit formula

`sum p_global * (log p_global - log p_local)`

in both:

- generator disagreement (generator maximizes it), and
- global/student distillation (global model minimizes it).

Generator-stage global/local parameters remain frozen while gradients with respect to fake features are preserved.

### P1 — RWR local-subgraph fallback corrected

The uploaded fallback filled short RWR samples with random nodes from the *entire client graph*.  On fragmented Louvain clients this could mix disconnected components into one mean-pooled local-subgraph representation.

v3 behavior:

- RWR first;
- if still short, BFS expands only inside the anchor connected component;
- if the component is smaller than the requested size, return a smaller valid local subgraph;
- never inject disconnected nodes.

Variable-size subgraphs are already supported by the existing batched mean-readout code.

### P1 — Radius projection made overflow-stable without changing its mathematics

The original `sqrt(sum(raw_x**2))` can overflow in float32 even when every raw element is still finite.  That can turn the projected feature into zero and silently kill the server adversarial signal.

v3 implements the same map `r * z / ||z||` after a detached per-row max rescaling.  The rescaling cancels algebraically and preserves the radius projection while avoiding squared-norm overflow for large finite raw trajectories.

Diagnostics compute raw/projected radii in float64, while the formal in-process guard separately rejects truly nonfinite raw/projected tensors.

### P1 — Checkpointed DDPM RNG and CLI wiring

- `--checkpoint_segments` is actually passed into `differentiable_sample()`.
- initial/reverse server noise uses dedicated deterministic server RNG streams.
- reverse-step noise is materialized outside checkpoint regions so backward recomputation reuses identical stochastic inputs.
- checkpointed segments 1/2/3/6 match full-mode output and parameter gradients exactly in the synthetic CPU regression.

### P2 — Formal protocol hardening retained

- strict single-GPU serial execution (`n_jobs=1` fail-fast);
- tuning has zero test evaluation;
- final 3-seed runner selects by validation and evaluates test once after loading val-best;
- process nonzero exit => failed trial;
- in-process finite guards for loss/gradient/parameters/raw/projected generator outputs;
- resume protocol/data/code-hash mismatch => fail-fast;
- formal search space remains v2 (`lambda_sem={0.01,0.1,1.0}`), but campaign/training-math identity is **v3**.

## Client-side innovation status

- weighted CE: retained;
- undirected edge perturbation: retained;
- subgraph-subgraph cross-view InfoNCE: retained;
- denominator includes the positive pair itself: verified;
- target-node feature masking: retained;
- shared GCN encoder + mean readout: retained;
- RWR locality bug: fixed as described above.

## Server-side innovation status

- conditional DDPM: retained;
- federated Stage1 true noise-prediction pretraining: retained;
- timestep-scalar residual: retained;
- posterior variance: retained;
- frozen Stage1 skip scale in Stage2: retained;
- stable radius manifold projection: retained;
- CKR-weighted semantic CE: retained;
- CKR-weighted `KL(global || local)`: corrected;
- cosine + KNN pseudo graph: retained;
- alternating generator/global optimization: retained;
- L2-SP anchor fixed at `1e-2`: retained.

No KL→L1, DDPM→MLP, InfoNCE-form, CKR, or patent-structure regression was introduced.

## Tests actually run in the audit environment

PASS:

- Python `compileall` over the package.
- `bash -n` over all shell scripts.
- final v3 pure-math/regression suite (10 tests):
  - exact `KL(global || local)` vs `torch.distributions` reference;
  - direct generator disagreement path equals `KL(global || local)` and preserves fake-feature gradient;
  - direct student distillation path equals `KL(global || local)` and detaches fake features;
  - InfoNCE denominator regression;
  - undirected edge-perturbation pairing;
  - disconnected-component RWR regression;
  - local BFS fallback regression;
  - huge-finite raw projection stays finite and preserves target radius;
  - radial projection gradient regression;
  - DDPM full vs checkpointed segment 1/2/3/6 exact output/gradient equivalence;
  - checkpoint-segment and formal-v3 wiring checks.
- formal runner import/version/code-hash check.
- formal health finite vs `raw_r=inf` regression.

NOT RUN locally:

- real PyTorch Geometric end-to-end training;
- CUDA/RTX 4090 execution;
- dataset-backed 4-stage training smoke.

Reason: the audit runtime has CPU PyTorch but does not have `torch_geometric` or CUDA.  These are therefore mandatory parts of the **server smoke test**, not silently marked PASS here.

## Mandatory server gate

On the rented RTX 4090 server, after environment/data upload:

```bash
cd <repo>
bash scripts/server_smoke_test.sh
```

Only if that prints:

`SERVER V3 SMOKE TEST: ALL REQUIRED CHECKS PASSED`

should you generate v3 Stage1 checkpoints and launch a formal cell.

Recommended first formal sequence:

```bash
bash scripts/generate_stage1_cell.sh Cora 5
bash scripts/run_formal_cell.sh Cora 5 12
```

Do not run all 15 cells until Cora-5 v3 finishes healthily and the v3 logs are reviewed.

## Study compatibility

**All prior v1/v2 tuning/final results are diagnostic/history only.**  The KL direction and RWR semantics changed, so v3 must use fresh studies under:

`runs/formal_campaign_v3/`

Stage1 DDPM architecture/training itself was not changed by the KL/RWR patch, but for the cleanest paper provenance the provided v3 scripts generate fresh per-cell Stage1 artifacts.
