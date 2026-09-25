"""
真实训练 Smoke Tests (Cora, 2 clients, 少量轮次)

  A. multiclass   : Cora 原始 7 类, hybrid_dynamic CKR, checkpointed 生成器,
                    保存 best/last 并恢复 checkpoint 继续训练, 结束加载 best 做最终 test
  B. anomaly_binary: normal=[0,2,3,4,5] anomaly=[1,6], hybrid_dynamic CKR,
                    reliability holdout, 至少一个类别支持不足的回退
  C. static_topology 消融: 静态模式仍可运行
  D. synthetic proxy DDPM 预训练 (不同特征维度, 含 FeatureAdapter),
                    加载进联邦训练

只报告真实运行结果, 不做性能提升声明。
"""
import sys, os, json, subprocess, tempfile, math
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BASE_ARGS = [
    '--seed', '2024', '--dataset', 'Cora', '--num_clients', '2',
    '--num_epochs', '1', '--hid_dim', '32', '--dropout', '0.2',
    '--contrastive_batch_size', '8', '--rwr_subgraph_size', '3',
    '--edge_perturb_ratio', '0.2', '--diffusion_steps', '4',
    '--diffusion_hidden', '32', '--generator_steps', '1',
    '--distill_steps', '1', '--fake_nodes', '48',
    '--generator_warmup_rounds', '0',
    '--rwr_cache', 'enabled', '--rwr_cache_view1_persistent', 'enabled',
    '--rwr_seed', '0',
]


def run(cmd, label, timeout=900):
    print(f"\n{'=' * 70}\n[RUN {label}] {cmd}\n{'=' * 70}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    tail = proc.stdout.splitlines()
    print('\n'.join(tail[-30:]))
    if proc.returncode != 0:
        print(f"[FAIL {label}] exit={proc.returncode}")
        print(proc.stderr[-3000:])
        raise SystemExit(f"smoke {label} failed (exit {proc.returncode})")
    print(f"[OK {label}] exit=0")
    return proc.stdout


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# =====================================================================
#  A. multiclass + hybrid CKR + checkpointed + save/resume/best-load
# =====================================================================
print("\n### SMOKE A: multiclass hybrid_dynamic, checkpoint save+resume ###")
with tempfile.TemporaryDirectory() as tmpA:
    ckptA = os.path.join(tmpA, 'ckpt')
    logA1 = os.path.join(tmpA, 'logs1')
    logA2 = os.path.join(tmpA, 'logs2')

    out1 = run([PY, 'train_fedtad.py'] + BASE_ARGS + [
        '--task_mode', 'multiclass', '--num_rounds', '2',
        '--ckr_mode', 'hybrid_dynamic', '--distill_weighting', 'dynamic_ckr',
        '--generator_backprop_mode', 'checkpointed',
        '--checkpoint_dir', ckptA, '--save_last_checkpoint',
        '--log_dir', logA1], 'A-phase1')

    bestA = os.path.join(ckptA, 'best.pt')
    lastA = os.path.join(ckptA, 'last.pt')
    assert os.path.exists(bestA), "best.pt missing"
    assert os.path.exists(lastA), "last.pt missing"
    import torch
    ck = torch.load(bestA, map_location='cpu', weights_only=False)
    assert 'global_model_state' in ck and 'generator_state' in ck \
        and 'ckr_tracker_state' in ck and 'global_optimizer_state' in ck, \
        "best.pt must contain global/generator/CKR/optimizer states"

    out2 = run([PY, 'train_fedtad.py'] + BASE_ARGS + [
        '--task_mode', 'multiclass', '--num_rounds', '3',
        '--ckr_mode', 'hybrid_dynamic', '--distill_weighting', 'dynamic_ckr',
        '--generator_backprop_mode', 'checkpointed',
        '--resume_checkpoint', bestA,
        '--checkpoint_dir', ckptA,
        '--log_dir', logA2], 'A-phase2')

    assert '[Resume]' in out2, "resume message missing"
    assert '加载 best.pt' in out2, "final test must load best.pt"
    assert '[Final Test]' in out2

    r1 = load_jsonl(os.path.join(logA1, 'dynamic_ckr_history.jsonl'))
    r2 = load_jsonl(os.path.join(logA2, 'dynamic_ckr_history.jsonl'))
    rounds1 = sorted({r['round'] for r in r1})
    rounds2 = sorted({r['round'] for r in r2})
    print(f"  phase1 rounds={rounds1}, phase2 rounds={rounds2}")
    assert len(rounds1) == 2 and len(rounds2) >= 1, "round continuity broken"
    # 恢复语义: 从 best.pt 的 round+1 继续 (ckpt 存 0-based 内部轮; tracker 记 1-based)。
    # 注意不能断言 max(rounds1)+1 == min(rounds2) —— 若 phase1 最优轮不是最后一轮,
    # 恢复会从最优轮之后继续 (语义正确, 但会重跑 phase1 已跑过的若干轮)。
    assert min(rounds2) == ck['round'] + 2, \
        f"resume must continue after checkpoint round {ck['round']}, " \
        f"got min phase2 round {min(rounds2)}"

    # EMA 连续性 (跨 checkpoint 边界): 恢复后首轮 vs 恢复前同轮次的末条记录
    #   ema(t) = 0.8*ema(t-1) + 0.2*(0.5*static_scaled + 0.5*dynamic_metric)
    # 若 tracker 未从 checkpoint 恢复 (重新初始化), 首轮 ema 会等于 candidate,
    # 该等式即失效 -> 这正是"恢复后 EMA 连续"的判定
    by_round1 = {}
    for rec in r1:
        by_round1.setdefault(rec['round'], []).append(rec)
    first2 = min(rounds2)
    t_prev = first2 - 1          # 恢复前 tracker 已记录的最后一轮
    assert t_prev in by_round1, f"phase1 history missing round {t_prev}"
    cells1 = {(rec['client_id'], rec['class_id']): rec for rec in by_round1[t_prev]}
    cells2 = {(rec['client_id'], rec['class_id']): rec
              for rec in r2 if rec['round'] == first2}
    checked = 0
    for key, rec2 in cells2.items():
        rec1 = cells1.get(key)
        if rec1 is None or rec1['status'] != 'available' or rec2['status'] != 'available':
            continue
        expected = (0.8 * rec1['ema_ckr']
                    + 0.2 * (0.5 * rec2['static_scaled'] + 0.5 * rec2['dynamic_signal']))
        assert abs(rec2['ema_ckr'] - expected) < 1e-4, \
            f"EMA continuity broken across resume at {key}: {rec2['ema_ckr']} vs {expected}"
        checked += 1
    assert checked > 0, "no available cells to check EMA continuity"
    print(f"  EMA continuity across checkpoint resume verified over "
          f"{checked} (client,class) cells (round {t_prev} -> {first2}) ✓")
    print("SMOKE A PASSED ✓")


# =====================================================================
#  B. anomaly_binary hybrid CKR with reliability holdout fallback
# =====================================================================
print("\n### SMOKE B: anomaly_binary, reliability holdout, fallback ###")
with tempfile.TemporaryDirectory() as tmpB:
    logB = os.path.join(tmpB, 'logs')
    ckptB = os.path.join(tmpB, 'ckpt')
    outB = run([PY, 'train_fedtad.py'] + BASE_ARGS + [
        '--task_mode', 'anomaly_binary',
        '--normal_classes', '0,2,3,4,5', '--anomaly_classes', '1,6',
        '--num_rounds', '2', '--ckr_mode', 'hybrid_dynamic',
        '--distill_weighting', 'dynamic_ckr',
        '--generator_backprop_mode', 'checkpointed',
        '--reliability_holdout_ratio', '0.2',
        # 异常类 reliability support ≈ 4/2 (两个客户端), 提高到 5 强制 unavailable
        '--reliability_min_support', '5',
        '--log_dir', logB, '--checkpoint_dir', ckptB], 'B-anomaly')

    assert 'WARNING: validation labels are being used' not in outB, \
        "default eval source must be reliability_holdout (no val warning)"
    assert '[Resplit stratified]' in outB
    assert '[Anomaly CKR]' in outB, "anomaly CKR summary missing"
    assert 'zero_anomaly_reliability_split' in outB

    recsB = load_jsonl(os.path.join(logB, 'dynamic_ckr_history.jsonl'))
    assert recsB, "dynamic CKR JSONL empty"
    falls = [r for r in recsB if r['fallback']]
    assert falls, "expected at least one fallback (class with insufficient support)"
    reasons = {r['fallback_reason'] for r in falls}
    print(f"  {len(falls)} fallback records, reasons={sorted(reasons)}")
    for r in falls[:4]:
        print(f"    c{r['client_id']} cls{r['class_id']} support={r['train_support']} "
              f"fallback={r['fallback_reason']}")
    # 回退记录不得为 NaN/0
    for r in falls:
        assert not math.isnan(r['ema_ckr']), "fallback ema must not be NaN"
    print("SMOKE B PASSED ✓")


# =====================================================================
#  C. static_topology ablation
# =====================================================================
print("\n### SMOKE C: static_topology ablation ###")
with tempfile.TemporaryDirectory() as tmpC:
    outC = run([PY, 'train_fedtad.py'] + BASE_ARGS + [
        '--task_mode', 'multiclass', '--num_rounds', '1',
        '--ckr_mode', 'static_topology',
        '--generator_backprop_mode', 'checkpointed',
        '--checkpoint_dir', os.path.join(tmpC, 'ckpt'),
        '--log_dir', os.path.join(tmpC, 'logs')], 'C-static')
    assert 'CKR mode = static_topology' in outC
    assert '[Generator Update]' in outC and '[Global Distillation]' in outC
    print("SMOKE C PASSED ✓")


# =====================================================================
#  D. synthetic proxy DDPM pretrain (dims differ -> FeatureAdapter) + load
# =====================================================================
print("\n### SMOKE D: synthetic proxy pretrain + load into federated ###")
with tempfile.TemporaryDirectory() as tmpD:
    proxy_ckpt = os.path.join(tmpD, 'proxy.pt')
    outD1 = run([PY, 'pretrain_diffusion.py',
                 '--proxy_n_samples', '1500', '--proxy_epochs', '4',
                 '--proxy_feature_dim', '32', '--target_feature_dim', '1433',
                 '--diffusion_steps', '4', '--diffusion_hidden', '32',
                 '--proxy_pretrain_mode', 'unconditional', '--num_classes', '7',
                 '--proxy_checkpoint', proxy_ckpt], 'D-pretrain')
    assert 'checkpoint saved' in outD1

    outD2 = run([PY, 'train_fedtad.py'] + BASE_ARGS + [
        '--task_mode', 'multiclass', '--num_rounds', '1',
        '--ckr_mode', 'hybrid_dynamic',
        '--generator_backprop_mode', 'checkpointed',
        '--generator_init', 'proxy_pretrained',
        '--proxy_checkpoint', proxy_ckpt,
        '--checkpoint_dir', os.path.join(tmpD, 'ckpt'),
        '--log_dir', os.path.join(tmpD, 'logs')], 'D-federated-load')
    assert '加载代理 DDPM 预训练权重' in outD2, "proxy weights not loaded"
    assert 'proxy checkpoint 特征维度不兼容' not in outD2, "dim check must pass"
    print("SMOKE D PASSED ✓")


print(f"\n{'=' * 70}")
print("ALL 4 TRAINING SMOKE TESTS PASSED (A multiclass / B anomaly_binary / C static / D proxy)")
print(f"{'=' * 70}")
