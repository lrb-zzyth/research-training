#!/usr/bin/env python3
"""
逐数据集调参 + 用最优配置重跑 (全自动队列)。

流程 (每个数据集):
  1. Optuna 搜索 (目标 = **验证集精度**, 严禁用测试集选参)
  2. 取最优参数
  3. 用最优参数在 5/10/20 三个档位上各跑 3 个种子 (2024/2025/2026, 固定集)
     3 种子是对齐论文 "3 standardized training"; 种子集是预先固定的, 不按结果挑选
  4. 全部落盘到 runs/accuracy_benchmark/TUNED_<ds>_c<nc>/seed_<s>/

严格串行 (本机 7GB 内存, 单 run 峰值可达 3.6GB)。

用法: python experiments/accuracy_benchmark/tune_then_run.py
"""
import json
import os
import subprocess
import sys
import time

import optuna

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = "/home/lrb/miniconda3/envs/fedtad5060/bin/python"
OUT = os.path.join(REPO, "runs", "accuracy_benchmark")
SEEDS = [2024, 2025, 2026]   # 3 种子: 对齐论文 "3 standardized training" (固定集, 非挑选)

# 每个数据集的调参预算 (按单 run 耗时定, Physics 最贵)
BUDGET = {
    "Cora":     20,
    "CiteSeer": 20,
    "CS":       16,
    "PubMed":   16,
    "Physics":  10,
}
TUNE_TIER = 10          # 在 10 客户端档上调参
FINAL_TIERS = [5, 10, 20]

SEARCH_SPACE_HELP = """
搜索空间 (对齐论文 λ1/λ2/I/Ig/Id, 另加本方法实测敏感项):
  lambda_sem        {0.001,0.01,0.1,1.0}   生成器语义损失系数 (=论文 λ1)
  lambda_diversity  {0.001,0.01,0.1}       生成器多样性损失系数 (=论文 λ2)
  distill_steps     {1,3,5,10,25}             蒸馏内迭代 (=论文 I)
  generator_steps   {1,3,5}                生成器内迭代 (=论文 Ig)
  contrastive_temperature {0.05,0.1,0.2,0.35}  对比温度 (实测越小越好)
  lambda_subgraph   {0.05,0.1,0.5}         对比损失权重
  knn_k             {3,5,10}               KNN 构图 k (实测默认 5 最差)
  fake_nodes        {50,100,200}           伪节点数
"""


def mem_ok(min_mb=3000):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024 >= min_mb
    return False


def wait_no_train(timeout=7200):
    """等所有训练进程退出 (串行保证)。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run(["pgrep", "-f", "train_fedtad.py --root"],
                           capture_output=True)
        if r.returncode != 0:
            return True
        time.sleep(15)
    return False


def run_one(ds, nc, seed, outdir, extra=()):
    """跑一次训练, 返回 accuracy 或 None。"""
    os.makedirs(outdir, exist_ok=True)
    if not wait_no_train():
        print("  [warn] 等训练进程超时")
    if not mem_ok():
        print("  [skip] 内存不足")
        return None
    cmd = [PY, "train_fedtad.py",
           "--root", "./dataset", "--dataset", ds,
           "--num_clients", str(nc), "--partition", "Louvain",
           "--num_rounds", "100", "--num_epochs", "3",
           "--hid_dim", "64", "--dropout", "0.5",
           "--lr", "1e-2", "--weight_decay", "5e-4",
           "--task_mode", "multiclass", "--selection_metric", "accuracy",
           "--f1_threshold=-1e6", "--auc_threshold=-1e6",
           "--seed", str(seed), "--checkpoint_dir", outdir,
           # 对齐 FedTAD: 客户端在**全部** train_idx 上训练。
           # 本仓库默认会留 20% 作 reliability_idx(仅动态 CKR 需要),
           # 在 static_topology 下该保留集不被使用, 却使训练数据少 20%。
           "--reliability_holdout_ratio", "0",
           "--emit_events", "--events_jsonl", os.path.join(outdir, "events.jsonl"),
           "--final_metrics_json", os.path.join(outdir, "final_metrics.json"),
           "--metrics_jsonl", os.path.join(outdir, "metrics.jsonl"),
           "--split_report_json", os.path.join(outdir, "split_report.json"),
           "--resource_usage_json", os.path.join(outdir, "resource_usage.json")]
    for k, v in extra:
        cmd += [f"--{k}", str(v)]
    with open(os.path.join(outdir, "stdout.log"), "w") as fo, \
         open(os.path.join(outdir, "stderr.log"), "w") as fe:
        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=REPO, stdout=fo, stderr=fe,
                           timeout=21600, check=False)
        except subprocess.TimeoutExpired:
            print("    [timeout]")
            return None
        wall = time.time() - t0
    fp = os.path.join(outdir, "final_metrics.json")
    if not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    val = d.get("best_val_primary")
    acc = (d.get("metrics", {}).get("pooled", {}) or {}).get("accuracy")
    return val, acc, wall


def suggest(trial):
    return {
        "lambda_sem": trial.suggest_categorical("lambda_sem", [0.001, 0.01, 0.1, 1.0]),
        "lambda_diversity": trial.suggest_categorical("lambda_diversity", [0.001, 0.01, 0.1]),
        # 加入 25: 原版 FedTAD 每轮蒸馏 = glb_epochs(5) × it_d(5) = 25,
        # 原搜索空间上限只有 10, 永远够不到原版强度
        "distill_steps": trial.suggest_categorical("distill_steps", [1, 3, 5, 10, 25]),
        "generator_steps": trial.suggest_categorical("generator_steps", [1, 3, 5]),
        "contrastive_temperature": trial.suggest_categorical(
            "contrastive_temperature", [0.05, 0.1, 0.2, 0.35]),
        "lambda_subgraph": trial.suggest_categorical("lambda_subgraph", [0.05, 0.1, 0.5]),
        "knn_k": trial.suggest_categorical("knn_k", [3, 5, 10]),
        "fake_nodes": trial.suggest_categorical("fake_nodes", [50, 100, 200]),
    }


def tune(ds, n_trials):
    storage = f"sqlite:///{os.path.join(OUT, 'optuna_tune.db')}"
    study = optuna.create_study(
        study_name=f"tune4_{ds}_c{TUNE_TIER}", storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=2024))

    def objective(trial):
        params = suggest(trial)
        d = os.path.join(OUT, f"TUNE_{ds}", f"trial_{trial.number:03d}")
        print(f"  [Trial {trial.number}] {params}")
        r = run_one(ds, TUNE_TIER, 2024, d, extra=list(params.items()))
        if r is None:
            raise optuna.TrialPruned()
        val, acc, wall = r
        print(f"    -> val={val:.2f} test={acc:.2f} wall={wall:.0f}s")
        trial.set_user_attr("test_acc", acc)
        return val

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    b = study.best_trial
    print(f"  [{ds}] 最优 (按验证集): val={b.value:.2f} params={b.params}")
    print(f"        该 trial 的 test={b.user_attrs.get('test_acc')}")
    return dict(b.params)


def main():
    print(__doc__)
    print(SEARCH_SPACE_HELP)
    for ds, n in BUDGET.items():
        print("\n" + "#" * 70)
        print(f"#  数据集 {ds}: 调参 {n} trials + 三档重跑")
        print("#" * 70)
        best = tune(ds, n)
        with open(os.path.join(OUT, f"tuned_params_{ds}.json"), "w") as f:
            json.dump(best, f, indent=1)
        # 用最优配置在三个档位上各跑 5 种子
        for nc in FINAL_TIERS:
            for s in SEEDS:
                d = os.path.join(OUT, f"TUNED_{ds}_c{nc}", f"seed_{s}")
                print(f"  [{ds} c{nc} seed{s}] 开始")
                r = run_one(ds, nc, s, d, extra=list(best.items()))
                if r:
                    print(f"    -> val={r[0]:.2f} test={r[1]:.2f} wall={r[2]:.0f}s")
    print("\nALL_TUNE_AND_RUN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
