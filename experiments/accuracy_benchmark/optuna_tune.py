#!/usr/bin/env python3
"""
Optuna 超参搜索 (复刻论文做法, 但更严格)。

论文原文: "we perform the hyperparameter search for FedTAD using the Optuna
framework on λ1 and λ2 within {10^-1,10^-2,10^-3}, and I,Ig,Id within {1,3,5,10}."

本脚本对应的参数映射:
    λ1 (生成器语义损失系数)   -> --lambda_sem
    λ2 (生成器多样性损失系数) -> --lambda_diversity
    I  (蒸馏迭代)             -> --distill_steps
    Ig (生成器内迭代)         -> --generator_steps
  另加入我们实测有效/待细化的:
    --contrastive_temperature (实测 τ 越小越好)
    --lambda_subgraph         (对比损失权重)
    --knn_k, --fake_nodes

⚠️ 目标函数 = **验证集精度 best_val_primary**, 不是测试集精度。
   任务书明确"严禁用测试集选轮"; 用 test 调参 = 过拟合测试集, 结果不可信。
   测试集精度只在调参结束后, 用最优配置跑多种子时报告。

严格串行 (本机 7GB 内存, 单 run 峰值 2.2~4.2GB)。

用法:
    python experiments/accuracy_benchmark/optuna_tune.py --dataset Cora --trials 24
"""
import argparse
import json
import os
import subprocess
import sys
import time

import optuna

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = "/home/lrb/miniconda3/envs/fedtad5060/bin/python"
OUT_ROOT = os.path.join(REPO, "runs", "accuracy_benchmark")


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return 0


def run_trial(args, params, trial_dir):
    """跑一次训练, 返回 (val_acc, test_acc, wall_sec)。失败返回 None。"""
    os.makedirs(trial_dir, exist_ok=True)
    avail = mem_available_mb()
    if avail < 3000:
        print(f"  [skip] 内存不足 {avail}MB")
        return None

    cmd = [PY, "train_fedtad.py",
           "--root", "./dataset", "--dataset", args.dataset,
           "--num_clients", str(args.num_clients), "--partition", "Louvain",
           "--num_rounds", str(args.num_rounds), "--num_epochs", "3",
           "--hid_dim", "64", "--dropout", "0.5",
           "--lr", "1e-2", "--weight_decay", "5e-4",
           "--task_mode", "multiclass", "--selection_metric", "accuracy",
           "--f1_threshold=-1e6", "--auc_threshold=-1e6",
           "--seed", str(args.seed),
           "--checkpoint_dir", trial_dir,
           "--final_metrics_json", os.path.join(trial_dir, "final_metrics.json")]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    with open(os.path.join(trial_dir, "stdout.log"), "w") as fo, \
         open(os.path.join(trial_dir, "stderr.log"), "w") as fe:
        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=REPO, stdout=fo, stderr=fe,
                           timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            print(f"  [timeout] >{args.timeout}s")
            return None
        wall = time.time() - t0

    fp = os.path.join(trial_dir, "final_metrics.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        d = json.load(f)
    val = d.get("best_val_primary")
    test = (d.get("metrics", {}).get("pooled", {}) or {}).get("accuracy")
    (d.setdefault("_trial", {})).update({"wall_sec": wall, "params": params})
    with open(fp, "w") as f:
        json.dump(d, f, indent=1)
    return val, test, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--num_rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--study", default=None)
    args = ap.parse_args()

    study_name = args.study or f"tune_{args.dataset}_c{args.num_clients}"
    storage = f"sqlite:///{os.path.join(OUT_ROOT, 'optuna.db')}"
    os.makedirs(OUT_ROOT, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed))

    def objective(trial):
        params = {
            # 论文搜索的三个: λ1(语义), λ2(多样性), I(蒸馏迭代), Ig(生成器迭代)
            "lambda_sem": trial.suggest_categorical(
                "lambda_sem", [0.001, 0.01, 0.1, 1.0]),
            "lambda_diversity": trial.suggest_categorical(
                "lambda_diversity", [0.001, 0.01, 0.1]),
            "distill_steps": trial.suggest_categorical("distill_steps", [1, 3, 5, 10]),
            "generator_steps": trial.suggest_categorical("generator_steps", [1, 3, 5]),
            # 我们实测待细化的
            "contrastive_temperature": trial.suggest_categorical(
                "contrastive_temperature", [0.05, 0.1, 0.2, 0.35]),
            "lambda_subgraph": trial.suggest_categorical(
                "lambda_subgraph", [0.05, 0.1, 0.5]),
            "knn_k": trial.suggest_categorical("knn_k", [3, 5, 10]),
        }
        d = os.path.join(OUT_ROOT, f"OPTUNA_{study_name}", f"trial_{trial.number:03d}")
        print(f"\n[Trial {trial.number}] {params}")
        r = run_trial(args, params, d)
        if r is None:
            raise optuna.TrialPruned()
        val, test, wall = r
        print(f"  -> val={val:.2f}  test={test:.2f}  wall={wall:.0f}s")
        trial.set_user_attr("test_acc", test)
        return val

    print(f"study={study_name}  dataset={args.dataset}  trials={args.trials}")
    print(f"已有 trial 数: {len(study.trials)}")
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("\n" + "=" * 60)
    print("最优 trial (按验证集精度):")
    b = study.best_trial
    print(f"  val={b.value:.2f}  params={b.params}")
    print(f"  该 trial 的 test={b.user_attrs.get('test_acc')}")
    print("\n注意: '最优'是按**验证集**选的; 对应的测试集精度仅供报告, 未参与选择。")
    with open(os.path.join(OUT_ROOT, f"optuna_{study_name}_best.json"), "w") as f:
        json.dump({"best_val": b.value, "best_params": b.params,
                   "test_acc_of_best": b.user_attrs.get("test_acc"),
                   "n_trials": len(study.trials)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
