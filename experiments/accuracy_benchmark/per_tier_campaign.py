#!/usr/bin/env python3
"""
每 (数据集, 档位) 独立调参 + 3 种子决赛 (per-tier campaign)。

与 tune_then_run.py 的区别:
  - 调参不再固定在 10 客户端档: 5/10/20 每个档位**单独**跑 Optuna (目标=验证集精度, 严禁用测试集选参)
  - 最优参数只用于本档位的 3 种子决赛 (2024/2025/2026 固定种子集, 不按结果挑选)
  - Optuna study 用 tune5_<ds>_c<tier> (独立 DB: optuna_tune5.db), 与旧 tune4
    (TUNE_TIER=10 + edge_perturbation bug 版代码) 完全隔离, 不混 trials
  - 输出目录 TUNED5_<ds>_c<tier>/seed_<s>, 不覆盖旧的 TUNED_* (bug 版参考) 结果

⚠️ 必须在 edge_perturbation 无向性修复 (runs/accuracy_benchmark/PENDING_edge_perturbation_fix.md)
   应用并通过冒烟之后启动。
严格串行: 复用 run_one 的 wait_no_train / mem_ok 安全闸 (本机 7GB 内存, 单 run 峰值 3.6GB)。

用法:
  python experiments/accuracy_benchmark/per_tier_campaign.py                      # 全量 15 组合 (CELL_ORDER 顺序)
  python experiments/accuracy_benchmark/per_tier_campaign.py --cells Cora:5       # 只跑指定组合

预算 (每数据集每档位, 与旧版一致): Cora 20, CiteSeer 20, CS 16, PubMed 16, Physics 10 trials。
预计总时长 (按旧版单 run 耗时): Cora 11.5h + CiteSeer 11.5h + CS 38h + PubMed 24.7h + Physics 63h ≈ 6.2 天。
"""
import argparse
import json
import os
import sys

import optuna

from tune_then_run import (REPO, PY, OUT, SEEDS, BUDGET, suggest, run_one)

ORDER = ["Cora", "CiteSeer", "CS", "PubMed", "Physics"]
TIERS = [5, 10, 20]

# 跑序 (用户 2026-09-25 拍板: 全部重跑; 已经调好参跑好的档位放最后)
# "已调好参" = 旧协议下在自己档位上调的参: 仅 Cora c10 / CiteSeer c10。
# 其余 13 组合 (含 Cora/CiteSeer 的 5/20 档 —— 旧数字是借 10 档参数跑的, 不算) 先行。
CELL_ORDER = [
    ("Cora", 5), ("Cora", 20),
    ("CiteSeer", 5), ("CiteSeer", 20),
    ("CS", 5), ("CS", 10), ("CS", 20),
    ("PubMed", 5), ("PubMed", 10), ("PubMed", 20),
    ("Physics", 5), ("Physics", 10), ("Physics", 20),
    # ---- 已经调好参跑好的档位, 最后在修复版代码上重跑 ----
    ("Cora", 10), ("CiteSeer", 10),
]


def tune_tier(ds, tier, n_trials):
    storage = f"sqlite:///{os.path.join(OUT, 'optuna_tune5.db')}"
    study = optuna.create_study(
        study_name=f"tune5_{ds}_c{tier}", storage=storage,
        load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2024))

    def objective(trial):
        params = suggest(trial)
        d = os.path.join(OUT, f"TUNE5_{ds}_c{tier}", f"trial_{trial.number:03d}")
        print(f"  [Trial {trial.number}] {params}", flush=True)
        r = run_one(ds, tier, 2024, d, extra=list(params.items()))
        if r is None:
            raise optuna.TrialPruned()
        val, acc, wall = r
        print(f"    -> val={val:.2f} test={acc:.2f} wall={wall:.0f}s", flush=True)
        trial.set_user_attr("test_acc", acc)
        return val

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    b = study.best_trial
    print(f"  [{ds} c{tier}] 最优 (按验证集): val={b.value:.2f} params={b.params}",
          flush=True)
    print(f"         该 trial 的 test={b.user_attrs.get('test_acc')}", flush=True)
    return dict(b.params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None,
                    help='逗号分隔 "数据集:档位" 列表, 默认按 CELL_ORDER '
                         '(已调好参的档位最后)。例: --cells Cora:5,Cora:20')
    args = ap.parse_args()
    if args.cells:
        cells = [tuple(c.strip().split(":")) for c in args.cells.split(",")]
        cells = [(ds, int(t)) for ds, t in cells]
    else:
        cells = CELL_ORDER

    for ds, tier in cells:
        print("\n" + "#" * 70, flush=True)
        print(f"#  {ds} c{tier}: 调参 {BUDGET[ds]} trials + {len(SEEDS)} 种子决赛",
              flush=True)
        print("#" * 70, flush=True)
        best = tune_tier(ds, tier, BUDGET[ds])
        with open(os.path.join(OUT, f"tuned5_params_{ds}_c{tier}.json"), "w") as f:
            json.dump(best, f, indent=1)
        for s in SEEDS:
            d = os.path.join(OUT, f"TUNED5_{ds}_c{tier}", f"seed_{s}")
            print(f"  [{ds} c{tier} seed{s}] 开始", flush=True)
            r = run_one(ds, tier, s, d, extra=list(best.items()))
            if r:
                print(f"    -> val={r[0]:.2f} test={r[1]:.2f} wall={r[2]:.0f}s",
                      flush=True)
    print("\nALL_PERTIER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
