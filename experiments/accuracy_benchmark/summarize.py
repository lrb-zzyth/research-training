#!/usr/bin/env python3
"""
accuracy_benchmark 结果汇总器。

用法:
    python experiments/accuracy_benchmark/summarize.py <run_dir> [<run_dir> ...]
    python experiments/accuracy_benchmark/summarize.py --table runs/accuracy_*/*/seed_*

功能:
    - 扫描 run 目录下的 final_metrics.json / metrics.jsonl
    - 按 "实验组名" 聚合多种子 -> mean±std
    - 生成 main_table.md / main_table.csv / curves/*.csv

诚实性约束(任务书): 所有数字必须可追溯到 runs/ 下的具体文件路径, 本脚本在输出里附上路径。
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys


def load_run(d):
    """读取单个 run 目录, 返回 dict 或 None。"""
    fm = os.path.join(d, "final_metrics.json")
    if not os.path.exists(fm):
        return None
    try:
        with open(fm) as f:
            data = json.load(f)
    except Exception:
        return None
    m = data.get("metrics", {})
    pooled = m.get("pooled", {}) or {}
    per_client = m.get("per_client", {}) or {}
    accs = [v.get("accuracy") for v in per_client.values()
            if isinstance(v, dict) and v.get("accuracy") is not None]
    seed_m = re.search(r"seed_(\d+)", d)
    return {
        "dir": d,
        "group": os.path.basename(os.path.dirname(d.rstrip("/"))),
        "seed": int(seed_m.group(1)) if seed_m else None,
        "task_mode": data.get("task_mode"),
        "best_round": data.get("best_round"),
        "acc": pooled.get("accuracy"),
        "macro_f1": pooled.get("macro_f1"),
        "worst_client": min(accs) if accs else None,
        "per_client": accs,
        "config_hash": data.get("config_hash"),
    }


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--table", action="store_true",
                    help="输出 markdown 主表 + csv")
    ap.add_argument("--out", default="runs/accuracy_benchmark")
    args = ap.parse_args()

    dirs = []
    for pat in args.dirs:
        dirs.extend(sorted(glob.glob(pat)))
    runs = [r for r in (load_run(d) for d in dirs) if r]
    if not runs:
        print("没有找到任何 final_metrics.json", file=sys.stderr)
        return 1

    groups = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)

    print(f"扫描到 {len(runs)} 个完成的 run, {len(groups)} 个实验组\n")
    print(f"{'实验组':<34}{'n':>3}{'acc mean±std':>18}{'worst':>9}{'best_round':>12}")
    print("-" * 78)
    rows = []
    for g in sorted(groups):
        rs = sorted(groups[g], key=lambda x: (x["seed"] is None, x["seed"]))
        accs = [r["acc"] for r in rs if r["acc"] is not None]
        worsts = [r["worst_client"] for r in rs if r["worst_client"] is not None]
        rounds = [r["best_round"] for r in rs if r["best_round"] is not None]
        if not accs:
            continue
        mean = st.mean(accs)
        std = st.stdev(accs) if len(accs) > 1 else 0.0
        rows.append({
            "group": g, "n": len(accs), "mean": mean, "std": std,
            "worst_mean": st.mean(worsts) if worsts else None,
            "round_mean": st.mean(rounds) if rounds else None,
            "accs": accs,
            "dirs": [r["dir"] for r in rs],
        })
        print(f"{g:<34}{len(accs):>3}{f'{mean:.2f}±{std:.2f}':>18}"
              f"{fmt(st.mean(worsts) if worsts else None):>9}"
              f"{fmt(st.mean(rounds) if rounds else None, 1):>12}")

    if args.table:
        os.makedirs(args.out, exist_ok=True)
        md = os.path.join(args.out, "main_table.md")
        cv = os.path.join(args.out, "main_table.csv")
        with open(md, "w") as f:
            f.write("# 主结果表 (Cora-10, multiclass, test accuracy %)\n\n")
            f.write("| 实验组 | 种子数 | test acc (mean±std) | worst-client | 最佳轮 | 产物路径 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in sorted(rows, key=lambda x: -x["mean"]):
                f.write(f"| {r['group']} | {r['n']} | "
                        f"**{r['mean']:.2f}±{r['std']:.2f}** | "
                        f"{fmt(r['worst_mean'])} | {fmt(r['round_mean'],1)} | "
                        f"`{r['dirs'][0].rsplit('/seed_',1)[0]}` |\n")
            f.write("\n> 所有数字由 `experiments/accuracy_benchmark/summarize.py` "
                    "从各 run 的 `final_metrics.json` 直接读取, 未做任何挑选。\n")
        with open(cv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "n_seeds", "acc_mean", "acc_std",
                        "worst_client_mean", "best_round_mean", "seeds_acc", "dirs"])
            for r in sorted(rows, key=lambda x: -x["mean"]):
                w.writerow([r["group"], r["n"], f"{r['mean']:.4f}", f"{r['std']:.4f}",
                            fmt(r["worst_mean"], 4), fmt(r["round_mean"], 1),
                            " ".join(f"{a:.2f}" for a in r["accs"]),
                            ";".join(r["dirs"])])
        print(f"\n已写入 {md}\n已写入 {cv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
