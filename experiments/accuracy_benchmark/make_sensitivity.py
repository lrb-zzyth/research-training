#!/usr/bin/env python3
"""
交付物 3: 参数敏感度分析。

数据源: runs/accuracy_benchmark/<组名>/seed_*/final_metrics.json
       组名约定 SEN_<参数><取值>, 例如 SEN_lamsg0.01 / SEN_tau0.2
默认值(基线)取自主实验 NR_mymethod, 在图上用竖虚线标出。

产出: curves/sens_<参数>.png + <参数>.csv, 以及 sensitivity.md

用法: python experiments/accuracy_benchmark/make_sensitivity.py
"""
import glob
import json
import os
import re
import statistics as st
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _p in ("/home/lrb/.fonts/NotoSansCJK-Regular.ttc",
           "/home/lrb/.fonts/NotoSansCJK-Medium.ttc"):
    try:
        font_manager.fontManager.addfont(_p)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "runs/accuracy_benchmark"
OUT = os.path.join(ROOT, "curves")

# 参数 -> (组名前缀, 基线值, 中文名, 是否对数轴)
PARAMS = {
    "lambda_subgraph": ("SEN_lamsg", 0.1, "对比损失权重 λ_subgraph", True),
    "contrastive_temperature": ("SEN_tau", 0.5, "对比温度 τ", True),
}


def read_group(group):
    accs = []
    for p in glob.glob(os.path.join(ROOT, group, "seed_*", "final_metrics.json")):
        try:
            d = json.load(open(p))
            a = d["metrics"]["pooled"]["accuracy"]
            if a is not None:
                accs.append(float(a))
        except Exception:
            continue
    return accs


def baseline():
    return read_group("NR_mymethod")


def main():
    os.makedirs(OUT, exist_ok=True)
    base = baseline()
    base_mean = st.mean(base) if base else None
    base_std = st.stdev(base) if len(base) > 1 else 0.0

    md = ["# 参数敏感度分析\n",
          f"\n> 每点 3 个种子, 误差棒 = 标准差。默认取值用竖虚线标出"
          f"（取自 `NR_mymethod`, {base_mean:.2f}±{base_std:.2f}, n={len(base)}）。\n",
          "\n> **受硬件限制（7GB 内存, 严格串行, 单 run ≈6.5 分钟）, 本次仅覆盖 2 个参数、每个 3 个取值。**\n"
          "> 原任务书要求的 ≥6 张敏感度图未能完成, 此为如实报告的限制。\n"]

    for param, (prefix, default, label, logx) in PARAMS.items():
        groups = sorted(glob.glob(os.path.join(ROOT, prefix + "*")))
        pts = []
        for g in groups:
            val_s = os.path.basename(g)[len(prefix):]
            try:
                val = float(val_s)
            except ValueError:
                continue
            accs = read_group(os.path.basename(g))
            if accs:
                pts.append((val, st.mean(accs), st.stdev(accs) if len(accs) > 1 else 0.0, len(accs)))
        if base_mean is not None:
            pts.append((default, base_mean, base_std, len(base)))
        if not pts:
            print(f"[skip] {param}: 没有数据")
            continue
        pts.sort()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=5, linewidth=2, markersize=7)
        ax.axvline(default, linestyle="--", color="gray", alpha=0.7,
                   label=f"默认值 {default}")
        if base_mean is not None:
            ax.axhline(base_mean, linestyle=":", color="tab:blue", alpha=0.6,
                       label=f"我的方法(默认) {base_mean:.2f}")
        ax.set_xlabel(label + (" (对数轴)" if logx else ""))
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"参数敏感度 — {label}")
        if logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        png = os.path.join(OUT, f"sens_{param}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)

        cpath = os.path.join(OUT, f"sens_{param}.csv")
        with open(cpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["value", "test_acc_mean", "test_acc_std", "n_seeds", "is_default"])
            for v, m, s, n in pts:
                w.writerow([v, f"{m:.4f}", f"{s:.4f}", n, int(v == default)])
        print(f"已写入 {png} + {cpath}")

        md.append(f"\n## {label}\n\n")
        md.append("| 取值 | test acc (mean±std) | 种子数 |\n|---|---|---|\n")
        for v, m, s, n in pts:
            mark = " **(默认)**" if v == default else ""
            md.append(f"| {v}{mark} | {m:.2f} ± {s:.2f} | {n} |\n")
        best = max(pts, key=lambda p: p[1])
        md.append(f"\n最优取值：**{best[0]}**（{best[1]:.2f}）\n")

    path = os.path.join(ROOT, "sensitivity.md")
    with open(path, "w") as f:
        f.writelines(md)
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()
