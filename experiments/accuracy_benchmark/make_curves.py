#!/usr/bin/env python3
"""
交付物 2: 训练过程曲线 (round vs test accuracy)。

数据源: 各 run 目录下的 events.jsonl 里 event=global_metric 的 global_test 字段。
        (metrics.jsonl 只有 val_primary, 不含 test)

产出: curves/<name>.png + curves/<name>.csv (同源数据, 一一对应)

用法: python experiments/accuracy_benchmark/make_curves.py
"""
import glob
import json
import os
import statistics as st
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体 (Noto Sans CJK), 否则中文标签会渲染成方块
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

# 与论文主表对照用的图: 我的方法 vs FedAvg (同口径, 均为本仓库自跑)
GROUPS = {
    "NR_fedavg": "FedAvg (自跑)",
    "NR_mymethod": "我的方法 (默认配置)",
    "SEN_tau0.2": "我的方法 (τ=0.2, 调参后)",
}
# 额外参考曲线: 原版代码跑的 FedAvg / FedTAD (不同管线, 仅供形态对比)
ORIG_LOGS = {
    "原版 FedAvg (原版代码)": "runs/accuracy_control/fedavg_orig/stdout.log",
    "原版 FedTAD (原版代码)": "runs/accuracy_control/fedtad_orig/stdout.log",
}


def read_group_curves(group):
    """返回 {round: [test_acc per seed]}"""
    curves = {}
    for ev in sorted(glob.glob(os.path.join(ROOT, group, "seed_*", "events.jsonl"))):
        for line in open(ev):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "global_metric" and d.get("round") is not None:
                if d.get("global_test") is not None:
                    curves.setdefault(int(d["round"]), []).append(float(d["global_test"]))
    return curves


def read_orig_curve(path):
    import re
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        m = re.search(r"current_round: (\d+)\s+global_val: [\d.]+\s+global_test: ([\d.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for g, label in GROUPS.items():
        c = read_group_curves(g)
        if c:
            data[label] = c
            print(f"{label}: {len(c)} 轮, {max(len(v) for v in c.values())} seeds")

    # ---- 图 1: 我的方法 vs FedAvg, 同口径 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for label, c in data.items():
        rs = sorted(c)
        mean = [st.mean(c[r]) for r in rs]
        ax.plot(rs, mean, label=label, linewidth=2)
        if any(len(c[r]) > 1 for r in rs):
            lo = [min(c[r]) for r in rs]
            hi = [max(c[r]) for r in rs]
            ax.fill_between(rs, lo, hi, alpha=0.18)
        for r in rs:
            rows.append({"method": label, "round": r,
                         "test_acc_mean": round(st.mean(c[r]), 4),
                         "n_seeds": len(c[r]),
                         "test_acc_all": " ".join(f"{v:.2f}" for v in c[r])})
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Cora-10 (Louvain, 与原版同一套划分) — no-resplit 口径")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curve_main.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(OUT, "curve_main.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "round", "test_acc_mean", "n_seeds", "test_acc_all"])
        w.writeheader()
        w.writerows(rows)
    print(f"已写入 {OUT}/curve_main.png + curve_main.csv")

    # ---- 图 2: 我的方法 vs 原版 FedAvg / 原版 FedTAD (参考) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    rows2 = []
    for label, c in data.items():
        rs = sorted(c)
        ax.plot(rs, [st.mean(c[r]) for r in rs], label=label + " [本仓库]", linewidth=2)
    for label, path in ORIG_LOGS.items():
        c = read_orig_curve(path)
        if c:
            rs = sorted(c)
            ax.plot(rs, [c[r] for r in rs], label=label + " [原版代码]", linewidth=2, linestyle="--")
            for r in rs:
                rows2.append({"method": label, "round": r, "test_acc": c[r]})
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Cora-10 — 收敛速度对比 (注: 原版代码管线与我们会话不同, 仅供参考形态)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curve_vs_original.png"), dpi=150)
    plt.close(fig)
    if rows2:
        with open(os.path.join(OUT, "curve_vs_original.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "round", "test_acc"])
            w.writeheader()
            w.writerows(rows2)
    print(f"已写入 {OUT}/curve_vs_original.png + curve_vs_original.csv")

    # ---- 收敛速度结论: 达到最终精度 99% 所需轮次 ----
    print("\n收敛速度 (达到各方法最终均值 99% 所需轮次):")
    for label, c in data.items():
        rs = sorted(c)
        mean = [st.mean(c[r]) for r in rs]
        final = st.mean(mean[-10:])
        target = 0.99 * final
        hit = next((r for r, v in zip(rs, mean) if v >= target), None)
        print(f"  {label:<22} 最终≈{final:.2f}%  达 99% 于 round {hit}")


if __name__ == "__main__":
    main()
