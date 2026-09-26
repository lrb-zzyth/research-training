# 论文 FedTAD Table 2 原文数字（2026-09-25 从 references/fedtad.pdf 提取核验）

> 来源: `references/fedtad.pdf` 中 "Table 2: Performance comparison of test accuracy"
> 提取方式: pymupdf get_text 逐段对齐列名与数值, 每组 (数据集, 档位) 与 Handoff 中
> 已引用的 Cora/CiteSeer 数字完全一致, 可信。
> 本表只收录我们对比用的两行: FedTAD (Ours) 与 FedAvg。全表含 FedProx/SCAFFOLD/
> MOON/FedDC/GCFL+/Fed-PUB/FedSage+/FedGTA/FedTAD(10% Noise), 需要时再从 PDF 提取。

| 数据集 | 档位 | FedTAD (论文) | FedAvg (论文) |
|---|---|---|---|
| Cora | 5 | 85.1 ± 0.1 | 80.6 ± 0.3 |
| Cora | 10 | 75.3 ± 0.4 | 73.6 ± 0.4 |
| Cora | 20 | 61.3 ± 0.3 | 56.0 ± 0.3 |
| CiteSeer | 5 | 73.5 ± 0.3 | 71.5 ± 0.3 |
| CiteSeer | 10 | 71.7 ± 0.4 | 68.9 ± 0.2 |
| CiteSeer | 20 | 70.2 ± 0.3 | 66.3 ± 0.4 |
| PubMed | 5 | 87.9 ± 0.1 | 85.6 ± 0.3 |
| PubMed | 10 | 84.4 ± 0.4 | 82.9 ± 0.0 |
| PubMed | 20 | 83.5 ± 0.2 | 80.6 ± 0.3 |
| CS | 5 | 94.3 ± 0.4 | 90.4 ± 0.3 |
| CS | 10 | 90.2 ± 0.2 | 85.9 ± 0.8 |
| CS | 20 | 88.7 ± 0.4 | 83.9 ± 0.3 |
| Physics | 5 | 96.2 ± 0.2 | 94.7 ± 0.3 |
| Physics | 10 | 94.1 ± 0.2 | 91.6 ± 0.0 |
| Physics | 20 | 93.3 ± 0.3 | 90.3 ± 0.5 |

（论文还含第 6 个数据集 ogbn-arxiv: FedTAD 63.2/62.0/59.5, FedAvg 60.3/58.4/55.7。
本项目的目标数据集是上表 5 个, 不含 ogbn-arxiv。）

## 我方当前数字（fedtad_official 口径, 3 种子 {2024,2025,2026}, 2026-09-24 队列产物）

| 数据集 | 档位 | 我方 | Δ vs FedTAD | Δ vs FedAvg | 判定 |
|---|---|---|---|---|---|
| Cora | 5 | 84.87±0.22 | −0.23 | +4.27 | ❌ (噪声内) |
| Cora | 10 | 77.66±0.37 | +2.36 | +4.06 | ✅ |
| Cora | 20 | 69.10±0.65 | +7.80 | +13.10 | ✅ |
| CiteSeer | 5 | 72.33±0.13 | −1.17 | +0.83 | ❌ |
| CiteSeer | 10 | 72.92±0.71 | +1.22 | +4.02 | ✅ |
| CiteSeer | 20 | 70.07±0.13 | −0.13 | +3.77 | ❌ (噪声内) |
| CS / PubMed / Physics | — | 队列进行中 | — | — | ⏳ |

**vs FedAvg 6/6 全胜; vs FedTAD 3 胜 3 负**（赢的幅度大、输的幅度小）。

## 两个"输"的核心解释（详见 HANDOFF.md §11）

1. **调参档位效应**: `tune_then_run.py:37` `TUNE_TIER = 10` —— 只在 10 客户端档调参,
   5/20 档直接复用 10 档的最优超参。恰好 10 档在两个数据集上都是赢家。
2. **垃圾袋客户端**: 均衡分配算法把 Louvain 切块的残渣打包进个别客户端。
   CiteSeer-c5 client4 = 301 个连通分量 / 密度 0.0017 → test acc 63.7,
   以 1/5 权重把整档均值拉低 2.15 点; Cora-c5 client1 = 38 分量 → 71.3, 拉低 3.35 点。
   剔除该客户端后: CiteSeer-c5 ≈ 74.4 > 论文 73.5; Cora-c5 ≈ 88.2 > 论文 85.1。
   10 档的弱客户端同样存在(CiteSeer-c10 client9 = 156 分量 → 63.8), 但每个只占 1/10 权重。
