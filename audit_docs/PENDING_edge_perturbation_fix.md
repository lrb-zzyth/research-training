# 待应用修复: edge_perturbation() 无向性 bug（任务 B/C 定稿存档）

> 2026-09-25 定稿。**当前不要改代码** —— 用户决定: 等 TUNE3 队列全部跑完后应用,
> 然后只重跑 Cora 一个数据集做「改前 vs 改后」探测, 有效再推广全 5 数据集。

## 决策记录（用户 2026-09-25 拍板）

- ✅ 修复方案与代码已定稿（本文档）
- ✅ **每档（5/10/20）单独调参**，全部重跑 —— 用户 2026-09-25 拍板
  （根因: 旧队列 TUNE_TIER=10, 5/20 档复用 10 档超参 = 5 档负分原因之一）
- ⏸ 应用时机 = 旧队列之后；🔬 验证 = 先用 Cora 三档做「改前 vs 改后」探测，
  有提升或持平再推广
- 原因: 队列每 trial 都 subprocess 新起进程现场 import 仓库文件, 现在改会导致
  口径混杂；且重跑应在修复后代码上进行, 避免跑两遍

## 修复后总战役（per-tier campaign）

驱动脚本: `experiments/accuracy_benchmark/per_tier_campaign.py`（已就绪, 勿在修复前启动）
- 15 个 (数据集 × 档位) 组合各自独立 Optuna 调参（目标=验证集）+ 本档位 3 种子决赛
- Optuna study = tune5_<ds>_c<tier>（独立 DB optuna_tune5.db, 与 bug 版 tune4 隔离）
- 输出 = TUNED5_<ds>_c<tier>/seed_<s>（不覆盖 bug 版 TUNED_* 参考）
- 预算/档位与旧版一致: Cora 20 / CiteSeer 20 / CS 16 / PubMed 16 / Physics 10 trials
- 预计总时长 ≈ 6.2 天（Cora 11.5h + CiteSeer 11.5h + CS 38h + PubMed 24.7h + Physics 63h）

**探测与推广判定**: 修复后先跑 Cora 三档 (~11.5h)。对比口径: c10 vs c10 是严格匹配
（两边都是自己档位调参）; c5/c20 修复后 vs 修复前混有"调参效应", 仅作参考。
- 修复后 Cora c10 ≥ 修复前 −0.3 → 推广全量
- 若明显变差 (>0.5) → 停下向用户汇报, 不自行决定

## 背景（已实测确认）

- 客户端 `edge_index` 成对存储无向边（Cora client0: 788 有向条目 = 394 无向边, 100% 成对, 无自环）
- 旧实现按**有向条目**随机删 → 无向边可能被削成单向; 新边只加 `(i,j)` 不补 `(j,i)` → 第二视图部分有向
- 影响路径: `train_fedtad.py:1496` `_batched_subgraph_encode(model, data.x, aug_edge_index, ...)`
  直接对扰动图做 GCN 前向 → 方向性破坏第二视图消息传递（对比学习视图不对称）
- 唯一调用方: `train_fedtad.py:2336`（另有 test_smoke.py 测试 13/22, 断言有限性+梯度非零, 兼容修复）

## 行为变化（必须如实告知/记录）

旧代码按有向条目算 `num_drop = int(2E × drop_rate/2)`, 实际扰动 ~**drop_rate 比例**的无向边
（且削成单向）; 修复版 `num_drop = int(E × drop_rate/2)`, 扰动 **drop_rate/2 比例**的无向边
（完整双向）。即修复版与文档语义「删 ratio/2 + 补 ratio/2」一致, 扰动强度约为旧版一半。
对最终分数影响未知, 只能实测。

## 修复代码（替换 `util/task_util.py:106-134` 整个函数, 含任务 C 的 docstring）

```python
def edge_perturbation(edge_index, num_nodes, drop_rate=0.2):
    """
    边扰动图增强 (用于子图-子图跨视图对比的第二视图)。

    扰动语义: drop_rate = 总扰动比例, 其中 **删边 drop_rate/2 + 加边 drop_rate/2**。
      例: drop_rate=0.2 → 删除 10% 的无向边, 并新增 10% 的无向边。

    以**无向边**为扰动单位: 客户端图成对存储 (u,v)/(v,u),
    先 canonicalize 为 (min,max) 去重 (自环忽略, 客户端图构建时已 rm_self_loops),
    删/加都在无向边集合上进行, 最后展开回双向 —— 保证第二视图始终是合法的无向图。
    """
    device = edge_index.device
    percent = drop_rate / 2.0

    # 1) 折叠成无向边 (min,max) 去重
    undirected = set()
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if u != v:
            undirected.add((min(u, v), max(u, v)))
    undirected = sorted(undirected)
    E = len(undirected)

    # 2) 以无向边为单位删/加
    num_drop = int(E * percent)
    if num_drop <= 0:
        return edge_index.clone()

    drop_idx = set(random.sample(range(E), min(num_drop, E)))
    remaining = [e for i, e in enumerate(undirected) if i not in drop_idx]
    existing = set(undirected)

    added = []
    attempts = 0
    max_attempts = num_drop * 20
    while len(added) < num_drop and attempts < max_attempts:
        attempts += 1
        i, j = random.randrange(num_nodes), random.randrange(num_nodes)
        if i != j:
            e = (min(i, j), max(i, j))
            if e not in existing:
                added.append(e)
                existing.add(e)

    # 3) 展开回双向
    final = remaining + added
    src = [u for u, v in final] + [v for u, v in final]
    dst = [v for u, v in final] + [u for u, v in final]
    new_edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return new_edge_index
```

## 任务 C 的另外两处

1. **`train_fedtad.py:196` help 文本**（当前无 help）:
   ```python
   parser.add_argument('--edge_perturb_ratio', type=float, default=0.2,
                       help='边扰动总比例: 删除 ratio/2 的无向边 + 新增 ratio/2 的无向边 '
                            '(第二视图增强, 以无向边为单位)')
   ```
2. **审计文档补记**（`AUDIT_vs_original.md` D 节加一条）:
   专利文字「以概率 p_drop 删除、以相同概率 p_add 添加」读起来像各 20%,
   代码语义是「删 ratio/2 + 补 ratio/2」（ratio=0.2 → 各 10%）。
   这是专利表述与代码的语义差异, **只记录, 不改专利**。

## 应用后的验证步骤（严格串行）

1. `CUDA_VISIBLE_DEVICES='' /home/lrb/miniconda3/envs/fedtad5060/bin/python test_smoke.py`
   （纯合成张量, 几秒; 测试 13/22 覆盖扰动→对比损失路径; 另加断言: 输出边全部成对、无自环）
2. 冒烟训练 1 run（cl_loss 不塌缩, 与改前同 config 的 cl_loss 量级对比）
3. 重跑 Cora: Optuna 调参（目标=验证集）→ 5/10/20 三档 × 3 种子 {2024,2025,2026}
4. 与 PAPER_TABLE2.md 的「改前」数字对比 → 有提升或持平则推广 CS/CiteSeer/PubMed/Physics
