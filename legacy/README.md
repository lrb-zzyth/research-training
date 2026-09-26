# LEGACY CODE — DO NOT USE FOR FORMAL V3 RESULTS

> ⚠️ 本目录内的代码**不生成任何正式实验结果**。
> Formal V3 正式实验代码 = 仓库根目录 + 冻结 tag `formal-v3-f38f7fe74ab5`
> （formal code hash `f38f7fe74ab5`，见根目录 `FORMAL_V3_VERSION.md`）。

## 目录内容

| 路径 | 内容 | 说明 |
|---|---|---|
| `anomaly_platform/` | 旧 anomaly 检测 + Web 可视化平台开发线 | 与 Formal V3 multiclass campaign 是两条开发线；平台代码假设以仓库根为 cwd 运行，移入此处后如要再次运行需自行修路径 |
| `train_fedavg.py` | 早期 FedAvg 基线脚本 | 仅历史参考 |

## 与正式实验的关系

- 正式 15-cell 结果（Cora/CiteSeer/PubMed/CS/Physics × 5/10/20）全部由根目录
  Formal V3 代码在服务器上生成，与本目录无关；
- 本目录代码从未参与 `runs/formal_campaign_v3/` 的任何产物；
- 不要从这里复制任何训练脚本去"复现"正式结果。
