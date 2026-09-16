"""
Dynamic CKR Tracker

管理静态拓扑先验与动态模型性能的混合可靠性权重。
使用指数滑动平均 (EMA) 平滑动态变化。

公式:
  R_static_scaled[k,c] = scale(static_ckr[k,c]) ∈ [0,1]
  R_dynamic[k,c,t] = α · R_static_scaled[k,c] + (1-α) · metric_local[k,c,t]   (hybrid_dynamic)
                     metric_local[k,c,t]                                       (dynamic_only)
                     R_static_scaled[k,c]                                      (static_topology)
  R_ema[k,c,t] = γ · R_ema[k,c,t-1] + (1-γ) · R_dynamic[k,c,t]    (首轮直接取 candidate)

服务端权重: R_server[k,c] = R_ema[k,c] / Σⱼ R_ema[j,c]   (按类别在客户端维度归一化)

回退规则 (available=False):
  - 已有历史: 保持上一轮 EMA
  - 无历史:   hybrid/static -> 静态拓扑先验;  dynamic_only -> 均匀权重
  - 不得把 unavailable 当 F1=0, 不得产生 NaN
"""
import os
import json
import math
import torch


def scale_static_ckr(static_ckr, method='max'):
    """
    将原始静态 CKR 缩放到 [0, 1]。

    Args:
        static_ckr: [num_clients, num_classes]
        method: 'max' | 'minmax' | 'rank'

    Returns:
        scaled: [num_clients, num_classes] 取值 [0,1]
    """
    eps = 1e-8
    num_clients, num_classes = static_ckr.shape
    scaled = torch.zeros_like(static_ckr)
    warned = False

    for c in range(num_classes):
        col = static_ckr[:, c]
        if method == 'max':
            col_max = col.max()
            if col_max > eps:
                scaled[:, c] = col.clamp(min=0) / col_max
            else:
                scaled[:, c] = 1.0 / num_clients
                warned = True
        elif method == 'minmax':
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > eps:
                scaled[:, c] = (col - col_min) / (col_max - col_min)
            else:
                scaled[:, c] = 1.0 / num_clients
                warned = True
        elif method == 'rank':
            ranks = torch.argsort(torch.argsort(col)).float()
            scaled[:, c] = ranks / (num_clients - 1) if num_clients > 1 else torch.ones_like(ranks)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    if warned:
        print(f"  [CKR] ⚠ 存在全零类别, 该类别使用均匀先验 (1/{num_clients})")

    if torch.isnan(scaled).any() or torch.isinf(scaled).any():
        print(f"  [CKR] ⚠ scaled has NaN/Inf, falling back to uniform")
        scaled = torch.ones_like(static_ckr) / num_clients

    return scaled


def compute_per_class_metrics(model, data, eval_idx, num_classes, min_support=3, metric='f1'):
    """
    计算每类别动态可靠性指标 (只读, 不执行 backward)。

    Args:
        model: GCN (内部 eval + no_grad)
        data: PyG Data
        eval_idx: bool mask [N] 或 LongTensor 索引 (reliability_idx 或 val_idx)
        num_classes: int
        min_support: 类别最小样本数阈值, 低于该值标记 unavailable
        metric: 'f1' | 'recall' | 'confidence' (决定 tracker 使用的动态指标)

    Returns:
        dict:
            per_class_f1: [C]          one-vs-rest 类别 F1 (缺失类别 NaN)
            per_class_precision: [C]
            per_class_recall: [C]
            per_class_confidence: [C]  真实类别 c 的可靠性节点上, 模型对 c 的平均 softmax 概率
            per_class_support: [C]
            available_mask: [C] bool   support >= min_support 才为 True
    """
    device = data.x.device
    model.eval()
    with torch.no_grad():
        logits = model.forward(data)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

    if eval_idx.dtype == torch.bool:
        eval_idx_t = eval_idx
    else:
        eval_idx_t = torch.zeros(data.x.shape[0], device=device, dtype=torch.bool)
        eval_idx_t[eval_idx] = True

    y_true = data.y[eval_idx_t]
    y_pred = pred[eval_idx_t]
    y_prob = probs[eval_idx_t]

    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []
    per_class_confidence = []
    per_class_support = []
    available_mask = []

    for c in range(num_classes):
        support = int((y_true == c).sum().item())
        per_class_support.append(support)

        if support < min_support:
            # 缺失/支持不足: NaN + available=False (不得静默置零)
            per_class_f1.append(float('nan'))
            per_class_precision.append(float('nan'))
            per_class_recall.append(float('nan'))
            per_class_confidence.append(float('nan'))
            available_mask.append(False)
            continue

        tp = int(((y_pred == c) & (y_true == c)).sum().item())
        fp = int(((y_pred == c) & (y_true != c)).sum().item())
        fn = int(((y_pred != c) & (y_true == c)).sum().item())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1)

        # Confidence: 真实类别 c 的节点上, 模型对类别 c 的平均 softmax 概率
        class_mask = y_true == c
        conf = float(y_prob[class_mask, c].mean().item()) if class_mask.sum() > 0 else float('nan')
        per_class_confidence.append(conf)
        available_mask.append(True)

    return {
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_confidence': per_class_confidence,
        'per_class_support': per_class_support,
        'available_mask': available_mask,
    }


# 规格要求的正式函数名 (与 compute_per_class_metrics 等价)
compute_per_class_reliability_metrics = compute_per_class_metrics


class DynamicCKRTracker:
    """
    动态 CKR 跟踪器。

    模式:
      - static_topology: 仅静态拓扑 CKR (消融基线)
      - dynamic_only:    仅动态指标 (缺失回退上一轮 EMA / 均匀权重)
      - hybrid_dynamic:  静态先验 + 动态指标 + EMA (默认)
    """

    def __init__(self, num_clients, num_classes, mode='hybrid_dynamic',
                 static_scaling='max', alpha=0.5, ema_decay=0.8,
                 metric='f1', min_support=3, log_dir=None):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.mode = mode
        self.static_scaling = static_scaling
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.metric = metric
        self.min_support = min_support
        self.log_dir = log_dir

        self.round = 0
        self.static_ckr_raw = None
        self.static_ckr_scaled = None
        self.current_ema = None          # [num_clients, num_classes]
        self.last_dynamic_metric = None
        self.last_available_mask = None
        self.last_candidate = None
        self.last_fallback = None
        self.last_fallback_reason = None
        self.seen = None                 # bool mask: 该 (k,c) 是否已有动态历史
        self.per_class_support = {}      # {client_id: [C]}  reliability/val support
        self.per_client_fit_support = {}  # {client_id: [C]}  fit support
        self.history = []                # 每轮日志记录 (dict 列表)
        self.last_server_weights = None  # 本轮归一化后的服务端权重
        self.last_delta = None           # [K,C] ema_t - ema_{t-1}
        self.aggregate_history = []      # 每轮聚合 {round: {...}}

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def initialize(self, static_ckr_raw):
        """用静态拓扑 CKR 初始化。"""
        self.static_ckr_raw = static_ckr_raw.clone()
        self.static_ckr_scaled = scale_static_ckr(static_ckr_raw, self.static_scaling)
        # 初始 EMA = 静态缩放值 (首轮回退目标)
        self.current_ema = self.static_ckr_scaled.clone()
        self.last_dynamic_metric = torch.zeros_like(self.static_ckr_scaled)
        self.last_available_mask = torch.zeros(self.num_clients, self.num_classes, dtype=torch.bool)
        self.last_candidate = self.static_ckr_scaled.clone()
        self.last_fallback = torch.zeros(self.num_clients, self.num_classes, dtype=torch.bool)
        self.last_fallback_reason = [[''] * self.num_classes for _ in range(self.num_clients)]
        self.seen = torch.zeros(self.num_clients, self.num_classes, dtype=torch.bool)
        self.per_class_support = {}
        self.per_client_fit_support = {}
        self.history = []
        self.last_delta = torch.zeros_like(self.static_ckr_scaled)
        self.aggregate_history = []
        self.round = 0

    # ------------------------------------------------------------------
    def _compute_candidate(self, ci, c, dynamic_val, available):
        """根据模式计算 candidate 可靠性。"""
        static_s = self.static_ckr_scaled[ci, c]
        if self.mode == 'static_topology':
            return static_s.item()
        if not available:
            # unavailable 回退: 已有历史 -> 上一轮 EMA; 无历史 -> 静态先验 / 均匀
            if self.seen[ci, c]:
                return self.current_ema[ci, c].item()
            if self.mode == 'dynamic_only':
                return 1.0 / self.num_clients
            return static_s.item()
        # available
        if self.mode == 'dynamic_only':
            return float(dynamic_val)
        # hybrid_dynamic
        return self.alpha * static_s.item() + (1 - self.alpha) * float(dynamic_val)

    def update(self, client_ids, per_client_metrics, round_idx,
               per_client_fit_support=None):
        """
        更新动态 CKR (本地训练完成之后调用)。

        Args:
            client_ids: list[int] 参与本轮更新的客户端
            per_client_metrics: dict {client_id: compute_per_class_metrics 的输出}
            round_idx: int 通信轮次 (从 1 开始)
            per_client_fit_support: dict {client_id: [C]} fit_idx support (可观测性)
        """
        self.round = round_idx
        metric_key = f'per_class_{self.metric}'
        c = self.num_classes

        if per_client_fit_support is not None:
            self.per_client_fit_support = per_client_fit_support

        for ci in client_ids:
            if ci not in per_client_metrics:
                continue
            pm = per_client_metrics[ci]
            dynamic_vals = torch.as_tensor(pm[metric_key], dtype=torch.float)
            available = torch.as_tensor(pm['available_mask'], dtype=torch.bool)
            support = list(pm['per_class_support'])
            self.per_class_support[ci] = support

            if torch.isnan(dynamic_vals).any():
                # NaN 指标不得当作 0: 标记为 unavailable 走回退
                available = available & ~torch.isnan(dynamic_vals)

            for j in range(c):
                prev_ema = self.current_ema[ci, j].item()
                dyn_val = dynamic_vals[j].item()
                avail = bool(available[j])

                candidate = self._compute_candidate(ci, j, dyn_val, avail)

                if self.mode == 'static_topology':
                    # 消融基线: 永远静态
                    new_ema = self.static_ckr_scaled[ci, j].item()
                    fallback, reason = True, 'static_mode'
                elif avail:
                    new_ema = candidate
                    if self.seen[ci, j]:
                        new_ema = (self.ema_decay * prev_ema
                                   + (1 - self.ema_decay) * candidate)
                    self.seen[ci, j] = True
                    fallback, reason = False, ''
                else:
                    # unavailable: 回退, 不更新 EMA 值本身
                    reason = self._fallback_reason(ci, j, dyn_val)  # 先算原因 (seen 未变)
                    new_ema = prev_ema
                    if not self.seen[ci, j]:
                        # 首轮无历史: 回退到静态先验 (或 dynamic_only 均匀)
                        new_ema = candidate  # candidate 已按模式回退
                        if self.mode == 'dynamic_only':
                            self.seen[ci, j] = True  # 均匀先验作为历史起点
                    fallback = True

                self.current_ema[ci, j] = new_ema
                self.last_candidate[ci, j] = candidate
                self.last_fallback[ci, j] = fallback
                self.last_fallback_reason[ci][j] = reason
                self.last_delta[ci, j] = new_ema - prev_ema

            self.last_dynamic_metric[ci] = dynamic_vals
            self.last_available_mask[ci] = available

        # 本轮服务端权重 (按类别在参与客户端维度归一化)
        self.last_server_weights = self.get_server_weights(selected_client_ids=client_ids)
        self._log_step(client_ids, per_client_metrics)
        self.aggregate_history.append(self.round_aggregate(client_ids))

    def _fallback_reason(self, ci, j, dyn_val):
        if self.seen[ci, j]:
            return 'previous_ema'
        target = ('first_round_uniform' if self.mode == 'dynamic_only'
                  else 'first_round_static')
        if math.isnan(dyn_val):
            return f'nan_metric_{target}'
        return target

    # ------------------------------------------------------------------
    def get_server_weights(self, selected_client_ids=None):
        """
        返回按类别归一化的服务端权重 R_server[k,c] = R_ema[k,c] / Σⱼ R_ema[j,c]。

        未参与客户端的权重为 0, 归一化在参与客户端维度进行。
        类别全零时回退均匀权重, 不允许 NaN/Inf。
        """
        if self.current_ema is None:
            raise RuntimeError('DynamicCKRTracker not initialized')

        if selected_client_ids is None:
            weights = self.current_ema.clone()
        else:
            weights = torch.zeros_like(self.current_ema)
            for ci in selected_client_ids:
                weights[ci] = self.current_ema[ci]

        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("  [DynamicCKR] ⚠ EMA 含 NaN/Inf, 对应列回退均匀权重")
            bad = torch.isnan(weights) | torch.isinf(weights)
            weights[bad] = 1.0 / self.num_clients

        denom = weights.sum(dim=0, keepdim=True)
        zero_mask = denom.squeeze() < 1e-8
        if zero_mask.any():
            for j in torch.where(zero_mask)[0].tolist():
                weights[:, j] = 1.0 / self.num_clients
                print(f"  [DynamicCKR] ⚠ 类别 {j} 总权重为 0, 均匀权重回退")
        else:
            weights = weights / denom

        return weights

    # ------------------------------------------------------------------
    def get_log_summary(self, client_ids, per_client_metrics):
        """每轮打印摘要: [Dynamic CKR][Round t][Client k] 每个类别明细。"""
        lines = [f"[Dynamic CKR][Round {self.round}]"]
        for ci in client_ids:
            if ci not in per_client_metrics:
                continue
            pm = per_client_metrics[ci]
            lines.append(f"  [Client {ci}] mode={self.mode}")
            for j in range(self.num_classes):
                avail = bool(pm['available_mask'][j])
                support = pm['per_class_support'][j]
                dyn = pm[f'per_class_{self.metric}'][j]
                dyn_s = 'nan' if (isinstance(dyn, float) and math.isnan(dyn)) else f'{dyn:.4f}'
                lines.append(
                    f"    class {j}: static_raw={self.static_ckr_raw[ci,j].item():.4f} "
                    f"static_scaled={self.static_ckr_scaled[ci,j].item():.4f} "
                    f"support={support} {self.metric}={dyn_s} available={avail} "
                    f"candidate={self.last_candidate[ci,j].item():.4f} "
                    f"ema={self.current_ema[ci,j].item():.4f} "
                    f"server_w={self.last_server_weights[ci,j].item():.4f} "
                    f"fallback={self.last_fallback[ci,j]}"
                    + (f" ({self.last_fallback_reason[ci][j]})" if self.last_fallback[ci][j] else '')
                )
        return '\n'.join(lines)

    def anomaly_binary_summary(self, normal_classes):
        """anomaly_binary 模式额外输出: normal/anomaly CKR, anomaly support, zero-anomaly 标志。"""
        lines = ["  [Anomaly CKR]"]

        for ci in range(self.num_clients):
            support = self.per_class_support.get(ci, [0] * self.num_classes)
            anomaly_support = support[1] if len(support) > 1 else 0
            zero_split = (support[1] == 0) if len(support) > 1 else True
            normal_val = self.current_ema[ci, 0].item() if self.num_classes > 1 \
                else self.current_ema[ci, 0].item()
            anomaly_val = self.current_ema[ci, 1].item() if self.num_classes > 1 else float('nan')
            lines.append(
                f"    client {ci}: normal_ckr={normal_val:.4f} anomaly_ckr={anomaly_val:.4f} "
                f"anomaly_reliability_support={anomaly_support} "
                f"zero_anomaly_reliability_split={zero_split}"
            )
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _cell_status(self, ci, j, available):
        if available:
            return 'available'
        reason = self.last_fallback_reason[ci][j]
        if reason in ('previous_ema',):
            return 'fallback_ema'
        return 'fallback_static'

    def round_aggregate(self, client_ids):
        """每轮聚合统计 (写入 ckr_availability.json)。"""
        K = self.num_clients
        C = self.num_classes
        total = K * C
        available = int(self.last_available_mask.float().sum().item())
        fallback = int(self.last_fallback.float().sum().item())
        changed = int((self.last_delta.abs() > 1e-9).float().sum().item())
        deltas = self.last_delta.abs().flatten().tolist()
        mean_abs = float(sum(deltas) / len(deltas)) if deltas else 0.0
        sorted_d = sorted(deltas)
        median_abs = float(sorted_d[len(sorted_d) // 2]) if sorted_d else 0.0
        per_class_avail = [float(self.last_available_mask[:, j].float().mean().item())
                           for j in range(C)]
        per_client_avail = [float(self.last_available_mask[k, :].float().mean().item())
                            for k in range(K)]
        return {
            'round': self.round,
            'total_client_class_cells': total,
            'available_cells': available,
            'fallback_cells': fallback,
            'unavailable_cells': total - available,
            'available_ratio': available / total,
            'fallback_ratio': fallback / total,
            'changed_cells': changed,
            'changed_ratio': changed / total,
            'mean_abs_delta': mean_abs,
            'median_abs_delta': median_abs,
            'max_abs_delta': max(deltas) if deltas else 0.0,
            'per_class_available_ratio': per_class_avail,
            'per_client_available_ratio': per_client_avail,
        }

    def _log_step(self, client_ids, per_client_metrics):
        """记录 JSONL 日志 (完整可观测性字段)。"""
        if self.log_dir:
            log_path = os.path.join(self.log_dir, 'dynamic_ckr_history.jsonl')
        else:
            log_path = None

        fit_support = self.per_client_fit_support
        records = []
        for ci in client_ids:
            if ci not in per_client_metrics:
                continue
            pm = per_client_metrics[ci]
            for j in range(self.num_classes):
                dyn = pm[f'per_class_{self.metric}'][j]
                avail = bool(pm['available_mask'][j])
                record = {
                    'round': self.round,
                    'client_id': ci,
                    'class_id': j,
                    'ckr_mode': self.mode,
                    'raw_static_ckr': round(float(self.static_ckr_raw[ci, j].item()), 6),
                    'static_scaled': round(float(self.static_ckr_scaled[ci, j].item()), 6),
                    'dynamic_signal': None if (isinstance(dyn, float) and math.isnan(dyn)) else round(float(dyn), 6),
                    'ema_ckr': round(float(self.current_ema[ci, j].item()), 6),
                    'effective_ckr': round(float(self.last_server_weights[ci, j].item()), 6),
                    'train_support': int(fit_support[ci][j]) if ci in fit_support else None,
                    'validation_support': int(pm['per_class_support'][j]),
                    'prediction_support': int(pm['per_class_support'][j]),
                    'support_threshold': self.min_support,
                    'status': self._cell_status(ci, j, avail),
                    'fallback': bool(self.last_fallback[ci, j]),
                    'fallback_reason': self.last_fallback_reason[ci][j],
                    'delta_from_previous_round': round(float(self.last_delta[ci, j].item()), 6),
                    'absolute_delta_from_static': round(
                        abs(float(self.current_ema[ci, j].item())
                            - float(self.static_ckr_scaled[ci, j].item())), 6),
                    'selected_metric': self.metric,
                    'ema_gamma': self.ema_decay,
                }
                records.append(record)
        self.history.extend(records)

        if log_path:
            with open(log_path, 'a') as f:
                for rec in records:
                    f.write(json.dumps(rec) + '\n')

    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            'static_ckr_raw': self.static_ckr_raw,
            'static_ckr_scaled': self.static_ckr_scaled,
            'current_ema': self.current_ema,
            'last_dynamic_metric': self.last_dynamic_metric,
            'last_available_mask': self.last_available_mask,
            'last_candidate': self.last_candidate,
            'last_fallback': self.last_fallback,
            'last_fallback_reason': self.last_fallback_reason,
            'seen': self.seen,
            'per_class_support': self.per_class_support,
            'per_client_fit_support': self.per_client_fit_support,
            'last_delta': self.last_delta,
            'round': self.round,
        }

    def load_state_dict(self, sd):
        self.static_ckr_raw = sd['static_ckr_raw']
        self.static_ckr_scaled = sd['static_ckr_scaled']
        self.current_ema = sd['current_ema']
        self.last_dynamic_metric = sd['last_dynamic_metric']
        self.last_available_mask = sd['last_available_mask']
        self.last_candidate = sd.get('last_candidate', self.current_ema.clone())
        self.last_fallback = sd.get('last_fallback',
                                    torch.zeros_like(self.last_available_mask))
        self.last_fallback_reason = sd.get('last_fallback_reason',
                                           [[''] * self.num_classes for _ in range(self.num_clients)])
        self.seen = sd.get('seen', torch.zeros_like(self.last_available_mask))
        self.per_class_support = sd.get('per_class_support', {})
        self.per_client_fit_support = sd.get('per_client_fit_support', {})
        self.last_delta = sd.get('last_delta', torch.zeros_like(self.current_ema))
        self.round = sd['round']
