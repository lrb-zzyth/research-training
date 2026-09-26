import json
import re
from typing import Optional


# 当前 train_fedtad.py 输出格式:
#   [Round 3]
#   [Global] pooled_pr_auc=0.85 (best=0.86 @ round 3)
#   训练结束. 最佳 round=3, best_val(pooled_pr_auc)=0.8643, best_test(pooled_pr_auc)=0.8211
# ([Global] 行的指标为当前轮 selection_metric, @ round N 为最佳轮次)

# [Round N] 轮次头
ROUND_PATTERN = re.compile(r"\[Round\s+(\d+)\]")

# [Global] <metric>=<val> (best=<best> @ round <N>)
GLOBAL_PATTERN = re.compile(
    r"\[Global\]\s+\S+=([\d.]+)\s+\(best=([\d.]+)\s+@\s+round\s+(\d+)\)"
)

# 训练结束. 最佳 round=N, best_val(...)=X, best_test(...)=Y
FINAL_PATTERN = re.compile(
    r"训练结束\.\s*最佳\s+round=(\d+),\s*best_val\(\S+\)=([\d.]+)"
    r"(?:,\s*best_test\(\S+\)=([\d.]+))?"
)


class ParsedMetric:
    """A single parsed metric entry ready for database insertion."""

    def __init__(self, metric_name: str, metric_value: float,
                 source: str = "server", round_num: Optional[int] = None):
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.source = source
        self.round = round_num


# 结构化事件 -> (指标行, 生命周期事件类型, 事件负载)
# 生命周期事件: round_started / task_started / task_finished / task_failed /
#               parameter_update_applied (由 training.py 消费, 不入指标表)
LIFECYCLE_EVENTS = {"round_started", "task_started", "task_finished",
                    "task_failed", "parameter_update_applied"}


def event_to_metrics(ev: dict):
    """把结构化事件转换为 (ParsedMetric 列表, 生命周期事件类型或 None, 负载)。

    结构化事件是平台指标的第一数据源; 自然语言正则解析仅作兼容 fallback。
    """
    et = ev.get("event")
    rnd = ev.get("round")
    if et in LIFECYCLE_EVENTS:
        return [], et, ev

    metrics = []
    if et == "global_metric":
        if ev.get("global_val") is not None:
            metrics.append(ParsedMetric("global_val", float(ev["global_val"]),
                                        source="server", round_num=rnd))
        if ev.get("global_test") is not None:
            metrics.append(ParsedMetric("global_test", float(ev["global_test"]),
                                        source="server", round_num=rnd))
        if ev.get("best_val") is not None:
            metrics.append(ParsedMetric("best_val", float(ev["best_val"]),
                                        source="server", round_num=rnd))
        if ev.get("best_test") is not None:
            metrics.append(ParsedMetric("best_test", float(ev["best_test"]),
                                        source="server", round_num=rnd))
        if rnd is not None:
            metrics.append(ParsedMetric("current_round", float(rnd),
                                        source="server", round_num=rnd))
    elif et == "client_training_metric":
        src = f"client_{ev.get('client_id', '?')}"
        for name in ("ce_loss", "cl_loss", "total_loss"):
            v = ev.get(name)
            if v is not None:
                metrics.append(ParsedMetric(name, float(v), source=src,
                                            round_num=rnd))
        if ev.get("train_samples") is not None:
            metrics.append(ParsedMetric("train_samples", float(ev["train_samples"]),
                                        source=src, round_num=rnd))
        metrics.append(ParsedMetric("zero_anomaly",
                                    float(bool(ev.get("zero_anomaly"))),
                                    source=src, round_num=rnd))
    elif et == "generator_metric":
        for name in ("generator_loss", "semantic_loss", "disagreement_loss",
                     "diversity_loss", "feature_norm_loss", "grad_norm",
                     "fake_x_mean", "fake_x_std", "fake_x_min", "fake_x_max",
                     "fake_x_norm", "peak_gpu_mb"):
            v = ev.get(name)
            if v is not None:
                metrics.append(ParsedMetric(name, float(v), source="server",
                                            round_num=rnd))
    elif et == "distillation_metric":
        for name in ("distillation_loss", "distill_grad_norm",
                     "fake_graph_nodes", "fake_graph_edges",
                     "fake_graph_avg_degree", "fake_graph_components"):
            v = ev.get(name)
            if v is not None:
                metrics.append(ParsedMetric(name, float(v), source="server",
                                            round_num=rnd))
    elif et == "ckr_update":
        w = ev.get("server_weights") or []
        for ci, row in enumerate(w):
            for j, v in enumerate(row):
                if v is not None:
                    metrics.append(ParsedMetric(f"ckr_c{ci}c{j}", float(v),
                                                source="ckr", round_num=rnd))
        if ev.get("ckr_mode"):
            metrics.append(ParsedMetric("ckr_mode_active", 1.0, source="ckr",
                                        round_num=rnd))
    elif et == "checkpoint_saved":
        metrics.append(ParsedMetric("checkpoint_saved", 1.0, source="server",
                                    round_num=rnd))
    elif et == "resource_usage":
        for name in ("round_time_sec", "generator_peak_gpu_mb",
                     "rwr_cache_hit_rate"):
            v = ev.get(name)
            if v is not None:
                metrics.append(ParsedMetric(name, float(v), source="server",
                                            round_num=rnd))
    return metrics, None, ev


# 结构化事件: [EVENT] {json} (train_fedtad.py --emit_events)
EVENT_PATTERN = re.compile(r"^\[EVENT\] (\{.*\})$")


def parse_event_line(line: str) -> Optional[dict]:
    """解析结构化事件行, 返回事件 dict; 非事件行返回 None。

    结构化事件是平台指标的第一数据源, 自然语言正则仅作 fallback。
    """
    m = EVENT_PATTERN.search(line.strip())
    if not m:
        return None
    try:
        ev = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(ev, dict) or "event" not in ev:
        return None
    return ev


def parse_line(line: str, current_round: Optional[int] = None) -> list[ParsedMetric]:
    """Parse a single line of training output and return a list of ParsedMetric.

    current_round: 最近一次 [Round N] 头解析出的轮次, 由调用方维护。
    结构化 [EVENT] 行由 parse_event_line 处理, 这里跳过。
    """
    metrics = []
    line_stripped = line.strip()

    # [Global] 当前轮全局指标 + best 曲线
    m = GLOBAL_PATTERN.search(line_stripped)
    if m:
        rnd = current_round or int(m.group(3))
        metrics.append(ParsedMetric("global_val", float(m.group(1)),
                                    source="server", round_num=rnd))
        metrics.append(ParsedMetric("best_val", float(m.group(2)),
                                    source="server", round_num=rnd))
        metrics.append(ParsedMetric("current_round", float(rnd),
                                    source="server", round_num=rnd))
        return metrics

    # 训练结束: 最终 best_val / best_test
    m = FINAL_PATTERN.search(line_stripped)
    if m:
        round_num = int(m.group(1))
        metrics.append(ParsedMetric("best_val", float(m.group(2)),
                                    source="server", round_num=round_num))
        if m.group(3):
            metrics.append(ParsedMetric("best_test", float(m.group(3)),
                                        source="server", round_num=round_num))
        metrics.append(ParsedMetric("current_round", float(round_num),
                                    source="server", round_num=round_num))
        return metrics

    # Skip known non-metric lines
    skip_prefixes = ("[CKR]", "[S1]", "[S3]", "[S4]", "[Round", "[EVENT",
                     "generator", "class weights", "--", "Core method",
                     "任务模式", "训练结束")
    if any(line_stripped.startswith(p) for p in skip_prefixes):
        return []

    return []
