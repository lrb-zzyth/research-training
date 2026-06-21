import re
from typing import Optional


# Regex patterns for FedTAD training output lines
SERVER_PATTERN = re.compile(
    r"\[server\]:\s*current_round:\s*(\d+)\s+global_val:\s*([\d.]+)\s+global_test:\s*([\d.]+)"
)
BEST_PATTERN = re.compile(
    r"\[server\]:\s*best_round:\s*(\d+)\s+best_val:\s*([\d.]+)\s+best_test:\s*([\d.]+)"
)
CLIENT_PATTERN = re.compile(
    r"\[client\s+(\d+)\]:\s*"
    r"acc_train:\s*([\d.]+)\s+"
    r"acc_val:\s*([\d.]+)\s+"
    r"acc_test:\s*([\d.]+)\s+"
    r"loss_train:\s*([\d.]+)\s+"
    r"loss_val:\s*([\d.]+)\s+"
    r"loss_test:\s*([\d.]+)"
)


class ParsedMetric:
    """A single parsed metric entry ready for database insertion."""

    def __init__(self, metric_name: str, metric_value: float,
                 source: str = "server", round_num: Optional[int] = None):
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.source = source
        self.round = round_num


def parse_line(line: str) -> list[ParsedMetric]:
    """Parse a single line of training output and return a list of ParsedMetric."""
    metrics = []
    line_stripped = line.strip()

    # Server metrics: global_val, global_test, current_round
    m = SERVER_PATTERN.search(line_stripped)
    if m:
        round_num = int(m.group(1))
        metrics.append(ParsedMetric("global_val", float(m.group(2)),
                                    source="server", round_num=round_num))
        metrics.append(ParsedMetric("global_test", float(m.group(3)),
                                    source="server", round_num=round_num))
        metrics.append(ParsedMetric("current_round", float(round_num),
                                    source="server", round_num=round_num))
        return metrics

    # Best metrics: best_round, best_val, best_test
    m = BEST_PATTERN.search(line_stripped)
    if m:
        round_num = int(m.group(1))
        metrics.append(ParsedMetric("best_val", float(m.group(2)),
                                    source="server", round_num=round_num))
        metrics.append(ParsedMetric("best_test", float(m.group(3)),
                                    source="server", round_num=round_num))
        return metrics

    # Client metrics
    m = CLIENT_PATTERN.search(line_stripped)
    if m:
        client_id = f"client_{m.group(1)}"
        metrics.append(ParsedMetric("acc_train", float(m.group(2)),
                                    source=client_id))
        metrics.append(ParsedMetric("acc_val", float(m.group(3)),
                                    source=client_id))
        metrics.append(ParsedMetric("acc_test", float(m.group(4)),
                                    source=client_id))
        metrics.append(ParsedMetric("loss_train", float(m.group(5)),
                                    source=client_id))
        metrics.append(ParsedMetric("loss_val", float(m.group(6)),
                                    source=client_id))
        metrics.append(ParsedMetric("loss_test", float(m.group(7)),
                                    source=client_id))
        return metrics

    # Generator info line
    if line_stripped.startswith("[generator]") or line_stripped.startswith("[class weights]"):
        return []

    return []
