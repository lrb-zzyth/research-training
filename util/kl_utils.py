"""KL helpers with explicit argument order.

The project follows FedTAD Eq. (10) and the patent S4.2 convention:
    KL(global_prediction || local_teacher_prediction)

Keeping this in a small pure-PyTorch module avoids the easy-to-miss input/target
reversal of ``torch.nn.functional.kl_div``.
"""

import torch
import torch.nn.functional as F


def categorical_kl_from_logits(p_logits, q_logits, temperature=1.0,
                               reduction='batchmean'):
    """Return KL(P || Q) for categorical distributions parameterized by logits."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    log_p = F.log_softmax(p_logits / temperature, dim=-1)
    log_q = F.log_softmax(q_logits / temperature, dim=-1)
    p = log_p.exp()
    per_example = (p * (log_p - log_q)).sum(dim=-1)

    if reduction == 'none':
        return per_example
    if reduction == 'sum':
        return per_example.sum()
    if reduction in ('batchmean', 'mean'):
        return per_example.mean()
    raise ValueError(f"unsupported reduction: {reduction}")
