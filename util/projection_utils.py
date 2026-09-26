"""Numerically stable radius-manifold projection utilities."""

import torch


def project_to_pretrain_radius(raw_x, labels, radius_bank, eps=1e-12):
    """Project each raw feature vector onto its pretrained class radius.

    This is algebraically the same map as ``r * z / ||z||`` but normalizes after
    first scaling each row into O(1) magnitude.  It therefore avoids float32
    squared-norm overflow when the DDPM raw trajectory is huge but still finite.
    """
    if raw_x.ndim != 2:
        raise ValueError(f"raw_x must be 2-D [B,F], got shape={tuple(raw_x.shape)}")
    if labels.ndim != 1 or labels.shape[0] != raw_x.shape[0]:
        raise ValueError("labels must be [B] and match raw_x batch size")
    if radius_bank.ndim != 2:
        raise ValueError("radius_bank must be [num_classes, bank_size]")

    batch_size = raw_x.shape[0]
    bank_size = radius_bank.shape[1]
    bank_idx = torch.arange(batch_size, device=raw_x.device) % bank_size
    target_radius = radius_bank[labels, bank_idx].unsqueeze(1)

    # The detached scale cancels algebraically; it is only a numerical device.
    row_scale = raw_x.detach().abs().amax(dim=1, keepdim=True).clamp_min(eps)
    scaled = raw_x / row_scale
    scaled_norm = torch.sqrt(scaled.pow(2).sum(dim=1, keepdim=True) + eps)
    unit = scaled / scaled_norm
    return unit * target_radius


def diagnostic_l2_norm(raw_x, dim=1):
    """Overflow-resistant diagnostic norm, computed in float64 and detached."""
    return torch.linalg.vector_norm(raw_x.detach().to(torch.float64), ord=2, dim=dim)
