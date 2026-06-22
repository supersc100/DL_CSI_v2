"""Evaluation metrics for downlink CSI prediction."""
from typing import Dict

import torch


def _ensure_float32(t: torch.Tensor) -> torch.Tensor:
    if t.is_complex():
        return t.to(torch.complex64)
    return t.to(torch.float32)


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized mean squared error (dB)."""
    pred = _ensure_float32(pred)
    target = _ensure_float32(target)
    diff = pred - target
    num = diff.abs().square().sum(dim=tuple(range(1, diff.ndim)))
    den = target.abs().square().sum(dim=tuple(range(1, target.ndim))) + 1e-12
    return 10.0 * torch.log10((num / den).mean() + 1e-12)


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between flattened CSI vectors."""
    pred = _ensure_float32(pred)
    target = _ensure_float32(target)
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)
    if pred_flat.is_complex():
        pred_real = torch.view_as_real(pred_flat).view(B, -1)
        target_real = torch.view_as_real(target_flat).view(B, -1)
    else:
        pred_real = pred_flat
        target_real = target_flat
    cos = torch.nn.functional.cosine_similarity(pred_real, target_real, dim=-1)
    return cos.mean()


def nmse_angle_delay(
    pred_spatial: torch.Tensor,
    target_spatial: torch.Tensor,
    transform,
) -> torch.Tensor:
    """NMSE computed in the angle-delay domain."""
    from src.data.transforms import spatial_to_angle_delay

    pred_ad = spatial_to_angle_delay(pred_spatial)
    target_ad = spatial_to_angle_delay(target_spatial)
    return nmse(pred_ad, target_ad)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Return a dict of common complex-domain metrics."""
    return {
        "nmse_db": float(nmse(pred, target).item()),
        "cosine_similarity": float(cosine_similarity(pred, target).item()),
    }


def compute_magnitude_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Return NMSE and cosine similarity computed on magnitude spectra.

    This is the primary evaluation for FDD UL->DL prediction because UL and DL
    share angle-delay power structure while their small-scale phases are
    independent.
    """
    mag_pred = pred.abs()
    mag_target = target.abs()
    return {
        "magnitude_nmse_db": float(nmse(mag_pred, mag_target).item()),
        "magnitude_cosine_similarity": float(
            cosine_similarity(mag_pred, mag_target).item()
        ),
    }


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Return both complex and magnitude metrics in one dict."""
    return {
        **compute_metrics(pred, target),
        **compute_magnitude_metrics(pred, target),
    }
