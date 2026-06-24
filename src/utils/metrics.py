"""Evaluation metrics for downlink CSI prediction."""
from typing import Dict, Optional

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


def phase_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NMSE computed only on the phase component."""
    pred = _ensure_float32(pred)
    target = _ensure_float32(target)
    pred_phase = pred / (pred.abs().clamp_min(1e-12))
    target_phase = target / (target.abs().clamp_min(1e-12))
    diff = pred_phase - target_phase
    num = diff.abs().square().sum(dim=tuple(range(1, diff.ndim)))
    den = target.abs().square().sum(dim=tuple(range(1, target.ndim))) + 1e-12
    return 10.0 * torch.log10((num / den).mean() + 1e-12)


def compute_phase2_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mag_stage1: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Metrics for Phase2 full-CSI evaluation."""
    metrics = compute_all_metrics(pred, target)
    metrics["phase_nmse_db"] = float(phase_nmse(pred, target).item())
    if mag_stage1 is not None:
        metrics["magnitude_nmse_db"] = float(nmse(mag_stage1, target.abs()).item())
    return metrics


def spectral_efficiency(
    pred_spatial: torch.Tensor,
    target_spatial: torch.Tensor,
    snr_db: float,
) -> Dict[str, float]:
    """Single-stream spectral efficiency from predicted CSI used for precoding.

    Operating model (per subcarrier, downlink BS->UE):
        - ``H[m]`` is the ``[N_rx, N_tx]`` channel matrix.
        - The BS forms a unit-norm precoder ``w`` from the *predicted* channel
          (dominant right singular vector / eigen-beamforming).
        - The achievable rate on the *true* channel is
          ``log2(1 + snr_lin * ||H_true[m] @ w||^2)`` assuming MRC at the UE.
        - SE is averaged over subcarriers (and the batch).

    A perfect-CSI upper bound (``w`` from the true channel) is also returned.

    Note: the model operates on per-sample normalized CSI, so the channel power
    is ~unit and ``snr_db`` is the post-normalization operating SNR.

    Args:
        pred_spatial: predicted CSI, complex ``[B, N_tx, N_rx, M]``.
        target_spatial: true CSI, complex ``[B, N_tx, N_rx, M]``.
        snr_db: operating SNR in dB.

    Returns:
        dict with ``se_pred`` and ``se_perfect`` (bps/Hz).
    """
    pred = _ensure_float32(pred_spatial)
    target = _ensure_float32(target_spatial)
    B, N_tx, N_rx, M = pred.shape
    snr_lin = 10.0 ** (snr_db / 10.0)

    # [B, M, N_rx, N_tx]
    H_pred = pred.permute(0, 3, 2, 1).reshape(B * M, N_rx, N_tx)
    H_true = target.permute(0, 3, 2, 1).reshape(B * M, N_rx, N_tx)

    def _dominant_precoder(H: torch.Tensor) -> torch.Tensor:
        # Right singular vector with largest singular value -> first row of Vh.
        _, _, Vh = torch.linalg.svd(H, full_matrices=False)
        w = Vh[:, 0, :].conj()  # [K, N_tx]
        return w / w.abs().square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)

    def _se(w: torch.Tensor) -> float:
        eff = torch.einsum("krt,kt->kr", H_true, w)  # [K, N_rx]
        gain = eff.abs().square().sum(dim=-1)         # [K] MRC combining
        rate = torch.log2(1.0 + snr_lin * gain)       # [K]
        return float(rate.mean().item())

    return {
        "se_pred": _se(_dominant_precoder(H_pred)),
        "se_perfect": _se(_dominant_precoder(H_true)),
    }
