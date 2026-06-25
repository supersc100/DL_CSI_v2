"""Baseline methods for FDD downlink CSI prediction."""
from typing import Dict, Optional

import torch

import numpy as np

from src.data.transforms import (
    angle_delay_to_spatial,
    real_channels_to_complex,
    spatial_to_angle_delay,
)


def baseline_copy_ul(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
    transform=None,
) -> Dict[str, torch.Tensor]:
    """Baseline 1: predict DL = current UL (performance lower bound)."""
    pred = current_ul_ad.clone()
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_angle_delay_interp(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
    history_ul_ad: Optional[torch.Tensor] = None,
    history_dl_ad: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Baseline 2: linear interpolation in angle-delay domain from recent history."""
    if history_dl_ad is None or history_dl_ad.shape[1] < 2:
        # Not enough history: fall back to copying current UL.
        return baseline_copy_ul(current_ul_ad, target_dl_ad)
    # Use last two historical DL snapshots to linearly extrapolate next DL.
    h0 = history_dl_ad[:, -2]
    h1 = history_dl_ad[:, -1]
    pred = 2.0 * h1 - h0
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_tdd_oracle(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Baseline 3: TDD upper bound, UL and DL share identical fast fading.

    In a real TDD oracle dataset h_ul == h_dl after calibration; here we simply
    copy the true DL target as the prediction to measure the dataset upper bound.
    """
    return {"pred_ad": target_dl_ad.clone(), "target_ad": target_dl_ad}


def baseline_no_large_scale(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
    history_ul_ad: Optional[torch.Tensor] = None,
    history_dl_ad: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Baseline 4 (ablation): use current UL but no explicit large-scale params.

    The large-scale vector is ignored for this simple baseline.
    """
    return baseline_copy_ul(current_ul_ad, target_dl_ad, None)


def baseline_no_history(
    current_ul_ad: torch.Tensor,
    large_scale: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Baseline 5 (ablation): use current UL + large-scale but no history.

    Implemented as current UL only because the simple baseline cannot consume
    the large-scale vector without a trained model.
    """
    return baseline_copy_ul(current_ul_ad, target_dl_ad, None)


# ---------------------------------------------------------------------------
# Phase 2 baselines
# ---------------------------------------------------------------------------

def baseline_magnitude_only(
    stage1_pred: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Phase2 baseline 1: use Stage1 magnitude, phase = 0."""
    pred = stage1_pred.abs().to(torch.complex64)
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_linear_interp(
    sparse_dl_ad: torch.Tensor,
    mask: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Phase2 baseline 2: linear interpolation along subcarrier dimension."""
    pred = torch.zeros_like(sparse_dl_ad)
    B, N_tx, N_rx, M = sparse_dl_ad.shape
    for b in range(B):
        sampled_indices = mask[b].nonzero(as_tuple=True)[0].cpu().numpy()
        if len(sampled_indices) < 2:
            pred[b] = sparse_dl_ad[b]
            continue
        sampled_indices = sampled_indices.astype(float)
        for tx in range(N_tx):
            for rx in range(N_rx):
                vals = sparse_dl_ad[b, tx, rx, mask[b]].cpu().numpy()
                # Linear interpolation for real and imaginary separately.
                pred_real = np.interp(
                    np.arange(M), sampled_indices, vals.real, left=vals.real[0], right=vals.real[-1]
                )
                pred_imag = np.interp(
                    np.arange(M), sampled_indices, vals.imag, left=vals.imag[0], right=vals.imag[-1]
                )
                pred[b, tx, rx] = torch.complex(
                    torch.from_numpy(pred_real),
                    torch.from_numpy(pred_imag),
                ).to(sparse_dl_ad.device)
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_full_feedback(
    target_dl_ad: torch.Tensor,
    quantizer=None,
) -> Dict[str, torch.Tensor]:
    """Phase2 baseline (upper bound): full-band DL CSI fed back directly.

    Represents an ideal feedback scheme that returns the complete downlink CSI
    (optionally passed through a scalar quantizer to model limited feedback).
    Serves as the "full feedback" performance ceiling in NMSE-vs-SNR plots.
    """
    pred = target_dl_ad.clone()
    if quantizer is not None:
        pred = quantizer(pred)
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_dft_interp(
    sparse_dl_ad: torch.Tensor,
    mask: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Phase2 baseline 3: mask-aware DFT interpolation for sparse OFDM pilots.

    The sampling mask is defined in the spatial-frequency domain, so the input
    (which is in the angle-delay domain) is first transformed back to the
    angle-frequency domain via FFT along subcarriers. A delay-domain least-squares
    problem is solved from the observed pilot indices, the full frequency response
    is reconstructed, and the result is transformed back to angle-delay domain.
    Works for both uniform and non-uniform sampling masks.
    """
    B, N_tx, N_rx, M = sparse_dl_ad.shape
    device = sparse_dl_ad.device
    dtype = sparse_dl_ad.dtype

    # Transform to angle-frequency domain, where the mask has meaning.
    sparse_af = torch.fft.fft(sparse_dl_ad, n=M, dim=-1, norm="ortho")
    pred_af = torch.zeros_like(sparse_af)

    # Collate yields [B, M]; support [M] shared mask for backward compatibility.
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(B, -1)

    N_spatial = N_tx * N_rx
    for b in range(B):
        mask_b = mask[b]
        sampled_indices = mask_b.nonzero(as_tuple=True)[0].to(torch.float32)
        K = sampled_indices.numel()
        if K == 0:
            continue

        # Effective delay taps: cap at M//2 (delay-sparse prior).
        L = min(K, M // 2)
        if L < 1:
            L = 1

        # Partial DFT matrix A[k, l] = (1/sqrt(M)) * exp(-j * 2π * m_k * l / M)
        # to match torch.fft.fft with norm="ortho".
        l_range = torch.arange(L, device=device, dtype=torch.float32)
        phase = -2.0 * torch.pi * torch.outer(sampled_indices, l_range) / M
        A = torch.exp(1j * phase).to(dtype) / (M ** 0.5)  # [K, L]

        # Sampled values for all TX/RX (angle-frequency): [N_spatial, K]
        y = sparse_af[b].reshape(N_spatial, M)[:, mask_b]

        # Tikhonov-regularized least squares for numerical stability under noise.
        # h = (A^H A + λ I)^{-1} A^H y.  We compute B = (A^H A + λI)^{-1} A^H
        # (shape [L, K]) and then h = y @ B.T.
        A_H = A.conj().T  # [L, K]
        AHA = A_H @ A  # [L, L]
        reg = 1e-4 * torch.eye(L, device=device, dtype=dtype)
        B = torch.linalg.solve(AHA + reg, A_H)  # [L, K]
        h_ls = y @ B.T.to(dtype)  # [N_spatial, L]

        # Zero-pad to M delay taps, FFT back to angle-frequency domain.
        h_full = torch.zeros(N_spatial, M, dtype=dtype, device=device)
        h_full[:, :L] = h_ls
        pred_af[b] = torch.fft.fft(h_full, n=M, dim=-1, norm="ortho").reshape(
            N_tx, N_rx, M
        )

    # Transform back to angle-delay domain.
    pred_ad = torch.fft.ifft(pred_af, n=M, dim=-1, norm="ortho")
    return {"pred_ad": pred_ad, "target_ad": target_dl_ad}


BASELINES = {
    "copy_ul": baseline_copy_ul,
    "angle_delay_interp": baseline_angle_delay_interp,
    "tdd_oracle": baseline_tdd_oracle,
    "no_large_scale": baseline_no_large_scale,
    "no_history": baseline_no_history,
    # Phase 2 baselines
    "magnitude_only": baseline_magnitude_only,
    "linear_interp": baseline_linear_interp,
    "dft_interp": baseline_dft_interp,
    "full_feedback": baseline_full_feedback,
}
