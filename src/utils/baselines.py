"""Baseline methods for FDD downlink CSI prediction."""
from typing import Dict, Optional

import torch

import numpy as np

from src.data.transforms import (
    angle_delay_to_spatial,
    normalize_csi,
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
# Phase 2 helpers
# ---------------------------------------------------------------------------


def _extract_uniform_subset(mask: torch.Tensor) -> torch.Tensor:
    """Extract the largest complete uniform comb subset from a boolean mask.

    A complete comb with spacing ``d`` and offset ``offset`` means every index
    ``offset + k*d`` (for k = 0, 1, ...) is present in ``mask``.  This is the
    natural grid for decimated-IFFT DFT interpolation.

    Args:
        mask: bool tensor of shape [M] or [B, M].

    Returns:
        bool tensor of the same shape, True only on the selected uniform
        comb positions that are also True in the input mask.
    """
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    B, M = mask.shape
    device = mask.device
    result = torch.zeros_like(mask)

    for b in range(B):
        best_count = 0
        best_d = 1
        best_offset = 0

        # Search over all candidate spacings and offsets.
        for d in range(1, M):
            for offset in range(d):
                positions = torch.arange(offset, M, d, device=device)
                if positions.numel() == 0:
                    continue
                valid = mask[b, positions]
                # Only consider complete combs (every position present).
                if not valid.all():
                    continue
                count = positions.numel()
                if count > best_count:
                    best_count = count
                    best_d = d
                    best_offset = offset

        if best_count > 0:
            positions = torch.arange(best_offset, M, best_d, device=device)
            result[b, positions] = True

    return result


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
    sparse_dl: torch.Tensor,
    mask: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Phase2 baseline 2: linear interpolation in the spatial-frequency domain.

    The sampling mask is defined in the spatial-frequency domain, so the
    interpolation is performed along subcarriers on ``sparse_dl``.  The
    reconstructed full-band spatial-frequency response is then transformed back
    to the normalized angle-delay domain for fair comparison with
    ``target_dl_ad``.
    """
    B, N_tx, N_rx, M = sparse_dl.shape
    device = sparse_dl.device
    dtype = sparse_dl.dtype

    # Collate mask to [B, M].
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(B, -1)

    pred_sf = torch.zeros_like(sparse_dl)

    for b in range(B):
        mask_b = mask[b]
        sampled_indices = mask_b.nonzero(as_tuple=True)[0].cpu().numpy()
        if len(sampled_indices) < 2:
            pred_sf[b] = sparse_dl[b]
            continue
        sampled_indices = sampled_indices.astype(float)
        for tx in range(N_tx):
            for rx in range(N_rx):
                vals = sparse_dl[b, tx, rx, mask_b].cpu().numpy()
                pred_real = np.interp(
                    np.arange(M), sampled_indices, vals.real,
                    left=vals.real[0], right=vals.real[-1]
                )
                pred_imag = np.interp(
                    np.arange(M), sampled_indices, vals.imag,
                    left=vals.imag[0], right=vals.imag[-1]
                )
                pred_sf[b, tx, rx] = torch.complex(
                    torch.from_numpy(pred_real),
                    torch.from_numpy(pred_imag),
                ).to(device, dtype)

    # Convert back to angle-delay domain and normalize to match target scale.
    pred_ad = spatial_to_angle_delay(pred_sf)
    pred_ad_norm, _ = normalize_csi(pred_ad)
    return {"pred_ad": pred_ad_norm, "target_ad": target_dl_ad}


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
    sparse_dl: torch.Tensor,
    mask: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Phase2 baseline 3: DFT interpolation in the spatial-frequency domain.

    The mask is defined in the spatial-frequency domain.  A complete uniform
    comb subset is extracted from the mask.  The classical OFDM-style DFT
    interpolation is then applied per antenna pair:

        1. Extract the K observed samples on the uniform comb.
        2. K-point IFFT to obtain K delay-domain taps.
        3. Compensate the linear phase induced by a non-zero comb offset.
        4. Zero-pad the taps to M and M-point FFT back to the full frequency
           response.

    The reconstructed full-band spatial-frequency response is finally
    transformed back to the normalized angle-delay domain.
    """
    B, N_tx, N_rx, M = sparse_dl.shape
    device = sparse_dl.device
    dtype = sparse_dl.dtype

    # Collate mask to [B, M]; support [M] shared mask for backward compatibility.
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(B, -1)

    # Extract the largest complete uniform comb from the mask.
    uniform_mask = _extract_uniform_subset(mask)

    pred_sf = torch.zeros_like(sparse_dl)

    for b in range(B):
        uni_b = uniform_mask[b]
        uni_idx = uni_b.nonzero(as_tuple=True)[0]
        K = uni_idx.numel()
        if K == 0:
            continue

        offset = int(uni_idx[0].item())
        d = int((uni_idx[1] - uni_idx[0]).item()) if K > 1 else 1

        # Comb samples: [N_tx, N_rx, K].
        h_comb = sparse_dl[b, :, :, uni_b]

        # K-point IFFT gives delay-domain taps of the phase-shifted sequence.
        h_time = torch.fft.ifft(h_comb, n=K, dim=-1, norm="ortho")

        # Undo the linear phase induced by the comb offset.
        if offset != 0:
            n_grid = torch.arange(K, device=device, dtype=torch.float32)
            phase = torch.exp(1j * 2.0 * torch.pi * offset * n_grid / M).to(dtype)
            h_time = h_time * phase.view(1, 1, K)

        # Scale correction: orthonormal K-point IFFT + M-point FFT introduces
        # a factor of 1/sqrt(d). Compensate so that full sampling (d=1) is exact.
        if d > 1:
            h_time = h_time * (d ** 0.5)

        # Zero-pad to M taps and FFT back to full frequency response.
        h_padded = torch.zeros(N_tx, N_rx, M, dtype=dtype, device=device)
        h_padded[:, :, :K] = h_time
        pred_sf[b] = torch.fft.fft(h_padded, n=M, dim=-1, norm="ortho")

    # Convert back to angle-delay domain and normalize to match target scale.
    pred_ad = spatial_to_angle_delay(pred_sf)
    pred_ad_norm, _ = normalize_csi(pred_ad)
    return {"pred_ad": pred_ad_norm, "target_ad": target_dl_ad}


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
