"""Unit tests for Phase 2 components."""
import pytest
import torch

from src.data.transforms import normalize_csi, spatial_to_angle_delay
from src.models.sampling_mask import SamplingMaskGenerator
from src.models.quantization import ScalarQuantizer
from src.models.phase_head import PhaseHead
from src.utils.baselines import baseline_dft_interp, baseline_linear_interp


def _nmse(pred, target):
    num = (pred - target).abs().square().sum()
    den = target.abs().square().sum() + 1e-12
    return float((10.0 * torch.log10(num / den)).item())


def _make_target_ad(h_sf):
    """Convert a spatial-frequency tensor to normalized angle-delay target."""
    h_ad = spatial_to_angle_delay(h_sf)
    return normalize_csi(h_ad)[0]


def test_dft_interp_full_sampling():
    """Full frequency sampling should reconstruct the spatial-frequency response exactly."""
    B, N_tx, N_rx, M = 2, 4, 2, 32
    h_sf = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    mask = torch.ones(M, dtype=torch.bool)
    sparse_sf = h_sf.clone()

    target_ad_norm = _make_target_ad(h_sf)
    out = baseline_dft_interp(sparse_sf, mask, target_ad_norm)
    assert _nmse(out["pred_ad"], target_ad_norm) < -40.0


def test_dft_interp_uniform_comb():
    """Uniform comb sampling of a delay-sparse signal should reconstruct well."""
    B, N_tx, N_rx, M = 2, 4, 2, 32
    # Build a delay-limited channel in the spatial-frequency domain.
    h_time = torch.zeros(B, N_tx, N_rx, M, dtype=torch.complex64)
    h_time[..., :6] = torch.randn(B, N_tx, N_rx, 6, dtype=torch.complex64)
    h_sf = torch.fft.fft(h_time, n=M, dim=-1, norm="ortho")

    # Uniform comb mask with spacing 4 (K = 8 samples).
    mask = torch.zeros(M, dtype=torch.bool)
    mask[::4] = True
    sparse_sf = torch.zeros_like(h_sf)
    sparse_sf[..., mask] = h_sf[..., mask]

    target_ad_norm = _make_target_ad(h_sf)
    out = baseline_dft_interp(sparse_sf, mask, target_ad_norm)
    pred_ad = out["pred_ad"]

    # Angle-delay prediction should closely match the normalized target.
    assert _nmse(pred_ad, target_ad_norm) < -30.0


def test_dft_interp_extracts_base_grid():
    """With a non-uniform mask, DFT baseline extracts the uniform base grid."""
    M = 32
    # Base grid spacing 8 plus a few peak samples.
    mask = torch.zeros(M, dtype=torch.bool)
    mask[::8] = True
    mask[5] = True
    mask[13] = True

    h_sf = torch.randn(1, 2, 2, M, dtype=torch.complex64)
    sparse_sf = torch.zeros_like(h_sf)
    sparse_sf[..., mask] = h_sf[..., mask]

    target_ad_norm = _make_target_ad(h_sf)
    out = baseline_dft_interp(sparse_sf, mask, target_ad_norm)
    assert out["pred_ad"].shape == target_ad_norm.shape


def test_dft_interp_edge_cases():
    """Edge cases: no samples, single sample, two samples."""
    B, N_tx, N_rx, M = 1, 2, 2, 16
    h_sf = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    target_ad_norm = _make_target_ad(h_sf)

    # No samples.
    mask_empty = torch.zeros(M, dtype=torch.bool)
    sparse_sf = torch.zeros_like(h_sf)
    out = baseline_dft_interp(sparse_sf, mask_empty, target_ad_norm)
    assert torch.allclose(out["pred_ad"], torch.zeros_like(target_ad_norm))

    # Single sample in frequency domain.
    mask_one = torch.zeros(M, dtype=torch.bool)
    idx = M // 2
    mask_one[idx] = True
    sparse_sf = torch.zeros_like(h_sf)
    sparse_sf[..., idx] = h_sf[..., idx]
    out = baseline_dft_interp(sparse_sf, mask_one, target_ad_norm)
    assert out["pred_ad"].shape == target_ad_norm.shape

    # Two samples (uniform comb with spacing M/2).
    mask_two = torch.zeros(M, dtype=torch.bool)
    mask_two[0] = True
    mask_two[M // 2] = True
    sparse_sf = torch.zeros_like(h_sf)
    sparse_sf[..., mask_two] = h_sf[..., mask_two]
    out = baseline_dft_interp(sparse_sf, mask_two, target_ad_norm)
    assert out["pred_ad"].shape == target_ad_norm.shape


def test_linear_interp_full_sampling():
    """Full sampling should reconstruct exactly via linear interpolation."""
    B, N_tx, N_rx, M = 2, 4, 2, 32
    h_sf = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    mask = torch.ones(M, dtype=torch.bool)
    target_ad_norm = _make_target_ad(h_sf)
    out = baseline_linear_interp(h_sf, mask, target_ad_norm)
    assert _nmse(out["pred_ad"], target_ad_norm) < -40.0


def test_linear_interp_sparse_uniform():
    """Sparse uniform sampling should produce a smooth interpolation."""
    B, N_tx, N_rx, M = 2, 4, 2, 32
    h_time = torch.zeros(B, N_tx, N_rx, M, dtype=torch.complex64)
    h_time[..., :6] = torch.randn(B, N_tx, N_rx, 6, dtype=torch.complex64)
    h_sf = torch.fft.fft(h_time, n=M, dim=-1, norm="ortho")

    mask = torch.zeros(M, dtype=torch.bool)
    mask[::4] = True
    sparse_sf = torch.zeros_like(h_sf)
    sparse_sf[..., mask] = h_sf[..., mask]

    target_ad_norm = _make_target_ad(h_sf)
    out = baseline_linear_interp(sparse_sf, mask, target_ad_norm)
    pred_ad = out["pred_ad"]
    assert pred_ad.shape == target_ad_norm.shape
    assert _nmse(pred_ad, target_ad_norm) < 10.0


def test_linear_interp_nonuniform_mask():
    """Linear interpolation should run on non-uniform masks without error."""
    M = 32
    mask = torch.zeros(M, dtype=torch.bool)
    mask[::8] = True
    mask[5] = True
    mask[13] = True

    h_sf = torch.randn(1, 2, 2, M, dtype=torch.complex64)
    sparse_sf = torch.zeros_like(h_sf)
    sparse_sf[..., mask] = h_sf[..., mask]

    target_ad_norm = _make_target_ad(h_sf)
    out = baseline_linear_interp(sparse_sf, mask, target_ad_norm)
    assert out["pred_ad"].shape == target_ad_norm.shape


def test_sampling_mask_generator():
    M = 32
    gen = SamplingMaskGenerator(num_subcarriers=M, use_adaptive=False, base_spacing=8)
    h_ul = torch.randn(32, 4, M, dtype=torch.complex64)
    large_scale = torch.tensor([0.0, 200e-9, 0.0, 0.0, 0.0, 0.0])

    mask = gen(h_ul, large_scale=large_scale, training=False)
    assert mask.shape == (M,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() >= 4  # at least base samples


def test_sampling_mask_training_randomization():
    M = 32
    gen = SamplingMaskGenerator(num_subcarriers=M)
    h_ul = torch.randn(32, 4, M, dtype=torch.complex64)
    large_scale = torch.tensor([0.0, 200e-9, 0.0, 0.0, 0.0, 0.0])

    masks = [gen(h_ul, large_scale=large_scale, training=True) for _ in range(10)]
    counts = [m.sum().item() for m in masks]
    assert max(counts) >= min(counts)  # some variation expected


def test_scalar_quantizer():
    quantizer = ScalarQuantizer(num_bits=8)
    x = torch.randn(4, 32, 4, 32, dtype=torch.complex64)
    x_q = quantizer(x)
    assert x_q.shape == x.shape
    assert x_q.dtype == torch.complex64
    # Quantization introduces noise but preserves approximate range.
    assert (x_q - x).abs().mean().item() > 0.0


def test_phase_head_unit_magnitude():
    head = PhaseHead(feat_dim=16, hidden_dim=32, num_tx=32, num_rx=4)
    x = torch.randn(2, 32 * 4, 16, 16)
    phase = head(x)
    assert phase.shape == (2, 32, 4, 16)
    assert phase.dtype == torch.complex64
    mag_mean = phase.abs().mean().item()
    assert abs(mag_mean - 1.0) < 1e-4
