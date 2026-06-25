"""Unit tests for Phase 2 components."""
import pytest
import torch

from src.models.sampling_mask import SamplingMaskGenerator
from src.models.quantization import ScalarQuantizer
from src.models.phase_head import PhaseHead
from src.utils.baselines import baseline_dft_interp


def _nmse(pred, target):
    num = (pred - target).abs().square().sum()
    den = target.abs().square().sum() + 1e-12
    return float((10.0 * torch.log10(num / den)).item())


def test_dft_interp_full_sampling():
    """Full frequency sampling of a delay-sparse signal should reconstruct exactly."""
    B, N_tx, N_rx, M = 2, 4, 2, 32
    # h_delay lives in the angle-delay domain (last dim = delay).
    h_delay = torch.zeros(B, N_tx, N_rx, M, dtype=torch.complex64)
    h_delay[..., : M // 2] = torch.randn(B, N_tx, N_rx, M // 2, dtype=torch.complex64)
    # Transform to angle-frequency; the mask is defined here.
    h_af = torch.fft.fft(h_delay, n=M, dim=-1, norm="ortho")

    mask = torch.ones(M, dtype=torch.bool)
    sparse_af = h_af.clone()
    # Baseline receives angle-delay input, like the real pipeline.
    sparse_ad = torch.fft.ifft(sparse_af, n=M, dim=-1, norm="ortho")

    out = baseline_dft_interp(sparse_ad, mask, h_delay)
    assert _nmse(out["pred_ad"], h_delay) < -40.0


def test_dft_interp_nonuniform_mask():
    """Non-uniform sparse frequency sampling: reconstructed AF samples are exact."""
    B, N_tx, N_rx, M = 2, 4, 2, 32
    h_delay = torch.zeros(B, N_tx, N_rx, M, dtype=torch.complex64)
    h_delay[..., :6] = torch.randn(B, N_tx, N_rx, 6, dtype=torch.complex64)
    h_af = torch.fft.fft(h_delay, n=M, dim=-1, norm="ortho")

    mask = torch.zeros(M, dtype=torch.bool)
    mask[::3] = True
    mask[5] = True
    mask[17] = True
    mask[29] = True

    sparse_af = torch.zeros_like(h_af)
    sparse_af[..., mask] = h_af[..., mask]
    sparse_ad = torch.fft.ifft(sparse_af, n=M, dim=-1, norm="ortho")

    out = baseline_dft_interp(sparse_ad, mask, h_delay)
    pred = out["pred_ad"]

    # The regularized LS solution approximately reproduces the observed samples.
    pred_af = torch.fft.fft(pred, n=M, dim=-1, norm="ortho")
    assert torch.allclose(pred_af[..., mask], h_af[..., mask], atol=1e-2)
    # Overall reconstruction should be decent for a delay-sparse signal.
    assert _nmse(pred, h_delay) < 0.0


def test_dft_interp_edge_cases():
    """Edge cases: no samples, single sample, underdetermined system."""
    B, N_tx, N_rx, M = 1, 2, 2, 16
    target_ad = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    target_af = torch.fft.fft(target_ad, n=M, dim=-1, norm="ortho")

    # No samples.
    mask_empty = torch.zeros(M, dtype=torch.bool)
    sparse_ad = torch.zeros_like(target_ad)
    out = baseline_dft_interp(sparse_ad, mask_empty, target_ad)
    assert torch.allclose(out["pred_ad"], torch.zeros_like(target_ad))

    # Single sample in frequency domain.
    mask_one = torch.zeros(M, dtype=torch.bool)
    idx = M // 2
    mask_one[idx] = True
    sparse_af = torch.zeros_like(target_af)
    sparse_af[..., idx] = target_af[..., idx]
    sparse_ad = torch.fft.ifft(sparse_af, n=M, dim=-1, norm="ortho")
    out = baseline_dft_interp(sparse_ad, mask_one, target_ad)
    # With one sample, the reconstructed AF is approximately constant equal to the sample.
    expected_af = sparse_af[..., idx : idx + 1].expand_as(target_af)
    expected_ad = torch.fft.ifft(expected_af, n=M, dim=-1, norm="ortho")
    assert torch.allclose(out["pred_ad"], expected_ad, atol=1e-2)

    # Two samples (underdetermined, L_eff capped at 2).
    mask_two = torch.zeros(M, dtype=torch.bool)
    mask_two[2] = True
    mask_two[10] = True
    sparse_af = torch.zeros_like(target_af)
    sparse_af[..., mask_two] = target_af[..., mask_two]
    sparse_ad = torch.fft.ifft(sparse_af, n=M, dim=-1, norm="ortho")
    out = baseline_dft_interp(sparse_ad, mask_two, target_ad)
    pred_af = torch.fft.fft(out["pred_ad"], n=M, dim=-1, norm="ortho")
    assert torch.allclose(pred_af[..., mask_two], target_af[..., mask_two], atol=1e-2)


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
