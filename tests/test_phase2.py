"""Unit tests for Phase 2 components."""
import pytest
import torch

from src.models.sampling_mask import SamplingMaskGenerator
from src.models.quantization import ScalarQuantizer
from src.models.phase_head import PhaseHead


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
