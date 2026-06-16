"""Unit tests for angle-delay transforms."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.data.transforms import (
    spatial_to_angle_delay,
    angle_delay_to_spatial,
    complex_to_real_channels,
    real_channels_to_complex,
)


def test_roundtrip():
    """Angle-delay transform should be invertible up to numerical precision."""
    N_tx, N_rx, M = 16, 4, 64
    h = torch.randn(N_tx, N_rx, M, dtype=torch.complex64)
    h_ad = spatial_to_angle_delay(h, antenna_dim=0, frequency_dim=-1)
    h_rec = angle_delay_to_spatial(h_ad, antenna_dim=0, frequency_dim=-1)
    assert torch.allclose(h, h_rec, atol=1e-5)


def test_real_channel_conversion():
    """complex <-> real-channel representation should be invertible."""
    B, N_tx, N_rx, M = 2, 8, 2, 32
    h = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    h_ri = complex_to_real_channels(h)
    assert h_ri.shape == (B, 2, N_tx, N_rx, M)
    h_rec = real_channels_to_complex(h_ri)
    assert torch.allclose(h, h_rec, atol=1e-6)


def test_gradient_flow():
    """Transforms should produce gradients when part of a model."""
    N_tx, N_rx, M = 8, 2, 16
    h = torch.randn(N_tx, N_rx, M, dtype=torch.complex64, requires_grad=True)
    h_ad = spatial_to_angle_delay(h, antenna_dim=0, frequency_dim=-1)
    loss = (h_ad.abs() ** 2).sum()
    loss.backward()
    assert h.grad is not None


if __name__ == "__main__":
    test_roundtrip()
    test_real_channel_conversion()
    test_gradient_flow()
    print("All transform tests passed.")
