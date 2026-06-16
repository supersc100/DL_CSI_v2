"""Spatial-frequency <-> Angle-Delay domain transforms.

All transforms operate on complex-valued CSI tensors and are implemented with
`torch.fft` so that gradients flow through them during training.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _check_antenna_dim(shape: Tuple[int, ...], antenna_dim: int) -> None:
    if antenna_dim < 0 or antenna_dim >= len(shape):
        raise ValueError(f"antenna_dim {antenna_dim} is out of range for shape {shape}")


def spatial_to_angle_delay(
    h: torch.Tensor,
    antenna_dim: int = -2,
    frequency_dim: int = -1,
    norm: str = "ortho",
    use_2d_antenna_dft: bool = False,
) -> torch.Tensor:
    """Convert spatial-frequency CSI to angle-delay domain.

    Args:
        h: Complex tensor of shape [..., N_ant, M] (or [..., N_y, N_x, M] for 2D arrays).
        antenna_dim: Dimension along which the BS array response lives.
        frequency_dim: Subcarrier dimension.
        norm: FFT normalization, "ortho" keeps energy preserved.
        use_2d_antenna_dft: If True, apply 2D-DFT on the two preceding dimensions
            before the frequency dim. Useful for UPA/2D arrays.

    Returns:
        Complex tensor in angle-delay domain, same rank as input.
    """
    _check_antenna_dim(h.shape, antenna_dim % h.ndim)

    if use_2d_antenna_dft:
        # For 2D arrays, the two dims immediately before frequency_dim are antennas.
        if frequency_dim < 0:
            freq_idx = frequency_dim % h.ndim
        else:
            freq_idx = frequency_dim
        # DFT along the two antenna dimensions.
        h = torch.fft.fft2(h, dim=(freq_idx - 2, freq_idx - 1), norm=norm)
    else:
        # DFT along the BS antenna dimension to move to angle domain.
        h = torch.fft.fft(h, dim=antenna_dim, norm=norm)

    # IFFT along subcarrier dimension to move to delay domain.
    h = torch.fft.ifft(h, dim=frequency_dim, norm=norm)
    return h


def angle_delay_to_spatial(
    h: torch.Tensor,
    antenna_dim: int = -2,
    frequency_dim: int = -1,
    norm: str = "ortho",
    use_2d_antenna_dft: bool = False,
) -> torch.Tensor:
    """Inverse transform: angle-delay -> spatial-frequency."""
    if use_2d_antenna_dft:
        if frequency_dim < 0:
            freq_idx = frequency_dim % h.ndim
        else:
            freq_idx = frequency_dim
        h = torch.fft.ifft2(h, dim=(freq_idx - 2, freq_idx - 1), norm=norm)
    else:
        h = torch.fft.ifft(h, dim=antenna_dim, norm=norm)

    h = torch.fft.fft(h, dim=frequency_dim, norm=norm)
    return h


def complex_to_real_channels(h: torch.Tensor) -> torch.Tensor:
    """Represent complex CSI as two real channels [..., 2, ...]."""
    return torch.stack([h.real, h.imag], dim=1)


def real_channels_to_complex(h: torch.Tensor) -> torch.Tensor:
    """Inverse of complex_to_real_channels."""
    return torch.complex(h[:, 0], h[:, 1])


def normalize_csi(
    h: torch.Tensor,
    dim: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample zero-mean unit-variance normalization.

    Returns:
        normalized_h, norm_stats (for denormalization).
    """
    if dim is None:
        dim = tuple(range(1, h.ndim))
    mean = h.mean(dim=dim, keepdim=True)
    std = h.abs().square().mean(dim=dim, keepdim=True).sqrt().clamp_min(eps)
    h_norm = (h - mean) / std
    return h_norm, torch.stack([mean, std], dim=0)


def denormalize_csi(
    h_norm: torch.Tensor,
    norm_stats: torch.Tensor,
) -> torch.Tensor:
    """Undo normalize_csi using stored [mean, std] stats."""
    mean, std = norm_stats[0], norm_stats[1]
    return h_norm * std + mean


def apply_delay_window(h: torch.Tensor, frequency_dim: int = -1) -> torch.Tensor:
    """Apply Hann window along delay domain to suppress leakage.

    This is performed in the delay domain, i.e., after spatial->angle-delay.
    """
    n = h.shape[frequency_dim]
    window = torch.hann_window(n, device=h.device, dtype=h.dtype).reshape(
        [1] * (h.ndim + frequency_dim) + [n]
    )
    return h * window


class AngleDelayTransform:
    """Callable transform object for dataset pipelines."""

    def __init__(
        self,
        antenna_dim: int = -2,
        frequency_dim: int = -1,
        use_2d_antenna_dft: bool = False,
        apply_window: bool = False,
        normalize: bool = True,
    ):
        self.antenna_dim = antenna_dim
        self.frequency_dim = frequency_dim
        self.use_2d_antenna_dft = use_2d_antenna_dft
        self.apply_window = apply_window
        self.normalize = normalize

    def __call__(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h_ad = spatial_to_angle_delay(
            h,
            antenna_dim=self.antenna_dim,
            frequency_dim=self.frequency_dim,
            use_2d_antenna_dft=self.use_2d_antenna_dft,
        )
        if self.apply_window:
            h_ad = apply_delay_window(h_ad, frequency_dim=self.frequency_dim)

        stats = None
        if self.normalize:
            h_ad, stats = normalize_csi(h_ad)
        return h_ad, stats

    def inverse(
        self,
        h_ad: torch.Tensor,
        stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if stats is not None:
            h_ad = denormalize_csi(h_ad, stats)
        return angle_delay_to_spatial(
            h_ad,
            antenna_dim=self.antenna_dim,
            frequency_dim=self.frequency_dim,
            use_2d_antenna_dft=self.use_2d_antenna_dft,
        )
