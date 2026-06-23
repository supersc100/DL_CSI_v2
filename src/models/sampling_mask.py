"""Adaptive hybrid subcarrier sampling mask generator for Phase 2.

The mask is applied in the spatial-frequency domain before the angle-delay
transform.  It combines a uniform base grid (density adapted to the RMS delay
spread) with non-uniform peaks detected from the uplink energy spectrum.
"""
from typing import Optional

import torch


class SamplingMaskGenerator:
    """Generate binary subcarrier sampling masks.

    Args:
        num_subcarriers: total number of subcarriers M.
        use_adaptive: if True, choose base spacing from RMS delay spread.
        base_spacing: default base spacing when use_adaptive is False.
        use_nonuniform_peaks: if True, add extra samples around UL energy peaks.
        peak_prob: probability of enabling peak encryption during training.
        peak_neighbors: number of neighbors on each side of a peak to sample.
        random_offset: if True, randomize the uniform grid offset during training.
        train_spacings: candidate spacings used for training randomization.
    """

    def __init__(
        self,
        num_subcarriers: int,
        use_adaptive: bool = True,
        base_spacing: int = 8,
        use_nonuniform_peaks: bool = True,
        peak_prob: float = 0.7,
        peak_neighbors: int = 2,
        random_offset: bool = True,
        train_spacings: tuple = (4, 8, 16),
    ):
        self.num_subcarriers = int(num_subcarriers)
        self.use_adaptive = bool(use_adaptive)
        self.base_spacing = int(base_spacing)
        self.use_nonuniform_peaks = bool(use_nonuniform_peaks)
        self.peak_prob = float(peak_prob)
        self.peak_neighbors = int(peak_neighbors)
        self.random_offset = bool(random_offset)
        self.train_spacings = tuple(int(s) for s in train_spacings)

    def _choose_spacing(self, rms_delay_spread: Optional[float]) -> int:
        """Pick base spacing for evaluation (no randomization)."""
        if self.use_adaptive and rms_delay_spread is not None:
            if rms_delay_spread < 100e-9:
                return 16
            if rms_delay_spread < 350e-9:
                return 8
            return 4
        return self.base_spacing

    def __call__(
        self,
        h_ul: torch.Tensor,
        large_scale: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> torch.Tensor:
        """Generate a binary mask of shape [M].

        Args:
            h_ul: [N_tx, N_rx, M] complex, current uplink CSI (spatial-freq).
            large_scale: [D_lsp] optional large-scale parameter vector.  The
                second element is assumed to be RMS delay spread if available.
            training: whether to apply training-time randomization.

        Returns:
            mask: [M] bool tensor, True at sampled subcarrier indices.
        """
        M = self.num_subcarriers
        if h_ul.shape[-1] != M:
            raise ValueError(
                f"h_ul last dim {h_ul.shape[-1]} != num_subcarriers {M}"
            )

        # RMS delay spread from large_scale if available.
        rms_delay_spread = None
        if large_scale is not None and len(large_scale) >= 2:
            rms_delay_spread = float(large_scale[1])

        # Choose spacing.
        if training:
            spacing = int(self.train_spacings[
                torch.randint(0, len(self.train_spacings), (1,)).item()
            ])
            offset = int(torch.randint(0, spacing, (1,)).item())
        else:
            spacing = self._choose_spacing(rms_delay_spread)
            offset = 0

        # Uniform base grid.
        base_indices = list(range(offset, M, spacing))

        # Non-uniform peak encryption from UL energy.
        extra_indices = []
        if self.use_nonuniform_peaks:
            enable_peaks = (not training) or (torch.rand(1).item() < self.peak_prob)
            if enable_peaks:
                # Energy per subcarrier (sum over spatial dims).
                ul_energy = h_ul.abs().square().sum(dim=(0, 1))  # [M]
                topk = min(3, max(1, M // spacing))
                peak_indices = torch.topk(ul_energy, k=topk).indices.tolist()
                for peak in peak_indices:
                    for delta in range(-self.peak_neighbors, self.peak_neighbors + 1):
                        idx = peak + delta
                        if 0 <= idx < M and idx not in base_indices:
                            extra_indices.append(idx)

        all_indices = sorted(set(base_indices + extra_indices))
        mask = torch.zeros(M, dtype=torch.bool)
        mask[all_indices] = True
        return mask
