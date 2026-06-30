"""Loss functions for FDD downlink CSI prediction."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CsiLoss(nn.Module):
    """Combined complex MSE + magnitude MSE + angle-delay L1 consistency loss.

    Forward receives complex-valued predictions and targets in the angle-delay
    domain. All internal computations are performed in float32 for numerical
    stability, regardless of the model's forward dtype.

    The magnitude term is the workhorse for FDD UL->DL prediction because UL and
    DL share large-scale geometry (angles/delays/powers) but have independent
    small-scale phases.  The complex term is kept as an optional auxiliary loss
    and can be reduced/removed when only magnitude structure is required.

    Args:
        use_ratio: If True, magnitude terms are computed in the ratio space
            |H| / |H_UL| instead of raw |H|. This forces the network to use the
            uplink CSI because a fixed |H_DL| template no longer minimizes the
            loss across samples with different |H_UL|.
    """

    def __init__(
        self,
        mse_weight: float = 0.0,
        magnitude_weight: float = 1.0,
        angle_delay_l1_weight: float = 0.1,
        diversity_weight: float = 0.0,
        use_ratio: bool = False,
        ratio_eps: float = 1e-6,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.magnitude_weight = magnitude_weight
        self.angle_delay_l1_weight = angle_delay_l1_weight
        self.diversity_weight = diversity_weight
        self.use_ratio = use_ratio
        self.ratio_eps = ratio_eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        current_ul: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pred/target are complex64 from the regression head / dataset.
        # Real/imag extraction yields float32 automatically, satisfying the
        # "loss in float32" requirement.
        diff = pred - target
        mse = (diff.real ** 2 + diff.imag ** 2).mean()

        # Magnitude terms. Optionally normalize by |H_UL| so the network must
        # exploit the uplink structure instead of outputting a fixed template.
        # We use log-ratio (log|H| - log|H_UL|) rather than raw ratio because
        # dividing by small UL magnitudes leads to gradient explosion.
        pred_mag = pred.abs()
        target_mag = target.abs()
        if self.use_ratio:
            if current_ul is None:
                raise ValueError("use_ratio=True requires current_ul to be passed to the loss.")
            ul_mag = current_ul.abs()
            pred_mag = torch.log(pred_mag + self.ratio_eps) - torch.log(ul_mag + self.ratio_eps)
            target_mag = torch.log(target_mag + self.ratio_eps) - torch.log(ul_mag + self.ratio_eps)

        # Magnitude MSE: the predictable part under FDD independent fast fading.
        magnitude_mse = (pred_mag - target_mag).square().mean()

        # Angle-delay consistency: L1 on magnitude to preserve large-scale structure.
        ad_l1 = (pred_mag - target_mag).abs().mean()

        loss = (
            self.mse_weight * mse
            + self.magnitude_weight * magnitude_mse
            + self.angle_delay_l1_weight * ad_l1
        )

        # Diversity regularization: penalize batch-wise collapse to a fixed
        # magnitude template. Encourages the network to use input-specific
        # information instead of memorizing the average spectrum.
        if self.diversity_weight > 0.0 and pred.shape[0] > 1:
            pred_mag_flat = pred_mag.view(pred.shape[0], -1)
            # Variance across the batch for each output dimension.
            diversity = pred_mag_flat.var(dim=0).mean()
            loss = loss - self.diversity_weight * diversity

        return loss


class NmseLoss(nn.Module):
    """Normalized mean squared error per sample, averaged over batch."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        num = (diff.abs() ** 2).sum(dim=list(range(1, diff.ndim)))
        den = (target.abs() ** 2).sum(dim=list(range(1, target.ndim))) + 1e-12
        return (num / den).mean()


class CosineSimilarityLoss(nn.Module):
    """Cosine similarity between flattened complex vectors (per sample)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        B = pred.shape[0]
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)
        pred_real = torch.view_as_real(pred_flat)  # [B, N, 2]
        target_real = torch.view_as_real(target_flat)
        # Treat each complex element as a 2D real vector.
        pred_real = pred_real.view(B, -1)
        target_real = target_real.view(B, -1)
        cos = F.cosine_similarity(pred_real, target_real, dim=-1)
        return cos.mean()
