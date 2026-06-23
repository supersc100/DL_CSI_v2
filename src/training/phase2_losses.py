"""Loss functions for Phase 2 phase recovery."""
import torch
import torch.nn as nn


class Phase2Loss(nn.Module):
    """Combined loss for full downlink CSI recovery.

    L_total = L_mse + lambda_phase * L_phase_weighted + lambda_angle * L_angle_l1

    All inputs are complex tensors in the angle-delay domain.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        phase_weight: float = 0.5,
        angle_l1_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.phase_weight = phase_weight
        self.angle_l1_weight = angle_l1_weight
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mag_stage1: torch.Tensor,
    ) -> tuple:
        """Compute loss.

        Args:
            pred: [B, N_tx, N_rx, M] complex, predicted full DL CSI.
            target: [B, N_tx, N_rx, M] complex, true DL CSI.
            mag_stage1: [B, N_tx, N_rx, M] real, Stage1 predicted magnitude.

        Returns:
            loss, dict of component losses.
        """
        # Full complex MSE.
        diff = pred - target
        mse = (diff.real ** 2 + diff.imag ** 2).mean()

        # Magnitude-weighted phase MSE.
        phase_mse = self._phase_weighted_mse(pred, target, mag_stage1)

        # Angle-delay domain magnitude L1 consistency.
        # Since pred/target are already in AD domain, this is simply L1 on
        # magnitudes.
        angle_l1 = (pred.abs() - target.abs()).abs().mean()

        loss = (
            self.mse_weight * mse
            + self.phase_weight * phase_mse
            + self.angle_l1_weight * angle_l1
        )

        return loss, {
            "loss_mse": mse.detach(),
            "loss_phase": phase_mse.detach(),
            "loss_angle": angle_l1.detach(),
            "loss_total": loss.detach(),
        }

    def _phase_weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mag_stage1: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted phase MSE where errors on strong paths are penalized more.

        Uses a numerically stable 1 - cos(delta_theta) formulation instead of
        asin to avoid infinite gradients at the +/- 1 boundary in float32.
        """
        # Normalize phase components using a safe magnitude to avoid NaN
        # gradients when pred/target are exactly zero.
        pred_phase = pred / torch.sqrt(pred.real ** 2 + pred.imag ** 2 + self.eps)
        target_phase = target / torch.sqrt(target.real ** 2 + target.imag ** 2 + self.eps)

        # Phase difference as exp(j*delta_theta) = pred_phase * conj(target_phase).
        phase_diff = pred_phase * target_phase.conj()
        # cos(delta_theta) = real part; 1 - cos(delta) is a smooth phase metric.
        cos_delta = phase_diff.real.clamp(-1.0, 1.0)
        phase_error = 1.0 - cos_delta

        # Magnitude weights: larger weights on stronger paths.
        weights = mag_stage1 / (mag_stage1.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps))
        weights = weights.detach()  # keep magnitude prior fixed for loss weighting

        phase_mse = (weights * (phase_error ** 2)).mean()
        return phase_mse
