"""Loss functions for FDD downlink CSI prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CsiLoss(nn.Module):
    """Combined MSE + angle-delay L1 consistency loss.

    Forward receives complex-valued predictions and targets in the angle-delay
    domain. All internal computations are performed in float32 for numerical
    stability, regardless of the model's forward dtype.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        angle_delay_l1_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.angle_delay_l1_weight = angle_delay_l1_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pred/target are complex64 from the regression head / dataset.
        # Real/imag extraction yields float32 automatically, satisfying the
        # "loss in float32" requirement.
        diff = pred - target
        mse = (diff.real ** 2 + diff.imag ** 2).mean()

        # Angle-delay consistency: L1 on magnitude to preserve large-scale structure.
        ad_l1 = (pred.abs() - target.abs()).abs().mean()

        loss = self.mse_weight * mse + self.angle_delay_l1_weight * ad_l1
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
