"""Phase prediction head for Phase 2."""
import torch
import torch.nn as nn


class PhaseHead(nn.Module):
    """Predict phase as cos/sin to avoid 2pi wrapping.

    Input:
        x: [B, N_s, M, feat_dim]

    Output:
        phase: [B, N_tx, N_rx, M] complex with unit magnitude, i.e. exp(j*theta)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        num_tx: int = 32,
        num_rx: int = 4,
    ):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_s, M, D = x.shape
        x = x.reshape(B, N_s * M, D)
        x = self.mlp(x)  # [B, N_s*M, 2]

        # L2-normalize to unit magnitude (cos^2 + sin^2 = 1).
        # Add a small epsilon to avoid NaN when the MLP outputs near-zero vectors.
        norm = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        cos = x[..., 0] / norm
        sin = x[..., 1] / norm

        phase_complex = torch.complex(cos, sin)
        phase_complex = phase_complex.reshape(B, self.num_tx, self.num_rx, M)
        return phase_complex
