"""Magnitude encoder for Phase 2."""
import torch
import torch.nn as nn


class MagnitudeEncoder(nn.Module):
    """Encode the Stage1 predicted magnitude into per-subcarrier features.

    Input:
        mag: [B, N_tx, N_rx, M] real, Stage1 predicted magnitude.

    Output:
        features: [B, N_tx * N_rx, M, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        channels = [1] + [min(feat_dim, 32 * (2 ** i)) for i in range(num_layers)]

        layers = []
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            layers.append(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=(1, 1, kernel_size),
                    padding=(0, 0, kernel_size // 2),
                    bias=False,
                )
            )
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))

        self.conv_layers = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], feat_dim)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # Add channel dim: [B, 1, N_tx, N_rx, M]
        x = mag.unsqueeze(1)
        x = self.conv_layers(x)  # [B, C, N_tx, N_rx, M]
        x = x.permute(0, 2, 3, 4, 1)  # [B, N_tx, N_rx, M, C]
        x = self.proj(x)  # [B, N_tx, N_rx, M, feat_dim]
        B, N_tx, N_rx, M, D = x.shape
        x = x.reshape(B, N_tx * N_rx, M, D)
        return x
