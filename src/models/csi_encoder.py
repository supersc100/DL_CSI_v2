"""Convolutional encoder for current uplink CSI in angle-delay domain."""
from typing import Tuple

import torch
import torch.nn as nn


class CsiEncoder(nn.Module):
    """Encode a single UL CSI tensor into a compact feature vector.

    Input shape:  [B, 2, N_tx, N_rx, M]
    Output shape: [B, feature_dim]
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        feature_dim: int = 512,
        use_bn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(
                nn.Conv3d(
                    channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False if use_bn else True,
                )
            )
            if use_bn:
                layers.append(nn.BatchNorm3d(out_channels))
            else:
                # GroupNorm is the CNN-friendly equivalent of LayerNorm over channels.
                num_groups = min(8, out_channels)
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
            layers.append(nn.GELU())
            # Avoid pooling the UE-antenna (N_rx) dimension so that small UE arrays
            # (e.g. 4 or 8) do not collapse to size 0 after repeated halving. The
            # BS-angle (N_tx) and subcarrier (M) dims are still pooled normally.
            layers.append(
                nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2), padding=0)
            )
            layers.append(nn.Dropout3d(dropout))
            channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(channels, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, N_tx, N_rx, M]
        x = self.conv(x)
        x = self.global_pool(x).flatten(1)
        x = self.projection(x)
        x = self.norm(x)
        return x


class CsiPatchEncoder(nn.Module):
    """Alternative encoder that treats CSI as 2D patches over angle-delay."""

    def __init__(
        self,
        in_channels: int = 2,
        patch_size: Tuple[int, int, int] = (4, 2, 4),
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        feature_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, N_tx, N_rx, M]
        x = self.proj(x)  # [B, E, P_tx, P_rx, P_m]
        B, E, *spatial = x.shape
        x = x.view(B, E, -1).transpose(1, 2)  # [B, num_patches, E]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x
