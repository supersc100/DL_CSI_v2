"""Sparse subband CNN encoder for Phase 2."""
import torch
import torch.nn as nn

from src.data.transforms import complex_to_real_channels


class FreqSparseEncoder(nn.Module):
    """Encode a zero-padded sparse downlink subband grid into per-subcarrier
    feature tokens.

    Input:
        sparse_dl_ad: [B, N_tx, N_rx, M] complex, zero at unsampled subcarriers.
        mask: [B, M] bool, 1 at sampled positions (currently unused by the
            encoder but kept for interface consistency and future mask-aware
            variants).

    Output:
        features: [B, N_tx * N_rx, M, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_layers = num_layers

        # Build a stack of 1D convolutions over the subcarrier dimension.
        # Input channels = 2 (real + imag).
        channels = [2] + [max(32, feat_dim // (2 ** (num_layers - i))) for i in range(num_layers)]
        # Simplify: use a geometric progression ending near feat_dim.
        channels = [2] + [min(feat_dim, 32 * (2 ** i)) for i in range(num_layers)]

        layers = []
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            # Conv only along subcarrier dimension (last axis).
            layers.append(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=(1, 1, kernel_size),
                    padding=(0, 0, kernel_size // 2),
                    bias=False,
                )
            )
            # GroupNorm with single group is equivalent to LayerNorm over channels.
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))

        self.conv_layers = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], feat_dim)

    def forward(
        self,
        sparse_dl_ad: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # mask is kept for interface compatibility.
        _ = mask

        # [B, N_tx, N_rx, M] -> [B, 2, N_tx, N_rx, M]
        x = complex_to_real_channels(sparse_dl_ad)

        # Apply frequency-domain CNN.
        x = self.conv_layers(x)  # [B, C, N_tx, N_rx, M]

        # Move channels to last dim and project.
        x = x.permute(0, 2, 3, 4, 1)  # [B, N_tx, N_rx, M, C]
        x = self.proj(x)  # [B, N_tx, N_rx, M, feat_dim]

        # Flatten spatial dims.
        B, N_tx, N_rx, M, D = x.shape
        x = x.reshape(B, N_tx * N_rx, M, D)
        return x
