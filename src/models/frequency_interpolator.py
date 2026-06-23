"""Frequency-domain Transformer interpolator for Phase 2."""
import math

import torch
import torch.nn as nn


class SubcarrierPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over subcarriers plus a learnable
    sampled/unsampled embedding.
    """

    def __init__(self, num_subcarriers: int, feat_dim: int):
        super().__init__()
        self.num_subcarriers = num_subcarriers
        self.feat_dim = feat_dim

        pe = torch.zeros(num_subcarriers, feat_dim)
        position = torch.arange(0, num_subcarriers, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feat_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / feat_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Learnable embedding indicating whether a position is sampled.
        self.sampled_embed = nn.Embedding(2, feat_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add positional and sampled embeddings.

        Args:
            x: [B, N_s, M, feat_dim]
            mask: [B, M] bool, True for sampled positions.

        Returns:
            x: [B, N_s, M, feat_dim]
        """
        B, N_s, M, D = x.shape
        # Add sinusoidal PE (broadcast over batch and spatial dim).
        x = x + self.pe.view(1, 1, M, D)

        # Add learnable sampled/unsampled embedding.
        # sampled_idx: [B, M] int, 1 if sampled else 0.
        sampled_idx = mask.long()  # [B, M]
        sampled_emb = self.sampled_embed(sampled_idx)  # [B, M, D]
        x = x + sampled_emb.unsqueeze(1)  # broadcast over N_s
        return x


class FrequencyInterpolator(nn.Module):
    """Transformer encoder over the subcarrier dimension.

    Input:
        x: [B, N_s, M, feat_dim]
        mask: [B, M] bool

    Output:
        x: [B, N_s, M, feat_dim]
    """

    def __init__(
        self,
        num_subcarriers: int,
        feat_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_subcarriers = num_subcarriers

        self.pos_enc = SubcarrierPositionalEncoding(num_subcarriers, feat_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=int(feat_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N_s, M, D = x.shape
        x = self.pos_enc(x, mask)

        # Merge batch and spatial dims; subcarrier becomes sequence length.
        x = x.reshape(B * N_s, M, D)
        x = self.transformer(x)  # [B*N_s, M, D]
        x = x.reshape(B, N_s, M, D)
        return x
