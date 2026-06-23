"""Magnitude-guided cross-attention for Phase 2."""
import torch
import torch.nn as nn


class MagnitudeGuidedAttention(nn.Module):
    """Cross-attention where sparse subband features query magnitude features.

    Input:
        query: [B, N_s, M, feat_dim] from sparse encoder.
        key:   [B, N_s, M, feat_dim] from magnitude encoder.
        value: [B, N_s, M, feat_dim] from magnitude encoder.

    Output:
        attended: [B, N_s, M, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(feat_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        B, N_s, M, D = query.shape
        # Operate independently per spatial position; subcarrier is the sequence.
        q = query.reshape(B * N_s, M, D)
        k = key.reshape(B * N_s, M, D)
        v = value.reshape(B * N_s, M, D)

        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)

        out, _ = self.attn(q, k, v, need_weights=False)
        out = out.reshape(B, N_s, M, D)
        out = self.norm_out(out + query)
        return out
