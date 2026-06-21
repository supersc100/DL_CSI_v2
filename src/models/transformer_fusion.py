"""Lightweight Transformer fusion block for multi-modal CSI features.

This module replaces the frozen DeepSeek-LLM backbone in the original design.
It operates on a small fixed number of soft tokens (e.g. current UL, temporal,
and environment features) and models cross-modal interactions via a standard
PyTorch TransformerEncoder.
"""
import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    """Fuse a fixed number of modality tokens via a small Transformer.

    Input shape:  [B, num_tokens, feature_dim]
    Output shape: [B, num_tokens, hidden_dim]

    Args:
        feature_dim: Dimension of each input modality token.
        hidden_dim: Hidden dimension of the Transformer (output dimension).
        num_layers: Number of TransformerEncoder layers.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio for the feed-forward hidden dim relative to hidden_dim.
        dropout: Dropout probability.
        num_tokens: Number of input tokens. Defaults to 3 (current, temporal, env).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_tokens: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        # Project features to transformer hidden dim only if needed.
        self.input_proj = (
            nn.Linear(feature_dim, hidden_dim)
            if feature_dim != hidden_dim
            else nn.Identity()
        )

        # Learnable positional embeddings for the fixed token slots.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_tokens, feature_dim]
        x = self.input_proj(x)  # [B, num_tokens, hidden_dim]
        x = x + self.pos_embed
        x = self.transformer(x)  # [B, num_tokens, hidden_dim]
        x = self.norm(x)
        return x
