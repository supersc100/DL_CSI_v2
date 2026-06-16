"""Temporal encoder for historical UL/DL CSI sequences."""
import torch
import torch.nn as nn

from src.models.csi_encoder import CsiEncoder


class TemporalEncoder(nn.Module):
    """Encode T historical CSI pairs into a single temporal feature vector.

    Input shape:  [B, T, 2, N_tx, N_rx, M]
    Output shape: [B, feature_dim]
    """

    def __init__(
        self,
        csi_in_channels: int = 2,
        csi_base_channels: int = 32,
        csi_num_layers: int = 3,
        csi_kernel_size: int = 3,
        csi_feature_dim: int = 256,
        transformer_hidden_dim: int = 256,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 4,
        transformer_mlp_ratio: float = 4.0,
        feature_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.csi_feature_dim = csi_feature_dim

        # Per-slot CSI compressor.
        self.per_slot_encoder = CsiEncoder(
            in_channels=csi_in_channels,
            base_channels=csi_base_channels,
            num_layers=csi_num_layers,
            kernel_size=csi_kernel_size,
            feature_dim=csi_feature_dim,
            dropout=dropout,
        )

        # Project paired UL/DL slot features into transformer hidden dim.
        self.pair_proj = nn.Linear(csi_feature_dim * 2, transformer_hidden_dim)

        # Positional embedding for temporal order.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, transformer_hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Lightweight Transformer over time.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=transformer_num_heads,
            dim_feedforward=int(transformer_hidden_dim * transformer_mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(transformer_hidden_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, history_ul: torch.Tensor, history_dl: torch.Tensor) -> torch.Tensor:
        # history_*: [B, T, 2, N_tx, N_rx, M]
        B, T = history_ul.shape[:2]

        ul_features = []
        dl_features = []
        for t in range(T):
            ul_features.append(self.per_slot_encoder(history_ul[:, t]))
            dl_features.append(self.per_slot_encoder(history_dl[:, t]))

        ul_features = torch.stack(ul_features, dim=1)  # [B, T, csi_feature_dim]
        dl_features = torch.stack(dl_features, dim=1)
        pair_features = torch.cat([ul_features, dl_features], dim=-1)
        pair_features = self.pair_proj(pair_features)

        # Add positional embedding.
        pair_features = pair_features + self.pos_embed[:, :T, :]
        pair_features = self.transformer(pair_features)

        # Temporal pooling.
        pooled = pair_features.transpose(1, 2)  # [B, D, T]
        pooled = self.time_pool(pooled).squeeze(-1)
        out = self.output_proj(pooled)
        out = self.norm(out)
        return out
