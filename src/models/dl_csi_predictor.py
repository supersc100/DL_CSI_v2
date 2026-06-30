"""CNN+Transformer based FDD downlink CSI predictor (no LLM)."""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.csi_encoder import CsiEncoder
from src.models.temporal_encoder import TemporalEncoder
from src.models.env_encoder import EnvironmentEncoder
from src.models.regression_head import RegressionHead
from src.models.transformer_fusion import TransformerFusion
from src.data.transforms import complex_to_real_channels


class CrossAttentionFusion(nn.Module):
    """Fuse a variable number of feature tokens via self-attention."""

    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_kv = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(feature_dim)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        # Stack variable number of tokens as [B, num_tokens, feature_dim].
        tokens = torch.stack(features, dim=1)
        # Self-like cross-attention over the tokens.
        q = self.norm_q(tokens)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.norm_out(out + tokens)
        return out


class DlCsiPredictor(nn.Module):
    """End-to-end FDD downlink CSI predictor using CNN encoders + Transformer fusion.

    Inputs:
        - current_ul_ad:  [B, N_tx, N_rx, M] (complex)
        - history_ul_ad:  [B, T, N_tx, N_rx, M] (complex), optional
        - history_dl_ad:  [B, T, N_tx, N_rx, M] (complex), optional
        - large_scale:    [B, num_large_scale], optional

    Outputs:
        - pred_dl_ad:     [B, N_tx, N_rx, M] (complex)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = int(config.model.feature_dim)
        self.hidden_dim = int(config.model.get("hidden_dim", self.feature_dim))
        self.output_mode = str(config.model.regression_head.output_mode)
        self.use_history = bool(config.model.get("use_history", True))
        self.use_large_scale = bool(config.model.get("use_large_scale", True))

        # Local encoders.
        self.csi_encoder = CsiEncoder(
            in_channels=int(config.model.csi_encoder.in_channels),
            base_channels=int(config.model.csi_encoder.base_channels),
            num_layers=int(config.model.csi_encoder.num_layers),
            kernel_size=int(config.model.csi_encoder.kernel_size),
            feature_dim=self.feature_dim,
            use_bn=bool(config.model.csi_encoder.use_bn),
            dropout=float(config.model.csi_encoder.dropout),
        )

        if self.use_history:
            self.temporal_encoder = TemporalEncoder(
                csi_in_channels=int(config.model.temporal_encoder.csi.in_channels),
                csi_base_channels=int(config.model.temporal_encoder.csi.base_channels),
                csi_num_layers=int(config.model.temporal_encoder.csi.num_layers),
                csi_kernel_size=int(config.model.temporal_encoder.csi.kernel_size),
                csi_feature_dim=int(config.model.temporal_encoder.csi.csi_feature_dim),
                transformer_hidden_dim=int(config.model.temporal_encoder.transformer.hidden_dim),
                transformer_num_layers=int(config.model.temporal_encoder.transformer.num_layers),
                transformer_num_heads=int(config.model.temporal_encoder.transformer.num_heads),
                transformer_mlp_ratio=float(config.model.temporal_encoder.transformer.mlp_ratio),
                feature_dim=self.feature_dim,
                dropout=float(config.model.temporal_encoder.transformer.dropout),
            )
        else:
            self.temporal_encoder = None

        if self.use_large_scale:
            self.env_encoder = EnvironmentEncoder(
                input_dim=int(config.model.env_encoder.input_dim),
                hidden_dims=list(config.model.env_encoder.hidden_dims),
                output_dim=self.feature_dim,
                dropout=float(config.model.env_encoder.dropout),
            )
        else:
            self.env_encoder = None

        # Optional cross-attention fusion.
        if bool(config.model.fusion.use_cross_attention):
            self.fusion = CrossAttentionFusion(
                feature_dim=self.feature_dim,
                num_heads=int(config.model.fusion.num_heads),
                dropout=float(config.model.fusion.dropout),
            )
        else:
            self.fusion = None

        # Lightweight Transformer fusion (replaces the frozen LLM backbone).
        num_tokens = 1 + (1 if self.use_history else 0) + (1 if self.use_large_scale else 0)
        self.transformer_fusion = TransformerFusion(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=int(config.model.transformer_fusion.num_layers),
            num_heads=int(config.model.transformer_fusion.num_heads),
            mlp_ratio=float(config.model.transformer_fusion.mlp_ratio),
            dropout=float(config.model.transformer_fusion.dropout),
            num_tokens=num_tokens,
        )

        # Output head.
        num_tx = int(config.data.bs_array.num_elements)
        num_rx = int(config.data.ue_array.num_elements)
        num_subcarriers = int(config.data.num_subcarriers)
        self.csi_shape = (num_tx, num_rx, num_subcarriers)
        num_elements = num_tx * num_rx * num_subcarriers
        output_dim = num_elements * 2  # real + imag

        self.use_residual_log_amp = bool(
            config.model.regression_head.get("use_residual_log_amp", False)
        )
        self.residual_eps = float(
            config.model.regression_head.get("residual_eps", 1e-6)
        )

        self.regression_head = RegressionHead(
            input_dim=self.hidden_dim,
            output_dim=output_dim,
            hidden_dim=int(config.model.regression_head.hidden_dim),
            output_mode=self.output_mode,
            dropout=float(config.model.regression_head.dropout),
            use_residual_log_amp=self.use_residual_log_amp,
            residual_eps=self.residual_eps,
        )

    def _prepare_inputs(
        self,
        current_ul_ad: torch.Tensor,
        large_scale: Optional[torch.Tensor] = None,
        history_ul_ad: Optional[torch.Tensor] = None,
        history_dl_ad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Convert complex CSI to real-channel representation.
        # current: [B, N_tx, N_rx, M] -> [B, 2, N_tx, N_rx, M]
        current_ul_ri = complex_to_real_channels(current_ul_ad)

        # Encode each modality.
        current_feat = self.csi_encoder(current_ul_ri)  # [B, feature_dim]

        features = [current_feat]

        if self.use_large_scale:
            if large_scale is None:
                raise ValueError("use_large_scale=True but large_scale not provided.")
            env_feat = self.env_encoder(large_scale)  # [B, feature_dim]
            features.append(env_feat)

        if self.use_history:
            if history_ul_ad is None or history_dl_ad is None:
                raise ValueError(
                    "use_history=True but history_ul_ad/history_dl_ad not provided."
                )
            # history: [B, T, N_tx, N_rx, M] -> [B, T, 2, N_tx, N_rx, M]
            history_ul_ri = complex_to_real_channels(history_ul_ad).permute(0, 2, 1, 3, 4, 5)
            history_dl_ri = complex_to_real_channels(history_dl_ad).permute(0, 2, 1, 3, 4, 5)
            temporal_feat = self.temporal_encoder(history_ul_ri, history_dl_ri)  # [B, feature_dim]
            features.append(temporal_feat)

        # Fuse variable number of tokens.
        if self.fusion is not None:
            tokens = self.fusion(*features)  # [B, num_tokens, feature_dim]
        else:
            tokens = torch.stack(features, dim=1)

        return tokens

    def forward(
        self,
        current_ul_ad: torch.Tensor,
        large_scale: Optional[torch.Tensor] = None,
        history_ul_ad: Optional[torch.Tensor] = None,
        history_dl_ad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Prepare modality tokens.
        tokens = self._prepare_inputs(
            current_ul_ad, large_scale, history_ul_ad, history_dl_ad
        )  # [B, num_tokens, feature_dim]

        # Transformer fusion.
        transformer_out = self.transformer_fusion(tokens)  # [B, num_tokens, hidden_dim]

        # Pool over tokens.
        pooled = transformer_out.mean(dim=1)  # [B, hidden_dim]

        # Predict downlink CSI in angle-delay domain (keep float32 for stability).
        with torch.amp.autocast("cuda", enabled=False):
            head_kwargs = {}
            if self.use_residual_log_amp:
                # Flatten current UL magnitude and take log for the residual head.
                b = current_ul_ad.shape[0]
                ul_log_mag = torch.log(
                    current_ul_ad.abs().view(b, -1).float() + self.residual_eps
                )
                head_kwargs["ul_log_mag"] = ul_log_mag
            pred_dl_ad = self.regression_head(
                pooled.float(), target_shape=(-1, *self.csi_shape), **head_kwargs
            )
        return pred_dl_ad

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    # Quick smoke test.
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.config import load_config

    cfg = load_config()
    model = DlCsiPredictor(cfg)
    print(model.count_parameters())
