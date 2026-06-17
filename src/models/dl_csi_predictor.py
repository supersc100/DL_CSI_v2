"""Main DeepSeek-LLM based FDD downlink CSI predictor."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.csi_encoder import CsiEncoder
from src.models.temporal_encoder import TemporalEncoder
from src.models.env_encoder import EnvironmentEncoder
from src.models.embedding_projection import ContinuousEmbeddingProjection
from src.models.regression_head import RegressionHead
from src.data.transforms import complex_to_real_channels, real_channels_to_complex


class CrossAttentionFusion(nn.Module):
    """Fuse three feature tokens via cross-attention."""

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

    def forward(
        self,
        current: torch.Tensor,
        temporal: torch.Tensor,
        env: torch.Tensor,
    ) -> torch.Tensor:
        # Stack as [B, 3, feature_dim]
        tokens = torch.stack([current, temporal, env], dim=1)
        # Self-like cross-attention over the three tokens.
        q = self.norm_q(tokens)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.norm_out(out + tokens)
        return out


class DlCsiPredictor(nn.Module):
    """End-to-end FDD downlink CSI predictor.

    Inputs:
        - current_ul_ad:  [B, N_tx, N_rx, M] (complex)
        - history_ul_ad:  [B, T, N_tx, N_rx, M] (complex)
        - history_dl_ad:  [B, T, N_tx, N_rx, M] (complex)
        - large_scale:    [B, num_large_scale]

    Outputs:
        - pred_dl_ad:     [B, N_tx, N_rx, M] (complex)
    """

    def __init__(self, config, llm: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.feature_dim = int(config.model.feature_dim)
        self.llm_hidden_dim = int(config.model.llm_hidden_dim)
        self.num_virtual_tokens = int(config.model.num_virtual_tokens)
        self.output_mode = str(config.model.regression_head.output_mode)

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

        self.env_encoder = EnvironmentEncoder(
            input_dim=int(config.model.env_encoder.input_dim),
            hidden_dims=list(config.model.env_encoder.hidden_dims),
            output_dim=self.feature_dim,
            dropout=float(config.model.env_encoder.dropout),
        )

        # Optional cross-attention fusion.
        if bool(config.model.fusion.use_cross_attention):
            self.fusion = CrossAttentionFusion(
                feature_dim=self.feature_dim,
                num_heads=int(config.model.fusion.num_heads),
                dropout=float(config.model.fusion.dropout),
            )
        else:
            self.fusion = None

        # Continuous projection to LLM hidden space.
        self.embedding_projection = ContinuousEmbeddingProjection(
            feature_dim=self.feature_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            num_virtual_tokens=self.num_virtual_tokens,
        )

        # DeepSeek / Qwen backbone, or a caller-supplied LLM (e.g. for smoke tests).
        self.llm = llm if llm is not None else self._build_llm(config)
        self.tokenizer = None  # loaded lazily in inference scripts if needed.

        # Output head.
        # Output shape: [B, N_tx, N_rx, M]
        num_tx = int(config.data.bs_array.num_elements)
        num_rx = int(config.data.ue_array.num_elements)
        num_subcarriers = int(config.data.num_subcarriers)
        self.csi_shape = (num_tx, num_rx, num_subcarriers)
        num_elements = num_tx * num_rx * num_subcarriers
        output_dim = num_elements * 2  # real + imag

        self.regression_head = RegressionHead(
            llm_hidden_dim=self.llm_hidden_dim,
            output_dim=output_dim,
            hidden_dim=int(config.model.regression_head.hidden_dim),
            output_mode=self.output_mode,
            dropout=float(config.model.regression_head.dropout),
        )

    def _build_llm(self, config) -> nn.Module:
        llm_path = str(config.model.llm_path)
        if not os.path.isabs(llm_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            llm_path = os.path.join(project_root, llm_path)

        if not os.path.isdir(llm_path):
            raise FileNotFoundError(
                f"DeepSeek model weights not found at {llm_path}. "
                "Download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B and place it there."
            )

        dtype = torch.bfloat16 if bool(config.project.mixed_precision) else torch.float32
        device = str(config.project.device)

        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        # Freeze all LLM parameters by default.
        for param in model.parameters():
            param.requires_grad = False

        # Apply LoRA adapters to the transformer.
        lora_cfg = LoraConfig(
            r=int(config.lora.r),
            lora_alpha=int(config.lora.lora_alpha),
            target_modules=list(config.lora.target_modules),
            lora_dropout=float(config.lora.lora_dropout),
            bias=str(config.lora.bias),
            task_type=str(config.lora.task_type),
            modules_to_save=list(config.lora.modules_to_save),
        )
        model = get_peft_model(model, lora_cfg)

        # Replace the native token embedding with our continuous projection.
        # We cannot literally swap nn.Embedding for nn.Linear due to incompatible
        # signatures; instead we bypass the embedding by feeding `inputs_embeds`
        # during forward. Here we register a placeholder so the model knows the
        # token embedding is intentionally overridden.
        model.get_input_embeddings().requires_grad_(False)
        return model

    def _prepare_inputs(
        self,
        current_ul_ad: torch.Tensor,
        history_ul_ad: torch.Tensor,
        history_dl_ad: torch.Tensor,
        large_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Convert complex CSI to real-channel representation.
        # current: [B, N_tx, N_rx, M] -> [B, 2, N_tx, N_rx, M]
        # history: [B, T, N_tx, N_rx, M] -> [B, 2, T, N_tx, N_rx, M] -> [B, T, 2, N_tx, N_rx, M]
        current_ul_ri = complex_to_real_channels(current_ul_ad)
        history_ul_ri = complex_to_real_channels(history_ul_ad).permute(0, 2, 1, 3, 4, 5)
        history_dl_ri = complex_to_real_channels(history_dl_ad).permute(0, 2, 1, 3, 4, 5)

        # Encode each modality.
        current_feat = self.csi_encoder(current_ul_ri)  # [B, feature_dim]
        temporal_feat = self.temporal_encoder(history_ul_ri, history_dl_ri)
        env_feat = self.env_encoder(large_scale)

        # Fuse.
        if self.fusion is not None:
            tokens = self.fusion(current_feat, temporal_feat, env_feat)  # [B, 3, feature_dim]
        else:
            tokens = torch.stack([current_feat, temporal_feat, env_feat], dim=1)

        # Project to LLM hidden space.
        inputs_embeds = self.embedding_projection(tokens)  # [B, 3, llm_hidden_dim]
        return inputs_embeds

    def forward(
        self,
        current_ul_ad: torch.Tensor,
        history_ul_ad: torch.Tensor,
        history_dl_ad: torch.Tensor,
        large_scale: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self._prepare_inputs(
            current_ul_ad, history_ul_ad, history_dl_ad, large_scale
        )

        # Forward through LLM backbone (frozen + LoRA adapters).
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # Pool over virtual tokens.
        last_hidden = outputs.hidden_states[-1]  # [B, 3, llm_hidden_dim]
        pooled = last_hidden.mean(dim=1)  # [B, llm_hidden_dim]

        # Predict downlink CSI in angle-delay domain.
        pred_dl_ad = self.regression_head(pooled, target_shape=(-1, *self.csi_shape))
        return pred_dl_ad

    def freeze_local_encoders(self, freeze: bool = True) -> None:
        for module in [self.csi_encoder, self.temporal_encoder, self.env_encoder, self.embedding_projection, self.regression_head]:
            for param in module.parameters():
                param.requires_grad = not freeze

    def freeze_llm(self, freeze: bool = True) -> None:
        for param in self.llm.parameters():
            param.requires_grad = not freeze

    def set_trainable(self, stage: str) -> None:
        """Configure gradient settings per training stage."""
        if stage == "warmup":
            # Keep LLM (and LoRA) frozen; train only local encoders/projection/head.
            self.freeze_llm(True)
            self.freeze_local_encoders(False)
        elif stage == "lora":
            # Base LLM remains frozen; LoRA adapters are trainable by default.
            # Do not call freeze_llm(False) here to avoid unfreezing base weights.
            self.freeze_llm(True)
            self.freeze_local_encoders(False)
        elif stage == "full":
            # Unfreeze everything (optional, rarely used with LLMs).
            self.freeze_llm(False)
            self.freeze_local_encoders(False)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

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
    # This will fail if model weights are missing; it's only for architecture checks.
    model = DlCsiPredictor(cfg)
    print(model.count_parameters())
