"""Phase 2 network: recover downlink CSI phase from sparse subband feedback."""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.dl_csi_predictor import DlCsiPredictor
from src.models.freq_sparse_encoder import FreqSparseEncoder
from src.models.magnitude_encoder import MagnitudeEncoder
from src.models.magnitude_guided_attention import MagnitudeGuidedAttention
from src.models.frequency_interpolator import FrequencyInterpolator
from src.models.phase_head import PhaseHead


class PhaseRecoveryNetwork(nn.Module):
    """Two-stage network for FDD downlink CSI prediction.

    Stage 1 (frozen): DlCsiPredictor predicts downlink magnitude.
    Stage 2 (trainable): sparse subband encoder + magnitude-guided attention +
        frequency-domain Transformer interpolator + phase head recover phase.

    Final output: H_pred = |H_stage1| * exp(j * theta_phase2)

    Inputs:
        h_ul_ad:        [B, N_tx, N_rx, M] complex
        sparse_dl_ad:   [B, N_tx, N_rx, M] complex (zero at unsampled positions)
        mask:           [B, M] bool
        large_scale:    [B, D_lsp] optional
        history_ul_ad:  [B, T, N_tx, N_rx, M] optional
        history_dl_ad:  [B, T, N_tx, N_rx, M] optional

    Outputs:
        h_dl_pred:      [B, N_tx, N_rx, M] complex
        aux: dict with mag_stage1, phase, stage1_pred
    """

    def __init__(self, config, stage1_checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config
        self.stage1_checkpoint_path = stage1_checkpoint_path

        # Load and freeze Stage1.
        self.stage1 = self._build_frozen_stage1(config, stage1_checkpoint_path)

        # Derive dimensions.
        self.num_tx = int(config.data.bs_array.num_elements)
        self.num_rx = int(config.data.ue_array.num_elements)
        self.num_subcarriers = int(config.data.num_subcarriers)

        # Phase2 model config with sensible defaults.
        p2_cfg = getattr(config, "phase2", {})
        model_cfg = getattr(p2_cfg, "model", {})
        self.feat_dim = int(getattr(model_cfg, "feat_dim", 128))

        sparse_cfg = getattr(model_cfg, "sparse_encoder", {})
        mag_cfg = getattr(model_cfg, "mag_encoder", {})
        attn_cfg = getattr(model_cfg, "mag_attention", {})
        interp_cfg = getattr(model_cfg, "freq_interpolator", {})
        head_cfg = getattr(model_cfg, "phase_head", {})

        self.sparse_encoder = FreqSparseEncoder(
            feat_dim=self.feat_dim,
            num_layers=int(getattr(sparse_cfg, "num_layers", 3)),
            kernel_size=int(getattr(sparse_cfg, "kernel_size", 5)),
            dropout=float(getattr(sparse_cfg, "dropout", 0.1)),
        )

        self.mag_encoder = MagnitudeEncoder(
            feat_dim=self.feat_dim,
            num_layers=int(getattr(mag_cfg, "num_layers", 2)),
            kernel_size=int(getattr(mag_cfg, "kernel_size", 3)),
            dropout=float(getattr(mag_cfg, "dropout", 0.1)),
        )

        self.mag_attention = MagnitudeGuidedAttention(
            feat_dim=self.feat_dim,
            num_heads=int(getattr(attn_cfg, "num_heads", 4)),
            dropout=float(getattr(attn_cfg, "dropout", 0.1)),
        )

        self.freq_interpolator = FrequencyInterpolator(
            num_subcarriers=self.num_subcarriers,
            feat_dim=self.feat_dim,
            num_layers=int(getattr(interp_cfg, "num_layers", 4)),
            num_heads=int(getattr(interp_cfg, "num_heads", 4)),
            mlp_ratio=float(getattr(interp_cfg, "mlp_ratio", 4.0)),
            dropout=float(getattr(interp_cfg, "dropout", 0.05)),
        )

        self.phase_head = PhaseHead(
            feat_dim=self.feat_dim,
            hidden_dim=int(getattr(head_cfg, "hidden_dim", 256)),
            num_tx=self.num_tx,
            num_rx=self.num_rx,
        )

    def _build_frozen_stage1(
        self,
        config,
        checkpoint_path: Optional[str],
    ) -> DlCsiPredictor:
        """Build Stage1 model and freeze all parameters."""
        stage1 = DlCsiPredictor(config)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            stage1.load_state_dict(state["model_state_dict"], strict=False)
            # Log a short summary.
            missing = set(stage1.state_dict().keys()) - set(state["model_state_dict"].keys())
            if missing:
                print(
                    f"[Phase2] Loaded Stage1 checkpoint; {len(missing)} keys not in checkpoint"
                )
        elif checkpoint_path is not None:
            print(
                f"[Phase2] WARNING: Stage1 checkpoint not found at {checkpoint_path}. "
                "Using randomly initialized Stage1 (debug only)."
            )

        # Freeze Stage1.
        for param in stage1.parameters():
            param.requires_grad = False
        stage1.eval()
        return stage1

    def forward(
        self,
        h_ul_ad: torch.Tensor,
        sparse_dl_ad: torch.Tensor,
        mask: torch.Tensor,
        large_scale: Optional[torch.Tensor] = None,
        history_ul_ad: Optional[torch.Tensor] = None,
        history_dl_ad: Optional[torch.Tensor] = None,
    ) -> tuple:
        # Stage1 predicts magnitude (frozen).
        with torch.no_grad():
            stage1_kwargs = {}
            if self.stage1.use_large_scale:
                stage1_kwargs["large_scale"] = large_scale
            if self.stage1.use_history:
                stage1_kwargs["history_ul_ad"] = history_ul_ad
                stage1_kwargs["history_dl_ad"] = history_dl_ad
            stage1_pred = self.stage1(h_ul_ad, **stage1_kwargs)

        mag_stage1 = stage1_pred.abs()

        # Phase2 modules.
        sparse_feat = self.sparse_encoder(sparse_dl_ad, mask)
        mag_feat = self.mag_encoder(mag_stage1)
        attended = self.mag_attention(sparse_feat, mag_feat, mag_feat)
        interp_feat = self.freq_interpolator(attended, mask)
        phase_complex = self.phase_head(interp_feat)

        # Fuse magnitude and phase.
        h_dl_pred = mag_stage1 * phase_complex

        aux = {
            "mag_stage1": mag_stage1,
            "phase": phase_complex,
            "stage1_pred": stage1_pred,
        }
        return h_dl_pred, aux

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stage1_total = sum(p.numel() for p in self.stage1.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "stage1_total": stage1_total,
        }
