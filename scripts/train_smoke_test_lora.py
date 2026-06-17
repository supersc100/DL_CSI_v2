"""Fast end-to-end LoRA smoke test without the real 1.5B LLM.

This script creates in-memory dummy data, builds a tiny dummy transformer whose
Linear layers are named like the DeepSeek/Qwen target modules (q_proj, k_proj,
v_proj, o_proj, gate_proj, up_proj, down_proj), wraps it with PEFT LoRA, and
runs a few epochs of LoRA fine-tuning. It is meant to verify that the LoRA
path (data -> encoders -> PEFT LLM -> LoRA backward -> optimizer) works before
downloading real LLM weights or generating Sionna data.
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from peft import LoraConfig, get_peft_model

from src.config import load_config
from src.models.dl_csi_predictor import DlCsiPredictor
from src.training.trainer import Trainer
from src.utils.logging import Logger


class DummyLoraTransformerLayer(nn.Module):
    """Single transformer layer with LoRA-targetable projection names."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention projections with the same names as DeepSeek/Qwen.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # FFN projections with the same names as Llama/Qwen MLP.
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        residual = x
        x = self.norm1(x)

        # Multi-head self-attention.
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        out = self.o_proj(out)
        x = residual + out

        # FFN with SwiGLU-like gating.
        residual = x
        x = self.norm2(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = residual + x
        return x


class DummyLoraLLM(nn.Module):
    """Minimal transformer whose Linear names match the LoRA target_modules."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList(
            [DummyLoraTransformerLayer(hidden_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        # PEFT's PeftModelForCausalLM forward inspects base_model.config.model_type.
        self.config = SimpleNamespace(model_type="dummy")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # PEFT's PeftModelForCausalLM stores this reference at construction time.
        # It is never actually called during our forward-only smoke test.
        return {"input_ids": input_ids, **kwargs}

    def get_input_embeddings(self):
        # Satisfy PEFT introspection without needing a real token embedding table.
        return nn.Identity()

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        # PEFT may pass input_ids (as None) along with inputs_embeds; we only use embeddings.
        hidden = inputs_embeds
        hidden_states = []
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)

        # Return a simple object compatible with DlCsiPredictor's access pattern.
        class Output:
            pass

        output = Output()
        output.last_hidden_state = hidden
        output.hidden_states = hidden_states
        return output


class DummyCsiDataset(TensorDataset):
    """In-memory dummy angle-delay CSI tensors."""

    def __init__(
        self,
        num_samples: int,
        num_tx: int,
        num_rx: int,
        num_subcarriers: int,
        history_window: int,
        num_large_scale: int,
    ):
        self.num_samples = num_samples
        h_ul_ad = torch.randn(
            num_samples, num_tx, num_rx, num_subcarriers, dtype=torch.complex64
        )
        h_dl_ad = torch.randn(
            num_samples, num_tx, num_rx, num_subcarriers, dtype=torch.complex64
        )
        history_ul_ad = torch.randn(
            num_samples,
            history_window,
            num_tx,
            num_rx,
            num_subcarriers,
            dtype=torch.complex64,
        )
        history_dl_ad = torch.randn(
            num_samples,
            history_window,
            num_tx,
            num_rx,
            num_subcarriers,
            dtype=torch.complex64,
        )
        large_scale = torch.randn(num_samples, num_large_scale)

        super().__init__(h_ul_ad, history_ul_ad, history_dl_ad, large_scale, h_dl_ad)

    def __getitem__(self, idx):
        h_ul_ad, history_ul_ad, history_dl_ad, large_scale, h_dl_ad = super().__getitem__(idx)
        return {
            "h_ul_ad": h_ul_ad,
            "history_ul_ad": history_ul_ad,
            "history_dl_ad": history_dl_ad,
            "large_scale": large_scale,
            "h_dl_ad": h_dl_ad,
        }


def build_dummy_dataloader(
    num_samples: int,
    batch_size: int,
    num_tx: int,
    num_rx: int,
    num_subcarriers: int,
    history_window: int,
    num_large_scale: int,
    shuffle: bool = True,
):
    dataset = DummyCsiDataset(
        num_samples, num_tx, num_rx, num_subcarriers, history_window, num_large_scale
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def override_config_for_lora_smoke(config):
    """Shrink model and training config so the LoRA smoke test runs in seconds."""
    # Reuse the same small architecture as the warmup smoke test.
    config.model.feature_dim = 128
    config.model.llm_hidden_dim = 128
    config.model.num_virtual_tokens = 3

    config.model.csi_encoder.base_channels = 16
    config.model.csi_encoder.num_layers = 2
    config.model.csi_encoder.feature_dim = 128

    config.model.temporal_encoder.csi.base_channels = 16
    config.model.temporal_encoder.csi.csi_num_layers = 2
    config.model.temporal_encoder.csi.csi_feature_dim = 128
    config.model.temporal_encoder.transformer.hidden_dim = 128
    config.model.temporal_encoder.transformer.num_heads = 4

    config.model.env_encoder.hidden_dims = [64, 128]
    config.model.env_encoder.output_dim = 128

    config.model.regression_head.hidden_dim = 256

    # Training / environment.
    config.training.batch_size = 4
    config.training.num_workers = 0
    config.training.pin_memory = False
    config.project.mixed_precision = False
    config.project.device = "cpu"

    config.training.early_stopping.patience = 10
    config.training.logging.eval_interval = 1
    config.training.logging.save_interval = 1
    config.training.gradient_clip = 1.0

    # Small LoRA config for speed.
    config.lora.r = 4
    config.lora.lora_alpha = 8

    # The real config uses the key ``lora_finetune``, but the trainer/model
    # expect the stage name ``lora``. Normalize it for this smoke test without
    # forcing a project-wide rename.
    if not hasattr(config.training, "lora") and hasattr(config.training, "lora_finetune"):
        config.training.lora = config.training.lora_finetune

    # Override LoRA-stage hyperparameters for a fast smoke run.
    config.training.lora.epochs = 2
    config.training.lora.lr = 1e-3
    config.training.lora.weight_decay = 1e-4
    config.training.lora.loss.mse_weight = 1.0
    config.training.lora.loss.angle_delay_l1_weight = 0.1

    return config


def build_peft_dummy_llm(config):
    """Build a tiny dummy LLM and wrap it with PEFT LoRA."""
    hidden_dim = int(config.model.llm_hidden_dim)
    dummy_llm = DummyLoraLLM(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
    )

    # Freeze base parameters before adding LoRA adapters, just like the real
    # DeepSeek backbone does in DlCsiPredictor._build_llm.
    for param in dummy_llm.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        r=int(config.lora.r),
        lora_alpha=int(config.lora.lora_alpha),
        target_modules=list(config.lora.target_modules),
        lora_dropout=float(config.lora.lora_dropout),
        bias=str(config.lora.bias),
        task_type=str(config.lora.task_type),
        modules_to_save=list(config.lora.modules_to_save),
    )
    return get_peft_model(dummy_llm, lora_cfg)


def main():
    parser = argparse.ArgumentParser(description="Fast LoRA smoke test without real LLM/data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=16, help="Dummy train/val samples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    config = override_config_for_lora_smoke(config)

    N_tx = int(config.data.bs_array.num_elements)
    N_rx = int(config.data.ue_array.num_elements)
    M = int(config.data.num_subcarriers)
    T = int(config.data.history_window)
    num_ls = int(config.model.env_encoder.input_dim)

    train_loader = build_dummy_dataloader(
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_tx=N_tx,
        num_rx=N_rx,
        num_subcarriers=M,
        history_window=T,
        num_large_scale=num_ls,
        shuffle=True,
    )
    val_loader = build_dummy_dataloader(
        num_samples=args.samples // 2,
        batch_size=args.batch_size,
        num_tx=N_tx,
        num_rx=N_rx,
        num_subcarriers=M,
        history_window=T,
        num_large_scale=num_ls,
        shuffle=False,
    )

    peft_dummy_llm = build_peft_dummy_llm(config)
    model = DlCsiPredictor(config, llm=peft_dummy_llm)

    logger = Logger(log_dir=str(config.project.log_dir), experiment_name="smoke_test_lora")
    trainer = Trainer(model, config, stage="lora", logger=logger)

    # Sanity check 1: there must be LoRA parameters and they must be trainable.
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lora_named = [(n, p) for n, p in trainable_named if "lora_" in n]
    assert lora_named, "No LoRA parameters found; LoRA adapters did not attach to the dummy LLM."
    print(f"LoRA smoke test parameters: {model.count_parameters()}")
    print(f"Training on {args.samples} dummy samples for {args.epochs} epochs...")

    result = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # Sanity check 2: metrics must be finite.
    best_val_nmse = result.get("best_val_nmse", float("nan"))
    print(f"Best val NMSE (dB): {best_val_nmse:.3f}")
    assert best_val_nmse < float("inf"), "Validation NMSE diverged to inf"
    assert not torch.isnan(torch.tensor(best_val_nmse)), "Validation NMSE is NaN"
    print("LoRA smoke test passed.")


if __name__ == "__main__":
    main()
