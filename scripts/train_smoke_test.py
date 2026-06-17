"""Fast end-to-end training smoke test without the real 1.5B LLM.

This script creates in-memory dummy data, swaps the LLM for a tiny random
transformer, and runs a few epochs of warmup training. It is meant to verify
that the full training loop (data -> model -> loss -> backward -> optimizer)
works before spending time on LLM weight downloads and Sionna data generation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.models.dl_csi_predictor import DlCsiPredictor
from src.training.trainer import Trainer
from src.utils.logging import Logger


class DummyLLM(torch.nn.Module):
    """Minimal transformer stand-in for fast smoke tests."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, inputs_embeds, output_hidden_states=False, return_dict=False):
        out = self.encoder(inputs_embeds)

        class Output:
            pass

        o = Output()
        o.hidden_states = [out]
        o.last_hidden_state = out
        return o


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
        # Angle-delay domain complex tensors.
        h_ul_ad = torch.randn(num_samples, num_tx, num_rx, num_subcarriers, dtype=torch.complex64)
        h_dl_ad = torch.randn(num_samples, num_tx, num_rx, num_subcarriers, dtype=torch.complex64)
        history_ul_ad = torch.randn(
            num_samples, history_window, num_tx, num_rx, num_subcarriers, dtype=torch.complex64
        )
        history_dl_ad = torch.randn(
            num_samples, history_window, num_tx, num_rx, num_subcarriers, dtype=torch.complex64
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


def override_config_for_smoke(config):
    """Shrink model and training config so the smoke test runs in seconds."""
    # Model dims.
    config.model.feature_dim = 128
    config.model.llm_hidden_dim = 128
    config.model.num_virtual_tokens = 3

    # CSI encoder.
    config.model.csi_encoder.base_channels = 16
    config.model.csi_encoder.num_layers = 2
    config.model.csi_encoder.feature_dim = 128

    # Temporal encoder.
    config.model.temporal_encoder.csi.base_channels = 16
    config.model.temporal_encoder.csi.csi_num_layers = 2
    config.model.temporal_encoder.csi.csi_feature_dim = 128
    config.model.temporal_encoder.transformer.hidden_dim = 128
    config.model.temporal_encoder.transformer.num_heads = 4

    # Environment encoder.
    config.model.env_encoder.hidden_dims = [64, 128]
    config.model.env_encoder.output_dim = 128

    # Regression head.
    config.model.regression_head.hidden_dim = 256

    # Training.
    config.training.batch_size = 4
    config.training.num_workers = 0
    config.training.pin_memory = False
    config.project.mixed_precision = False
    config.project.device = "cpu"

    # Warmup stage.
    config.training.warmup.epochs = 2
    config.training.warmup.lr = 1e-3
    config.training.warmup.weight_decay = 1e-4
    config.training.warmup.loss.mse_weight = 1.0
    config.training.warmup.loss.angle_delay_l1_weight = 0.1

    # Disable early stopping / long eval intervals for smoke test.
    config.training.early_stopping.patience = 10
    config.training.logging.eval_interval = 1
    config.training.logging.save_interval = 1
    config.training.gradient_clip = 1.0

    return config


def main():
    parser = argparse.ArgumentParser(description="Fast training smoke test without real LLM/data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=16, help="Dummy train/val samples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    config = override_config_for_smoke(config)

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

    model = DlCsiPredictor(config, llm=DummyLLM(int(config.model.llm_hidden_dim)))

    logger = Logger(log_dir=str(config.project.log_dir), experiment_name="smoke_test")
    trainer = Trainer(model, config, stage="warmup", logger=logger)

    print("Smoke test parameters:", model.count_parameters())
    print(f"Training on {args.samples} dummy samples for {args.epochs} epochs...")

    result = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # Sanity assertions: the loop must finish and produce finite metrics.
    best_val_nmse = result.get("best_val_nmse", float("nan"))
    print(f"Best val NMSE (dB): {best_val_nmse:.3f}")
    assert best_val_nmse < float("inf"), "Validation NMSE diverged to inf"
    assert not torch.isnan(torch.tensor(best_val_nmse)), "Validation NMSE is NaN"
    print("Training smoke test passed.")


if __name__ == "__main__":
    main()
