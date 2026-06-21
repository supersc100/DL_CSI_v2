"""Fast end-to-end training smoke test for the CNN+Transformer CSI predictor.

This script creates in-memory dummy data, shrinks the model dims, and runs a few
epochs of training. It is meant to verify that the full training loop works
before spending time on Sionna data generation and full-scale training.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.models.dl_csi_predictor import DlCsiPredictor
from src.training.trainer import Trainer
from src.utils.logging import Logger


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
    config.model.hidden_dim = 128

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

    # Transformer fusion.
    config.model.transformer_fusion.num_layers = 2
    config.model.transformer_fusion.num_heads = 4
    config.model.transformer_fusion.mlp_ratio = 4.0
    config.model.transformer_fusion.dropout = 0.1

    # Regression head.
    config.model.regression_head.hidden_dim = 256

    # Training.
    config.training.batch_size = 4
    config.training.num_workers = 0
    config.training.pin_memory = False
    config.project.mixed_precision = False
    config.project.device = "cpu"

    config.training.epochs = 2
    config.training.lr = 1e-3
    config.training.weight_decay = 1e-4
    config.training.loss.mse_weight = 1.0
    config.training.loss.angle_delay_l1_weight = 0.1

    # Disable early stopping / long eval intervals for smoke test.
    config.training.early_stopping.patience = 10
    config.training.logging.eval_interval = 1
    config.training.logging.save_interval = 1
    config.training.gradient_clip = 1.0

    return config


def main():
    parser = argparse.ArgumentParser(description="Fast training smoke test without real data.")
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

    model = DlCsiPredictor(config)

    logger = Logger(log_dir=str(config.project.log_dir), experiment_name="smoke_test")
    trainer = Trainer(model, config, logger=logger)

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
