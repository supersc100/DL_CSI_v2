"""Shape sanity check for Phase 2 network."""
import argparse
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import load_config
from src.models.phase_recovery_network import PhaseRecoveryNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Use a random Stage1 for shape checking (no checkpoint needed).
    model = PhaseRecoveryNetwork(config, stage1_checkpoint_path=None)
    model.eval()

    B, T, N_tx, N_rx, M = 2, 16, 32, 4, 32
    num_ls = 6

    h_ul_ad = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    sparse_dl_ad = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[:, ::8] = True
    large_scale = torch.randn(B, num_ls)
    history_ul_ad = torch.randn(B, T, N_tx, N_rx, M, dtype=torch.complex64)
    history_dl_ad = torch.randn(B, T, N_tx, N_rx, M, dtype=torch.complex64)

    with torch.no_grad():
        pred, aux = model(
            h_ul_ad,
            sparse_dl_ad,
            mask,
            large_scale=large_scale,
            history_ul_ad=history_ul_ad,
            history_dl_ad=history_dl_ad,
        )

    assert pred.shape == (B, N_tx, N_rx, M), f"pred shape mismatch: {pred.shape}"
    assert pred.dtype == torch.complex64
    assert aux["mag_stage1"].shape == (B, N_tx, N_rx, M)
    assert aux["phase"].shape == (B, N_tx, N_rx, M)

    # Phase should have unit magnitude.
    phase_mag_mean = aux["phase"].abs().mean().item()
    assert abs(phase_mag_mean - 1.0) < 1e-3, f"phase unit magnitude violated: {phase_mag_mean}"

    # Stage1 should be frozen.
    trainable_stage1 = sum(p.numel() for p in model.stage1.parameters() if p.requires_grad)
    assert trainable_stage1 == 0, f"Stage1 has {trainable_stage1} trainable parameters"

    counts = model.count_parameters()
    print("Phase2 sanity check PASSED.")
    print(f"  total parameters: {counts['total']:,}")
    print(f"  trainable parameters: {counts['trainable']:,}")
    print(f"  frozen parameters: {counts['frozen']:,}")
    print(f"  stage1 parameters: {counts['stage1_total']:,}")


if __name__ == "__main__":
    main()
