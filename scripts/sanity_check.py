"""Shape sanity check without loading real data.

This script builds the CNN+Transformer model with dummy inputs to verify tensor
shapes. No LLM weights are required.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.models.dl_csi_predictor import DlCsiPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    B = 2
    T = int(config.data.history_window)
    N_tx = int(config.data.bs_array.num_elements)
    N_rx = int(config.data.ue_array.num_elements)
    M = int(config.data.num_subcarriers)
    num_ls = int(config.model.env_encoder.input_dim)
    use_history = bool(getattr(config.model, "use_history", True))

    model = DlCsiPredictor(config)
    print(f"Model parameters: {model.count_parameters()}")
    print(f"use_history: {use_history}")

    current_ul_ad = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    large_scale = torch.randn(B, num_ls)

    inputs = [current_ul_ad, large_scale]
    if use_history:
        history_ul_ad = torch.randn(B, T, N_tx, N_rx, M, dtype=torch.complex64)
        history_dl_ad = torch.randn(B, T, N_tx, N_rx, M, dtype=torch.complex64)
        inputs.extend([history_ul_ad, history_dl_ad])

    with torch.no_grad():
        pred = model(*inputs)

    print(f"Input current UL AD shape:  {current_ul_ad.shape}")
    if use_history:
        print(f"Input history UL AD shape:  {history_ul_ad.shape}")
        print(f"Input history DL AD shape:  {history_dl_ad.shape}")
    print(f"Input large-scale shape:    {large_scale.shape}")
    print(f"Output pred DL AD shape:    {pred.shape}")
    assert pred.shape == (B, N_tx, N_rx, M), f"Expected {(B, N_tx, N_rx, M)}, got {pred.shape}"
    print("Shape check passed.")


if __name__ == "__main__":
    main()
