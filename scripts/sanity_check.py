"""Shape sanity check without loading real data or LLM weights.

This script builds the model with dummy inputs to verify tensor shapes. It skips
LLM loading by substituting a tiny random Transformer if no weights are found.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.models.dl_csi_predictor import DlCsiPredictor


class DummyLLM(torch.nn.Module):
    """Minimal transformer stand-in for shape checking."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-llm", action="store_true", help="Replace LLM with a tiny dummy transformer")
    args = parser.parse_args()

    config = load_config(args.config)
    B = 2
    T = int(config.data.history_window)
    N_tx = int(config.data.bs_array.num_elements)
    N_rx = int(config.data.ue_array.num_elements)
    M = int(config.data.num_subcarriers)
    num_ls = int(config.model.env_encoder.input_dim)

    if args.skip_llm or not os.path.isdir(str(config.model.llm_path)):
        print("Substituting LLM with dummy transformer for shape check.")
        model = DlCsiPredictor(config, llm=DummyLLM(int(config.model.llm_hidden_dim)))
    else:
        model = DlCsiPredictor(config)

    current_ul_ad = torch.randn(B, N_tx, N_rx, M, dtype=torch.complex64)
    history_ul_ad = torch.randn(B, T, N_tx, N_rx, M, dtype=torch.complex64)
    history_dl_ad = torch.randn(B, T, N_tx, N_rx, M, dtype=torch.complex64)
    large_scale = torch.randn(B, num_ls)

    with torch.no_grad():
        pred = model(current_ul_ad, history_ul_ad, history_dl_ad, large_scale)

    print(f"Input current UL AD shape:  {current_ul_ad.shape}")
    print(f"Input history UL AD shape:  {history_ul_ad.shape}")
    print(f"Input history DL AD shape:  {history_dl_ad.shape}")
    print(f"Input large-scale shape:    {large_scale.shape}")
    print(f"Output pred DL AD shape:    {pred.shape}")
    assert pred.shape == (B, N_tx, N_rx, M), f"Expected {(B, N_tx, N_rx, M)}, got {pred.shape}"
    print("Shape check passed.")


if __name__ == "__main__":
    main()
