"""Check whether Stage1 magnitude prediction collapses to a dataset template.

Computes the NMSE of a constant "mean magnitude spectrum" predictor on the
validation/test set. If this template already achieves very low NMSE, then any
network can ignore the input and still get good magnitude metrics.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from src.data.transforms import AngleDelayTransform


def _nmse_db(pred: np.ndarray, target: np.ndarray) -> float:
    num = ((np.abs(pred - target)) ** 2).sum()
    den = ((np.abs(target)) ** 2).sum() + 1e-12
    return 10.0 * np.log10(num / den + 1e-12)


def _nmse_db_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return per-sample NMSE in dB."""
    diff2 = (np.abs(pred - target) ** 2).reshape(pred.shape[0], -1).sum(axis=1)
    den = (np.abs(target) ** 2).reshape(target.shape[0], -1).sum(axis=1) + 1e-12
    return 10.0 * np.log10(diff2 / den + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./data/processed/train_perturb.h5")
    parser.add_argument("--test", default="./data/processed/val_perturb.h5")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    args = parser.parse_args()

    transform = AngleDelayTransform(normalize=True)

    print(f"Loading up to {args.num_train} train samples from {args.train}")
    train_mags = []
    with h5py.File(args.train, "r") as f:
        n = f["h_dl"].shape[0]
        k = min(args.num_train, n)
        for i in range(k):
            h_dl = torch.from_numpy(np.array(f["h_dl"][i]))
            h_dl_ad, _ = transform(h_dl)
            train_mags.append(np.abs(h_dl_ad.numpy()))
    train_mags = np.stack(train_mags, axis=0)  # [K, N_tx, N_rx, M]
    mean_mag = train_mags.mean(axis=0, keepdims=True)  # [1, ...]
    train_var = ((train_mags - mean_mag) ** 2).mean()
    print(f"Train magnitude variance around mean template: {train_var:.6e}")

    print(f"\nEvaluating template on up to {args.num_test} test samples from {args.test}")
    template_nmses = []
    copy_ul_nmses = []
    with h5py.File(args.test, "r") as f:
        n = f["h_dl"].shape[0]
        k = min(args.num_test, n)
        for i in range(k):
            h_ul = torch.from_numpy(np.array(f["h_ul"][i]))
            h_dl = torch.from_numpy(np.array(f["h_dl"][i]))
            h_ul_ad, _ = transform(h_ul)
            h_dl_ad, _ = transform(h_dl)
            template_nmses.append(_nmse_db(mean_mag, np.abs(h_dl_ad.numpy())))
            copy_ul_nmses.append(_nmse_db(np.abs(h_ul_ad.numpy()), np.abs(h_dl_ad.numpy())))

    print(f"Template predictor mag_NMSE: {np.mean(template_nmses):.3f} dB")
    print(f"copy_ul mag_NMSE:            {np.mean(copy_ul_nmses):.3f} dB")
    print("\nInterpretation:")
    print("  - If template NMSE is < -30 dB, the dataset magnitude spectra are")
    print("    almost identical across samples. Any network can collapse to the")
    print("    mean template and get good NMSE without using UL.")
    print("  - If copy_ul NMSE is much worse than template, UL/DL magnitude spectra")
    print("    differ significantly and a good model must exploit UL.")


if __name__ == "__main__":
    main()
