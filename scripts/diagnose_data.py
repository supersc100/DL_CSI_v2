"""Diagnose FDD CSI H5 datasets for leakage, overlap, and UL/DL correlation."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np


def _load_sample(path: str, idx: int):
    with h5py.File(path, "r") as f:
        h_ul = np.array(f["h_ul"][idx])
        h_dl = np.array(f["h_dl"][idx])
        ls = np.array(f["large_scale"][idx]) if "large_scale" in f else None
    return h_ul, h_dl, ls


def _energy(x: np.ndarray) -> float:
    return float((np.abs(x) ** 2).mean())


def _nmse_db(pred: np.ndarray, target: np.ndarray) -> float:
    num = (np.abs(pred - target) ** 2).sum()
    den = (np.abs(target) ** 2).sum() + 1e-12
    return 10.0 * np.log10(num / den + 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Diagnose H5 CSI datasets.")
    parser.add_argument("--train", default="./data/processed/train_cdld.h5")
    parser.add_argument("--val", default="./data/processed/val_cdld.h5")
    parser.add_argument("--test", default="./data/processed/test_cdld.h5")
    parser.add_argument("--num-check", type=int, default=5, help="Number of head/tail samples to compare")
    args = parser.parse_args()

    splits = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }

    # Basic shapes and sizes.
    print("=" * 60)
    print("1. Dataset shapes")
    print("=" * 60)
    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"{name}: NOT FOUND -> {path}")
            continue
        with h5py.File(path, "r") as f:
            n = f["h_ul"].shape[0]
            shape = f["h_ul"].shape[1:]
            ls_shape = f["large_scale"].shape[1:] if "large_scale" in f else None
            print(f"{name}: {path}")
            print(f"   samples={n}, h_shape={shape}, large_scale_shape={ls_shape}")

    # Check train/val/test overlap by comparing first/last samples.
    print("\n" + "=" * 60)
    print("2. Cross-split overlap check (first/last samples)")
    print("=" * 60)
    samples = {}
    for name, path in splits.items():
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            n = f["h_ul"].shape[0]
            idxs = list(range(args.num_check)) + list(range(max(0, n - args.num_check), n))
            samples[name] = [(i, np.array(f["h_ul"][i]), np.array(f["h_dl"][i])) for i in idxs]

    for a in samples:
        for b in samples:
            if a == b:
                continue
            matches = 0
            total = 0
            for (ia, h_ul_a, h_dl_a) in samples[a]:
                for (ib, h_ul_b, h_dl_b) in samples[b]:
                    total += 1
                    if np.allclose(h_ul_a, h_ul_b) and np.allclose(h_dl_a, h_dl_b):
                        matches += 1
            print(f"{a} vs {b}: {matches}/{total} head/tail samples are identical")

    # Check UL/DL independence within one split.
    print("\n" + "=" * 60)
    print("3. UL/DL independence (on train first 100 samples)")
    print("=" * 60)
    train_path = args.train
    if os.path.exists(train_path):
        with h5py.File(train_path, "r") as f:
            n = f["h_ul"].shape[0]
            k = min(100, n)
            nmse_ul_as_dl = []
            mag_corr = []
            for i in range(k):
                h_ul = np.array(f["h_ul"][i])
                h_dl = np.array(f["h_dl"][i])
                nmse_ul_as_dl.append(_nmse_db(h_ul, h_dl))
                mag_corr.append(np.corrcoef(np.abs(h_ul).flatten(), np.abs(h_dl).flatten())[0, 1])
            print(f"  NMSE(h_ul as pred, h_dl) mean = {np.mean(nmse_ul_as_dl):.3f} dB")
            print(f"  |H_ul| vs |H_dl| correlation  = {np.mean(mag_corr):.4f}")
            print("  Note: if NMSE is < -30 dB or correlation > 0.999, UL/DL may share small-scale fading.")

    # Check large_scale uniqueness.
    print("\n" + "=" * 60)
    print("4. large_scale uniqueness (first 1000 train samples)")
    print("=" * 60)
    if os.path.exists(train_path):
        with h5py.File(train_path, "r") as f:
            n = f["large_scale"].shape[0]
            k = min(1000, n)
            ls = np.array(f["large_scale"][:k])
            unique = np.unique(ls, axis=0).shape[0]
            print(f"  unique large_scale vectors = {unique}/{k}")
            print("  If unique count is very low, large_scale may not provide enough per-sample info.")

    print("\n" + "=" * 60)
    print("Diagnosis done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
