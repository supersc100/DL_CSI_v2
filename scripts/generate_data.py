"""Generate train/val/test datasets with Sionna."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.sionna_generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate FDD CSI datasets.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to generate",
    )
    parser.add_argument(
        "--tdd-oracle-split",
        default=None,
        choices=["train", "val", "test"],
        help="Generate one split with identical UL/DL fast fading (TDD upper bound).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for split in args.split:
        if split == args.tdd_oracle_split:
            oracle = True
            tag = "_tdd_oracle"
        else:
            oracle = False
            tag = ""

        num_samples = int(getattr(config.data.samples, split))
        output_path = getattr(config.data, f"h5_{split}")
        if tag:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}{tag}{ext}"

        print(f"Generating {num_samples} samples for split={split}{tag} ...")
        generate_dataset(
            config,
            num_samples=num_samples,
            output_path=output_path,
            seed_offset={"train": 0, "val": 1000000, "test": 2000000}[split],
            synthesize_ul=not oracle,
        )
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
