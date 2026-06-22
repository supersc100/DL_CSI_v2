"""Run baseline methods on the test set."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform
from src.utils.baselines import BASELINES
from src.utils.logging import Logger
from src.utils.metrics import compute_all_metrics
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Run baselines.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        choices=list(BASELINES.keys()),
        help="Baseline names to run",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    use_history = bool(getattr(config.model, "use_history", True))
    use_large_scale = bool(getattr(config.model, "use_large_scale", True))

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )

    h5_path = getattr(config.data, f"h5_{args.split}")
    loader = build_dataloader(
        h5_path=str(h5_path),
        batch_size=int(config.evaluation.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        transform=transform,
        pin_memory=bool(config.training.pin_memory),
        use_history=use_history,
        use_large_scale=use_large_scale,
    )

    logger = Logger(log_dir=str(config.project.log_dir), experiment_name="baselines")
    baselines = args.baselines or list(BASELINES.keys())

    for name in baselines:
        fn = BASELINES[name]
        all_preds = []
        all_targets = []
        for batch in tqdm(loader, desc=f"Baseline {name}"):
            current_ul_ad = batch["h_ul_ad"]
            large_scale = batch.get("large_scale")
            target_dl_ad = batch["h_dl_ad"]

            if name in ("copy_ul", "tdd_oracle"):
                result = fn(current_ul_ad, target_dl_ad)
            elif name == "angle_delay_interp":
                if use_history:
                    history_ul_ad = batch["history_ul_ad"]
                    history_dl_ad = batch["history_dl_ad"]
                    result = fn(current_ul_ad, target_dl_ad, history_ul_ad, history_dl_ad)
                else:
                    # No history available: baseline gracefully falls back.
                    result = fn(current_ul_ad, target_dl_ad)
            elif name == "no_large_scale":
                if use_history:
                    history_ul_ad = batch["history_ul_ad"]
                    history_dl_ad = batch["history_dl_ad"]
                    result = fn(current_ul_ad, target_dl_ad, history_ul_ad, history_dl_ad)
                else:
                    result = fn(current_ul_ad, target_dl_ad)
            elif name == "no_history":
                result = fn(current_ul_ad, large_scale, target_dl_ad)
            else:
                raise ValueError(f"Unknown baseline {name}")

            all_preds.append(result["pred_ad"])
            all_targets.append(result["target_ad"])

        pred = torch.cat(all_preds, dim=0)
        target = torch.cat(all_targets, dim=0)
        metrics = compute_all_metrics(pred, target)
        logger.log(f"Baseline {name}: {metrics}")
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
