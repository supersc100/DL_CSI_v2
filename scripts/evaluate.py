"""Evaluate a trained model on the test set."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform
from src.models.dl_csi_predictor import DlCsiPredictor
from src.utils.logging import Logger
from src.utils.metrics import compute_all_metrics
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(str(config.project.device))

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )

    h5_path = getattr(config.data, f"h5_{args.split}")
    use_history = bool(getattr(config.model, "use_history", True))
    use_large_scale = bool(getattr(config.model, "use_large_scale", True))
    test_loader = build_dataloader(
        h5_path=str(h5_path),
        batch_size=int(config.evaluation.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        transform=transform,
        pin_memory=bool(config.training.pin_memory),
        use_history=use_history,
        use_large_scale=use_large_scale,
    )

    model = DlCsiPredictor(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    logger = Logger(log_dir=str(config.project.log_dir), experiment_name="evaluate")
    logger.log(f"Evaluating checkpoint {args.checkpoint} on {args.split}")

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            current_ul_ad = batch["h_ul_ad"].to(device)
            target_dl_ad = batch["h_dl_ad"].to(device)

            large_scale = None
            if use_large_scale:
                large_scale = batch["large_scale"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model_kwargs = {}
                if use_history:
                    model_kwargs["history_ul_ad"] = batch["history_ul_ad"].to(device)
                    model_kwargs["history_dl_ad"] = batch["history_dl_ad"].to(device)
                if use_large_scale:
                    model_kwargs["large_scale"] = large_scale
                pred_dl_ad = model(current_ul_ad, **model_kwargs)

            all_preds.append(pred_dl_ad.detach().cpu())
            all_targets.append(target_dl_ad.detach().cpu())

    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(pred, target)

    logger.log(f"Test metrics: {metrics}")
    print(metrics)


if __name__ == "__main__":
    main()
