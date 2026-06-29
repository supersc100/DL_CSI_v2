"""Stage 1: plot Magnitude NMSE vs SNR for copy_ul baseline and proposed model."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform
from src.models.dl_csi_predictor import DlCsiPredictor
from src.utils.baselines import baseline_copy_ul
from src.utils.logging import Logger
from src.utils.metrics import compute_all_metrics
from tqdm import tqdm


def run_copy_ul(loader):
    """Run the copy_ul baseline and return metrics."""
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc="copy_ul"):
        result = baseline_copy_ul(batch["h_ul_ad"], batch["h_dl_ad"])
        all_preds.append(result["pred_ad"])
        all_targets.append(result["target_ad"])
    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    return compute_all_metrics(pred, target)


def run_proposed(loader, model, device, use_history, use_large_scale):
    """Run the trained proposed model and return metrics."""
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="proposed"):
            current_ul_ad = batch["h_ul_ad"].to(device)
            target_dl_ad = batch["h_dl_ad"].to(device)

            model_kwargs = {}
            if use_history:
                model_kwargs["history_ul_ad"] = batch["history_ul_ad"].to(device)
                model_kwargs["history_dl_ad"] = batch["history_dl_ad"].to(device)
            if use_large_scale:
                model_kwargs["large_scale"] = batch["large_scale"].to(device)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                pred_dl_ad = model(current_ul_ad, **model_kwargs)

            all_preds.append(pred_dl_ad.detach().cpu())
            all_targets.append(target_dl_ad.detach().cpu())

    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    return compute_all_metrics(pred, target)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: copy_ul vs proposed model under SNR sweep."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True, help="Trained Stage 1 checkpoint.")
    parser.add_argument(
        "--snr-list",
        nargs="+",
        type=float,
        default=[-10, -5, 0, 5, 10, 15, 20],
        help="SNR points in dB.",
    )
    parser.add_argument("--output-csv", default="./outputs/stage1_snr_curve.csv")
    parser.add_argument("--output-png", default="./outputs/stage1_snr_curve.png")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(str(config.project.device))
    use_history = bool(getattr(config.model, "use_history", True))
    use_large_scale = bool(getattr(config.model, "use_large_scale", True))

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )

    # Load proposed model once.
    model = DlCsiPredictor(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    logger = Logger(
        log_dir=str(config.project.log_dir), experiment_name="stage1_snr_curve"
    )
    logger.log(f"Stage 1 SNR sweep: {args.snr_list}")

    records = []
    for snr_db in args.snr_list:
        print(f"\n=== SNR = {snr_db} dB ===")
        loader = build_dataloader(
            h5_path=str(getattr(config.data, f"h5_{args.split}")),
            batch_size=int(config.evaluation.batch_size),
            shuffle=False,
            num_workers=int(config.training.num_workers),
            transform=transform,
            pin_memory=bool(config.training.pin_memory),
            use_history=use_history,
            use_large_scale=use_large_scale,
            snr_db=snr_db,
            noise_base_seed=42,
        )

        # copy_ul baseline.
        m_copy = run_copy_ul(loader)
        print(
            f"  copy_ul:  mag_NMSE={m_copy['magnitude_nmse_db']:.3f} dB, "
            f"NMSE={m_copy['nmse_db']:.3f} dB"
        )
        records.append({
            "snr_db": snr_db,
            "method": "copy_ul",
            "magnitude_nmse_db": m_copy["magnitude_nmse_db"],
            "nmse_db": m_copy["nmse_db"],
            "cosine_similarity": m_copy["cosine_similarity"],
        })

        # Proposed model.
        m_prop = run_proposed(loader, model, device, use_history, use_large_scale)
        print(
            f"  proposed: mag_NMSE={m_prop['magnitude_nmse_db']:.3f} dB, "
            f"NMSE={m_prop['nmse_db']:.3f} dB"
        )
        records.append({
            "snr_db": snr_db,
            "method": "proposed",
            "magnitude_nmse_db": m_prop["magnitude_nmse_db"],
            "nmse_db": m_prop["nmse_db"],
            "cosine_similarity": m_prop["cosine_similarity"],
        })

    # Save CSV.
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved CSV to {args.output_csv}")

    # Plot.
    plt.figure(figsize=(8, 5))
    for method, marker in [("copy_ul", "s"), ("proposed", "o")]:
        sub = df[df["method"] == method]
        plt.plot(
            sub["snr_db"],
            sub["magnitude_nmse_db"],
            marker=marker,
            linewidth=2,
            label=method,
        )
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Magnitude NMSE (dB)", fontsize=12)
    plt.title("Stage 1: Magnitude NMSE vs. SNR", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300)
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
