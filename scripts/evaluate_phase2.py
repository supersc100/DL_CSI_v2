"""Evaluation script for Phase 2 phase recovery network."""
import argparse
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform
from src.models.phase_recovery_network import PhaseRecoveryNetwork
from src.models.quantization import ScalarQuantizer
from src.models.sampling_mask import SamplingMaskGenerator
from src.utils.baselines import BASELINES
from src.utils.metrics import compute_phase2_metrics
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 network.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--quantized", action="store_true", help="Use quantized sparse inputs.")
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def build_mask_generator(config):
    p2_cfg = config.phase2
    sampling_cfg = p2_cfg.sampling
    return SamplingMaskGenerator(
        num_subcarriers=int(config.data.num_subcarriers),
        use_adaptive=bool(getattr(sampling_cfg, "use_adaptive", True)),
        base_spacing=int(getattr(sampling_cfg, "base_spacing", 8)),
        use_nonuniform_peaks=bool(getattr(sampling_cfg, "use_nonuniform_peaks", True)),
        peak_prob=0.0,  # deterministic for eval
        peak_neighbors=int(getattr(sampling_cfg, "peak_neighbors", 2)),
        random_offset=False,
    )


@torch.no_grad()
def evaluate_model(model, dataloader, device, quantizer=None):
    model.eval()
    all_metrics = {}
    num_batches = 0

    for batch in dataloader:
        h_ul_ad = batch["h_ul_ad"].to(device)
        sparse_dl_ad = batch["sparse_dl_ad"].to(device)
        mask = batch["sampling_mask"].to(device)
        target_dl_ad = batch["h_dl_ad"].to(device)

        if quantizer is not None:
            sparse_dl_ad = quantizer(sparse_dl_ad)

        large_scale = batch.get("large_scale")
        if large_scale is not None:
            large_scale = large_scale.to(device)

        history_ul_ad = batch.get("history_ul_ad")
        history_dl_ad = batch.get("history_dl_ad")
        if history_ul_ad is not None:
            history_ul_ad = history_ul_ad.to(device)
            history_dl_ad = history_dl_ad.to(device)

        pred_dl_ad, aux = model(
            h_ul_ad,
            sparse_dl_ad,
            mask,
            large_scale=large_scale,
            history_ul_ad=history_ul_ad,
            history_dl_ad=history_dl_ad,
        )

        metrics = compute_phase2_metrics(pred_dl_ad, target_dl_ad, aux["mag_stage1"])
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in all_metrics.items()}


@torch.no_grad()
def evaluate_baselines(model, dataloader, device, baselines):
    results = {}
    for name in baselines:
        fn = BASELINES[name]
        all_metrics = {}
        num_batches = 0
        for batch in dataloader:
            target_dl_ad = batch["h_dl_ad"].to(device)
            kwargs = {"target_dl_ad": target_dl_ad}

            if name == "magnitude_only":
                # Use the frozen Stage1 magnitude prediction.
                h_ul_ad = batch["h_ul_ad"].to(device)
                large_scale = batch.get("large_scale")
                if large_scale is not None:
                    large_scale = large_scale.to(device)
                history_ul_ad = batch.get("history_ul_ad")
                history_dl_ad = batch.get("history_dl_ad")
                if history_ul_ad is not None:
                    history_ul_ad = history_ul_ad.to(device)
                    history_dl_ad = history_dl_ad.to(device)
                stage1_kwargs = {}
                if model.stage1.use_large_scale:
                    stage1_kwargs["large_scale"] = large_scale
                if model.stage1.use_history:
                    stage1_kwargs["history_ul_ad"] = history_ul_ad
                    stage1_kwargs["history_dl_ad"] = history_dl_ad
                with torch.no_grad():
                    stage1_pred = model.stage1(h_ul_ad, **stage1_kwargs)
                kwargs["stage1_pred"] = stage1_pred
            elif name in ("linear_interp", "dft_interp"):
                kwargs["sparse_dl"] = batch["sparse_dl"].to(device)
                kwargs["mask"] = batch["sampling_mask"].to(device)
            else:
                # Skip unsupported baselines for Phase2 eval.
                continue

            out = fn(**kwargs)
            pred = out["pred_ad"]
            metrics = compute_phase2_metrics(pred, target_dl_ad)
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            num_batches += 1

        if num_batches > 0:
            results[name] = {k: v / num_batches for k, v in all_metrics.items()}
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.project.seed))

    device = torch.device(str(config.project.device))

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )

    h5_path = {
        "train": config.data.h5_train,
        "val": config.data.h5_val,
        "test": config.data.h5_test,
    }[args.split]

    mask_generator = build_mask_generator(config)
    loader = build_dataloader(
        h5_path=str(h5_path),
        batch_size=int(config.phase2.evaluation.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        pin_memory=bool(config.training.pin_memory),
        transform=transform,
        load_history=True,
        use_history=bool(config.model.use_history),
        use_large_scale=bool(config.model.use_large_scale),
        phase2_enabled=True,
        mask_generator=mask_generator,
        training=False,
    )

    if args.num_samples is not None:
        loader.dataset.num_samples = min(args.num_samples, loader.dataset.num_samples)

    stage1_checkpoint = str(config.phase2.stage1_checkpoint) if getattr(config.phase2, "stage1_checkpoint", None) else None
    model = PhaseRecoveryNetwork(config, stage1_checkpoint_path=stage1_checkpoint)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device)

    quantizer = ScalarQuantizer(num_bits=int(config.phase2.quantization.num_bits)) if args.quantized else None

    print(f"Evaluating Phase2 model on {args.split} split ...")
    model_metrics = evaluate_model(model, loader, device, quantizer=quantizer)
    for k, v in model_metrics.items():
        print(f"  {k}: {v:.4f}")

    baselines = list(getattr(config.phase2.evaluation, "baselines", []))
    if baselines:
        print("\nBaseline comparison:")
        baseline_metrics = evaluate_baselines(model, loader, device, baselines)
        for name, metrics in baseline_metrics.items():
            print(f"  {name}:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
