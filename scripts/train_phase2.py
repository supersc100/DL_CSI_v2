"""Training script for Phase 2 phase recovery network."""
import argparse
import os
import sys

import torch

# Ensure project root is on path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform
from src.models.phase_recovery_network import PhaseRecoveryNetwork
from src.models.sampling_mask import SamplingMaskGenerator
from src.training.phase2_trainer import Phase2Trainer
from src.utils.logging import Logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 phase recovery network.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset size for smoke test.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs per stage (for smoke test).")
    return parser.parse_args()


def build_mask_generator(config):
    p2_cfg = config.phase2
    sampling_cfg = p2_cfg.sampling
    return SamplingMaskGenerator(
        num_subcarriers=int(config.data.num_subcarriers),
        use_adaptive=bool(getattr(sampling_cfg, "use_adaptive", True)),
        base_spacing=int(getattr(sampling_cfg, "base_spacing", 8)),
        use_nonuniform_peaks=bool(getattr(sampling_cfg, "use_nonuniform_peaks", True)),
        peak_prob=float(getattr(sampling_cfg, "peak_prob", 0.7)),
        peak_neighbors=int(getattr(sampling_cfg, "peak_neighbors", 2)),
        random_offset=bool(getattr(sampling_cfg, "random_offset", True)),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available() and config.project.device == "cuda":
        print("WARNING: CUDA not available; falling back to CPU for smoke test.")
        config.project.device = "cpu"
        config.project.mixed_precision = False
    set_seed(int(config.project.seed))

    # Smoke-test overrides.
    if args.smoke_test:
        if args.num_samples is None:
            args.num_samples = 128
        if args.epochs is not None:
            # Apply small epochs to each stage.
            for stage in ("warmup", "main", "qat"):
                getattr(config.phase2.training, stage).epochs = args.epochs

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )

    mask_generator = build_mask_generator(config)

    # DataLoaders.
    common_kwargs = {
        "batch_size": int(config.training.batch_size),
        "num_workers": int(config.training.num_workers),
        "pin_memory": bool(config.training.pin_memory),
        "transform": transform,
        "load_history": True,
        "use_history": bool(config.model.use_history),
        "use_large_scale": bool(config.model.use_large_scale),
        "phase2_enabled": True,
        "mask_generator": mask_generator,
    }

    train_loader = build_dataloader(
        h5_path=str(config.data.h5_train),
        shuffle=True,
        training=True,
        **common_kwargs,
    )
    val_loader = build_dataloader(
        h5_path=str(config.data.h5_val),
        shuffle=False,
        training=False,
        **common_kwargs,
    )

    if args.smoke_test and args.num_samples is not None:
        # Limit dataset length for smoke test.
        train_loader.dataset.num_samples = min(args.num_samples, train_loader.dataset.num_samples)
        val_loader.dataset.num_samples = min(args.num_samples // 4, val_loader.dataset.num_samples)

    # Model.
    stage1_checkpoint = str(config.phase2.stage1_checkpoint) if getattr(config.phase2, "stage1_checkpoint", None) else None
    model = PhaseRecoveryNetwork(config, stage1_checkpoint_path=stage1_checkpoint)

    print(f"Phase2 model parameters: {model.count_parameters()}")

    # Trainer.
    logger = Logger(
        log_dir=str(config.project.log_dir),
        experiment_name="phase2",
        use_tensorboard=False,
    )
    trainer = Phase2Trainer(model, config, logger=logger)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    results = trainer.fit(train_loader, val_loader)
    print("Phase2 training completed.")
    print(results)


if __name__ == "__main__":
    main()
