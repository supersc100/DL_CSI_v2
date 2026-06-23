"""Single-stage end-to-end training of the CNN+Transformer CSI predictor."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform
from src.models.dl_csi_predictor import DlCsiPredictor
from src.training.trainer import Trainer
from src.utils.logging import Logger


def main():
    parser = argparse.ArgumentParser(description="Train CNN+Transformer CSI predictor.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Quick run on a small subset.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset size for smoke test.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    use_history = bool(getattr(config.model, "use_history", True))
    use_large_scale = bool(getattr(config.model, "use_large_scale", True))

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )

    train_loader = build_dataloader(
        h5_path=str(config.data.h5_train),
        batch_size=int(config.training.batch_size),
        shuffle=True,
        num_workers=int(config.training.num_workers),
        transform=transform,
        pin_memory=bool(config.training.pin_memory),
        use_history=use_history,
        use_large_scale=use_large_scale,
    )
    val_loader = build_dataloader(
        h5_path=str(config.data.h5_val),
        batch_size=int(config.training.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        transform=transform,
        pin_memory=bool(config.training.pin_memory),
        use_history=use_history,
        use_large_scale=use_large_scale,
    )

    if args.smoke_test:
        if args.num_samples is not None:
            train_loader.dataset.num_samples = min(args.num_samples, train_loader.dataset.num_samples)
            val_loader.dataset.num_samples = min(max(args.num_samples // 4, 4), val_loader.dataset.num_samples)
        # Default smoke-test subset if not specified.
        elif train_loader.dataset.num_samples > 256:
            train_loader.dataset.num_samples = 256
            val_loader.dataset.num_samples = 64

    model = DlCsiPredictor(config)
    logger = Logger(
        log_dir=str(config.project.log_dir),
        experiment_name="train",
        use_tensorboard=False,
    )
    trainer = Trainer(model, config, logger=logger)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    epochs = args.epochs if args.epochs is not None else None
    trainer.fit(train_loader, val_loader, epochs=epochs)


if __name__ == "__main__":
    main()
