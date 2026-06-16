"""Stage 1: Warmup training of local encoders + regression head."""
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
    parser = argparse.ArgumentParser(description="Warmup training (encoders + head).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)

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
    )
    val_loader = build_dataloader(
        h5_path=str(config.data.h5_val),
        batch_size=int(config.training.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        transform=transform,
        pin_memory=bool(config.training.pin_memory),
    )

    model = DlCsiPredictor(config)
    logger = Logger(
        log_dir=str(config.project.log_dir),
        experiment_name="warmup",
    )
    trainer = Trainer(model, config, stage="warmup", logger=logger)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
