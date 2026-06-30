"""Single-stage trainer for the CNN+Transformer FDD CSI predictor."""
import os
import random
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.losses import CsiLoss, NmseLoss
from src.utils.channel_noise import add_awgn
from src.utils.logging import Logger
from src.utils.metrics import compute_all_metrics


class Trainer:
    """End-to-end trainer for the non-LLM CSI predictor."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(str(config.project.device))
        self.logger = logger or Logger(str(config.project.log_dir))

        # Move the whole model to the target device.
        self.model.to(self.device)

        # Optimizer over all trainable parameters.
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )

        # Losses.
        self.criterion = CsiLoss(
            mse_weight=float(config.training.loss.mse_weight),
            magnitude_weight=float(config.training.loss.get("magnitude_weight", 1.0)),
            angle_delay_l1_weight=float(config.training.loss.angle_delay_l1_weight),
            diversity_weight=float(config.training.loss.get("diversity_weight", 0.0)),
            use_ratio=bool(config.training.loss.get("use_ratio", False)),
            ratio_eps=float(config.training.loss.get("ratio_eps", 1e-6)),
        )
        self.nmse_loss = NmseLoss()
        self.use_ratio = bool(config.training.loss.get("use_ratio", False))

        # Training-time UL corruption augmentation.
        self.ul_noise_prob = float(getattr(config.training, "ul_noise_prob", 0.0))
        self.ul_noise_snr_list = list(getattr(config.training, "ul_noise_snr_list", []))
        self.ul_mask_prob = float(getattr(config.training, "ul_mask_prob", 0.0))
        self.ul_mask_ratio = float(getattr(config.training, "ul_mask_ratio", 0.0))

        # AMP / scaler.
        self.use_amp = bool(config.project.mixed_precision) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Checkpoint / logging.
        self.global_step = 0
        self.best_val_metric = float("inf")
        self.best_val_nmse = float("inf")  # kept for checkpoint compatibility
        self.patience_counter = 0
        self.start_epoch = 0
        self.metric_to_monitor = str(self.config.training.early_stopping.get("monitor", "val_nmse"))
        self.use_history = bool(getattr(self.config.model, "use_history", True))
        self.use_large_scale = bool(getattr(self.config.model, "use_large_scale", True))

    def _build_scheduler(self, total_steps: int) -> Optional[Any]:
        scheduler_name = str(self.config.training.scheduler).lower()
        if scheduler_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=total_steps)
        return None

    def _run_epoch(
        self,
        dataloader,
        is_training: bool = True,
        epoch: int = 0,
    ) -> Dict[str, float]:
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_nmse = 0.0
        total_cos = 0.0
        total_mag_nmse = 0.0
        total_mag_cos = 0.0
        num_batches = 0

        context = torch.enable_grad if is_training else torch.no_grad
        pbar = tqdm(dataloader, desc=f"{'train' if is_training else 'val'} epoch {epoch}")

        with context():
            for batch in pbar:
                current_ul_ad = batch["h_ul_ad"].to(self.device)
                target_dl_ad = batch["h_dl_ad"].to(self.device)

                # Training-time UL corruption augmentation: randomly corrupt the
                # uplink input so the network cannot ignore it and memorize a
                # fixed downlink spectrum.
                if is_training:
                    if self.ul_noise_prob > 0.0 and len(self.ul_noise_snr_list) > 0:
                        if random.random() < self.ul_noise_prob:
                            snr_db = float(random.choice(self.ul_noise_snr_list))
                            current_ul_ad = add_awgn(current_ul_ad, snr_db)

                    if self.ul_mask_prob > 0.0 and self.ul_mask_ratio > 0.0:
                        if random.random() < self.ul_mask_prob:
                            # Randomly zero out a fraction of spatial-frequency elements.
                            mask = torch.rand_like(current_ul_ad.real) > self.ul_mask_ratio
                            current_ul_ad = current_ul_ad * mask

                if self.use_history:
                    history_ul_ad = batch["history_ul_ad"].to(self.device)
                    history_dl_ad = batch["history_dl_ad"].to(self.device)

                large_scale = None
                if self.use_large_scale:
                    large_scale = batch["large_scale"].to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                with autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                    model_kwargs = {}
                    if self.use_history:
                        model_kwargs["history_ul_ad"] = history_ul_ad
                        model_kwargs["history_dl_ad"] = history_dl_ad
                    if self.use_large_scale:
                        model_kwargs["large_scale"] = large_scale
                    pred_dl_ad = self.model(current_ul_ad, **model_kwargs)
                    loss_kwargs = {}
                    if self.use_ratio:
                        loss_kwargs["current_ul"] = current_ul_ad
                    loss = self.criterion(pred_dl_ad, target_dl_ad, **loss_kwargs)

                if is_training:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            float(self.config.training.gradient_clip),
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            float(self.config.training.gradient_clip),
                        )
                        self.optimizer.step()

                    self.global_step += 1

                metrics = compute_all_metrics(pred_dl_ad.detach(), target_dl_ad.detach())
                total_loss += float(loss.item())
                total_nmse += metrics["nmse_db"]
                total_cos += metrics["cosine_similarity"]
                total_mag_nmse += metrics["magnitude_nmse_db"]
                total_mag_cos += metrics["magnitude_cosine_similarity"]
                num_batches += 1

                pbar.set_postfix(
                    loss=total_loss / num_batches,
                    mag_nmse=total_mag_nmse / num_batches,
                    mag_cos=total_mag_cos / num_batches,
                )

        avg_metrics = {
            "loss": total_loss / max(num_batches, 1),
            "nmse_db": total_nmse / max(num_batches, 1),
            "cosine_similarity": total_cos / max(num_batches, 1),
            "magnitude_nmse_db": total_mag_nmse / max(num_batches, 1),
            "magnitude_cosine_similarity": total_mag_cos / max(num_batches, 1),
        }
        return avg_metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        if epochs is None:
            epochs = int(self.config.training.epochs)

        total_steps = epochs * len(train_loader)
        scheduler = self._build_scheduler(total_steps)

        self.logger.log(f"Starting training for {epochs} epochs.")
        self.logger.log(f"Trainable parameters: {self.model.count_parameters()['trainable']}")

        for epoch in range(self.start_epoch, epochs):
            train_metrics = self._run_epoch(train_loader, is_training=True, epoch=epoch)
            self.logger.log_metrics(train_metrics, step=self.global_step, prefix="train")
            self.logger.log(
                f"Epoch {epoch}/{epochs} train: loss={train_metrics['loss']:.4f}, "
                f"NMSE={train_metrics['nmse_db']:.3f} dB, cos={train_metrics['cosine_similarity']:.4f}, "
                f"mag_NMSE={train_metrics['magnitude_nmse_db']:.3f} dB, mag_cos={train_metrics['magnitude_cosine_similarity']:.4f}"
            )

            if val_loader is not None and (
                epoch % int(self.config.training.logging.eval_interval) == 0
                or epoch == epochs - 1
            ):
                val_metrics = self._run_epoch(val_loader, is_training=False, epoch=epoch)
                self.logger.log_metrics(val_metrics, step=self.global_step, prefix="val")
                self.logger.log(
                    f"Epoch {epoch}/{epochs} val:   loss={val_metrics['loss']:.4f}, "
                    f"NMSE={val_metrics['nmse_db']:.3f} dB, cos={val_metrics['cosine_similarity']:.4f}, "
                    f"mag_NMSE={val_metrics['magnitude_nmse_db']:.3f} dB, mag_cos={val_metrics['magnitude_cosine_similarity']:.4f}"
                )

                # Early stopping on the configured metric (default: val_nmse).
                monitored_name = self.metric_to_monitor.replace("val_", "")
                val_monitored = val_metrics.get(monitored_name)
                if val_monitored is None:
                    raise ValueError(
                        f"Early stopping metric '{self.metric_to_monitor}' not found in val metrics. "
                        f"Available: {list(val_metrics.keys())}"
                    )
                if val_monitored < self.best_val_metric:
                    self.best_val_metric = val_monitored
                    self.best_val_nmse = val_metrics["nmse_db"]
                    self.patience_counter = 0
                    self.save_checkpoint(
                        os.path.join(self.config.project.checkpoint_dir, "best.pt")
                    )
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= int(self.config.training.early_stopping.patience):
                        self.logger.log(f"Early stopping at epoch {epoch}")
                        break

            if epoch % int(self.config.training.logging.save_interval) == 0:
                self.save_checkpoint(
                    os.path.join(self.config.project.checkpoint_dir, f"epoch{epoch}.pt")
                )

            if scheduler is not None:
                scheduler.step()

        self.logger.close()
        return {"best_val_metric": self.best_val_metric, "best_val_nmse": self.best_val_nmse}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "epoch": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_nmse": self.best_val_nmse,
        }
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, path)
        self.logger.log(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("epoch", 0)
        self.best_val_nmse = state.get("best_val_nmse", float("inf"))
        self.best_val_metric = state.get("best_val_metric", self.best_val_nmse)
        if self.scaler is not None and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.logger.log(f"Loaded checkpoint from {path}")
