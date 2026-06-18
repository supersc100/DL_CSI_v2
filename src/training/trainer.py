"""Generic trainer for the FDD CSI predictor."""
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.transforms import real_channels_to_complex
from src.training.losses import CsiLoss, NmseLoss
from src.utils.logging import Logger
from src.utils.metrics import compute_metrics


class Trainer:
    """Trainer that supports warmup, LoRA, and optional full-finetune stages."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        stage: str = "warmup",
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.config = config
        self.stage = stage
        self.device = torch.device(str(config.project.device))
        self.logger = logger or Logger(str(config.project.log_dir))

        # Move local modules to target device. The LLM was already placed via
        # device_map; calling .to() on the whole model can be intercepted by
        # accelerate hooks, so we move the non-LLM submodules individually.
        for module in (
            self.model.csi_encoder,
            self.model.temporal_encoder,
            self.model.env_encoder,
            self.model.fusion,
            self.model.embedding_projection,
            self.model.regression_head,
        ):
            if module is not None:
                module.to(self.device)

        # Stage-specific training config.
        self.stage_cfg = getattr(config.training, stage)
        self.model.set_trainable(stage)

        # Optimizer.
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(self.stage_cfg.lr),
            weight_decay=float(self.stage_cfg.weight_decay),
        )

        # Losses.
        self.criterion = CsiLoss(
            mse_weight=float(self.stage_cfg.loss.mse_weight),
            angle_delay_l1_weight=float(self.stage_cfg.loss.angle_delay_l1_weight),
        )
        self.nmse_loss = NmseLoss()

        # AMP / scaler.
        self.use_amp = bool(config.project.mixed_precision) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpoint / logging.
        self.global_step = 0
        self.best_val_nmse = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0

    def _build_scheduler(self, total_steps: int) -> Optional[Any]:
        scheduler_name = str(self.stage_cfg.scheduler).lower()
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
        num_batches = 0

        context = torch.enable_grad if is_training else torch.no_grad
        pbar = tqdm(dataloader, desc=f"{self.stage} {'train' if is_training else 'val'} epoch {epoch}")

        with context():
            for batch in pbar:
                current_ul_ad = batch["h_ul_ad"].to(self.device)
                history_ul_ad = batch["history_ul_ad"].to(self.device)
                history_dl_ad = batch["history_dl_ad"].to(self.device)
                large_scale = batch["large_scale"].to(self.device)
                target_dl_ad = batch["h_dl_ad"].to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                    pred_dl_ad = self.model(
                        current_ul_ad, history_ul_ad, history_dl_ad, large_scale
                    )
                    loss = self.criterion(pred_dl_ad, target_dl_ad)

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

                metrics = compute_metrics(pred_dl_ad.detach(), target_dl_ad.detach())
                total_loss += float(loss.item())
                total_nmse += metrics["nmse_db"]
                total_cos += metrics["cosine_similarity"]
                num_batches += 1

                pbar.set_postfix(
                    loss=total_loss / num_batches,
                    nmse=total_nmse / num_batches,
                    cos=total_cos / num_batches,
                )

        avg_metrics = {
            "loss": total_loss / max(num_batches, 1),
            "nmse_db": total_nmse / max(num_batches, 1),
            "cosine_similarity": total_cos / max(num_batches, 1),
        }
        return avg_metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        if epochs is None:
            epochs = int(self.stage_cfg.epochs)

        total_steps = epochs * len(train_loader)
        scheduler = self._build_scheduler(total_steps)

        self.logger.log(f"Starting {self.stage} training for {epochs} epochs.")
        self.logger.log(f"Trainable parameters: {self.model.count_parameters()['trainable']}")

        for epoch in range(self.start_epoch, epochs):
            train_metrics = self._run_epoch(train_loader, is_training=True, epoch=epoch)
            self.logger.log_metrics(train_metrics, step=self.global_step, prefix=f"{self.stage}/train")
            self.logger.log(
                f"Epoch {epoch}/{epochs} train: loss={train_metrics['loss']:.4f}, "
                f"NMSE={train_metrics['nmse_db']:.3f} dB, cos={train_metrics['cosine_similarity']:.4f}"
            )

            if val_loader is not None and (
                epoch % int(self.config.training.logging.eval_interval) == 0
                or epoch == epochs - 1
            ):
                val_metrics = self._run_epoch(val_loader, is_training=False, epoch=epoch)
                self.logger.log_metrics(val_metrics, step=self.global_step, prefix=f"{self.stage}/val")
                self.logger.log(
                    f"Epoch {epoch}/{epochs} val:   loss={val_metrics['loss']:.4f}, "
                    f"NMSE={val_metrics['nmse_db']:.3f} dB, cos={val_metrics['cosine_similarity']:.4f}"
                )

                # Early stopping.
                val_nmse = val_metrics["nmse_db"]
                if val_nmse < self.best_val_nmse:
                    self.best_val_nmse = val_nmse
                    self.patience_counter = 0
                    self.save_checkpoint(
                        os.path.join(self.config.project.checkpoint_dir, f"best_{self.stage}.pt")
                    )
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= int(self.config.training.early_stopping.patience):
                        self.logger.log(f"Early stopping at epoch {epoch}")
                        break

            if epoch % int(self.config.training.logging.save_interval) == 0:
                self.save_checkpoint(
                    os.path.join(self.config.project.checkpoint_dir, f"{self.stage}_epoch{epoch}.pt")
                )

            if scheduler is not None:
                scheduler.step()

        self.logger.close()
        return {"best_val_nmse": self.best_val_nmse}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "stage": self.stage,
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
        if self.scaler is not None and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.logger.log(f"Loaded checkpoint from {path}")
