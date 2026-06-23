"""Multi-stage trainer for the Phase 2 phase recovery network."""
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

from src.models.quantization import ScalarQuantizer
from src.training.phase2_losses import Phase2Loss
from src.utils.logging import Logger
from src.utils.metrics import compute_phase2_metrics


class Phase2Trainer:
    """Trainer supporting Warmup, Main, and QAT stages for Phase2."""

    STAGE_WARMUP = "warmup"
    STAGE_MAIN = "main"
    STAGE_QAT = "qat"

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

        self.model.to(self.device)

        self.use_amp = bool(config.project.mixed_precision) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Stage configs.
        p2_cfg = getattr(config, "phase2", {})
        train_cfg = getattr(p2_cfg, "training", {})
        self.stage_configs = {
            self.STAGE_WARMUP: getattr(train_cfg, "warmup", {}),
            self.STAGE_MAIN: getattr(train_cfg, "main", {}),
            self.STAGE_QAT: getattr(train_cfg, "qat", {}),
        }

        self.quantizer = ScalarQuantizer(
            num_bits=int(getattr(getattr(p2_cfg, "quantization", {}), "num_bits", 16))
        )

        self.global_step = 0
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        self.current_stage = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def _configure_freezing(self, stage: str) -> None:
        """Freeze/unfreeze parameters for the current stage."""
        # Always freeze Stage1.
        for param in self.model.stage1.parameters():
            param.requires_grad = False

        if stage == self.STAGE_WARMUP:
            # Freeze the heavy Transformer interpolator during warmup.
            for name, param in self.model.named_parameters():
                if "stage1" in name:
                    param.requires_grad = False
                elif "freq_interpolator" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            # Main / QAT: train all Phase2 modules.
            for name, param in self.model.named_parameters():
                param.requires_grad = ("stage1" not in name)

    def _build_optimizer(self, cfg: Any) -> AdamW:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(
            trainable,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    def _build_scheduler(self, optimizer: AdamW, cfg: Any, total_steps: int) -> Optional[Any]:
        """Cosine annealing with optional linear warmup."""
        scheduler_name = str(getattr(cfg, "scheduler", "cosine")).lower()
        warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.1))
        warmup_steps = int(total_steps * warmup_ratio)

        if scheduler_name != "cosine":
            return None

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))

        return LambdaLR(optimizer, lr_lambda)

    def _build_criterion(self, cfg: Any) -> Phase2Loss:
        loss_cfg = getattr(cfg, "loss", {})
        return Phase2Loss(
            mse_weight=float(getattr(loss_cfg, "mse_weight", 1.0)),
            phase_weight=float(getattr(loss_cfg, "phase_weight", 0.5)),
            angle_l1_weight=float(getattr(loss_cfg, "angle_l1_weight", 0.1)),
        )

    def set_stage(self, stage: str) -> None:
        """Configure model, optimizer, scheduler, and loss for a stage."""
        self.current_stage = stage
        cfg = self.stage_configs[stage]

        self._configure_freezing(stage)
        self.optimizer = self._build_optimizer(cfg)
        self.criterion = self._build_criterion(cfg)

        self.logger.log(
            f"[Phase2] Entering stage '{stage}'; trainable parameters: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

    def _apply_qat(self, sparse_dl_ad: torch.Tensor, qat_ratio: float) -> torch.Tensor:
        """Randomly quantize a fraction of the sparse input batch."""
        if qat_ratio <= 0.0:
            return sparse_dl_ad
        if torch.rand(1).item() < qat_ratio:
            return self.quantizer(sparse_dl_ad)
        return sparse_dl_ad

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

        cfg = self.stage_configs[self.current_stage]
        grad_clip = float(getattr(cfg, "gradient_clip", 1.0))
        grad_accum = int(getattr(cfg, "gradient_accumulation", 1))
        qat_ratio = float(getattr(cfg, "qat_ratio", 0.0)) if self.current_stage == self.STAGE_QAT else 0.0

        total_loss = 0.0
        metrics_sums = {}
        num_batches = 0

        context = torch.enable_grad if is_training else torch.no_grad
        pbar = tqdm(dataloader, desc=f"{self.current_stage} {'train' if is_training else 'val'} epoch {epoch}")

        with context():
            for batch_idx, batch in enumerate(pbar):
                h_ul_ad = batch["h_ul_ad"].to(self.device)
                sparse_dl_ad = batch["sparse_dl_ad"].to(self.device)
                mask = batch["sampling_mask"].to(self.device)
                target_dl_ad = batch["h_dl_ad"].to(self.device)

                large_scale = batch.get("large_scale")
                if large_scale is not None:
                    large_scale = large_scale.to(self.device)

                history_ul_ad = batch.get("history_ul_ad")
                history_dl_ad = batch.get("history_dl_ad")
                if history_ul_ad is not None:
                    history_ul_ad = history_ul_ad.to(self.device)
                    history_dl_ad = history_dl_ad.to(self.device)

                if is_training:
                    sparse_dl_ad = self._apply_qat(sparse_dl_ad, qat_ratio)

                with autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                    pred_dl_ad, aux = self.model(
                        h_ul_ad,
                        sparse_dl_ad,
                        mask,
                        large_scale=large_scale,
                        history_ul_ad=history_ul_ad,
                        history_dl_ad=history_dl_ad,
                    )
                    loss, _ = self.criterion(pred_dl_ad, target_dl_ad, aux["mag_stage1"])

                if is_training:
                    loss = loss / grad_accum
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (batch_idx + 1) % grad_accum == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                grad_clip,
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                grad_clip,
                            )
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.global_step += 1

                metrics = compute_phase2_metrics(
                    pred_dl_ad.detach(), target_dl_ad.detach(), aux["mag_stage1"].detach()
                )
                total_loss += float(loss.item()) * grad_accum
                for k, v in metrics.items():
                    metrics_sums[k] = metrics_sums.get(k, 0.0) + float(v)
                num_batches += 1

                pbar.set_postfix(
                    loss=total_loss / num_batches,
                    nmse=metrics_sums.get("nmse_db", 0.0) / num_batches,
                )

        avg_metrics = {"loss": total_loss / max(num_batches, 1)}
        for k, v in metrics_sums.items():
            avg_metrics[k] = v / max(num_batches, 1)
        return avg_metrics

    def _train_single_stage(
        self,
        stage: str,
        train_loader,
        val_loader,
    ) -> Dict[str, float]:
        cfg = self.stage_configs[stage]
        epochs = int(getattr(cfg, "epochs", 10))
        total_steps = epochs * len(train_loader) // int(getattr(cfg, "gradient_accumulation", 1))
        self.scheduler = self._build_scheduler(self.optimizer, cfg, total_steps)

        monitor = str(getattr(getattr(cfg, "early_stopping", {}), "monitor", "nmse_db"))
        patience = int(getattr(getattr(cfg, "early_stopping", {}), "patience", 10))
        save_interval = int(getattr(getattr(cfg, "logging", {}), "save_interval", 5))
        eval_interval = int(getattr(getattr(cfg, "logging", {}), "eval_interval", 1))

        for epoch in range(epochs):
            train_metrics = self._run_epoch(train_loader, is_training=True, epoch=epoch)
            self.logger.log_metrics(train_metrics, step=self.global_step, prefix=f"{stage}/train")
            self.logger.log(
                f"[{stage}] Epoch {epoch}/{epochs} train: "
                f"loss={train_metrics['loss']:.4f}, "
                f"NMSE={train_metrics.get('nmse_db', 0.0):.3f} dB, "
                f"phase_NMSE={train_metrics.get('phase_nmse_db', 0.0):.3f} dB"
            )

            if val_loader is not None and (
                epoch % eval_interval == 0 or epoch == epochs - 1
            ):
                val_metrics = self._run_epoch(val_loader, is_training=False, epoch=epoch)
                self.logger.log_metrics(val_metrics, step=self.global_step, prefix=f"{stage}/val")
                self.logger.log(
                    f"[{stage}] Epoch {epoch}/{epochs} val:   "
                    f"loss={val_metrics['loss']:.4f}, "
                    f"NMSE={val_metrics.get('nmse_db', 0.0):.3f} dB, "
                    f"phase_NMSE={val_metrics.get('phase_nmse_db', 0.0):.3f} dB"
                )

                val_monitored = val_metrics.get(monitor)
                if val_monitored is not None:
                    if val_monitored < self.best_val_metric:
                        self.best_val_metric = val_monitored
                        self.patience_counter = 0
                        self.save_checkpoint(
                            os.path.join(
                                self.config.project.checkpoint_dir,
                                f"phase2_{stage}_best.pt",
                            )
                        )
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= patience:
                            self.logger.log(f"[{stage}] Early stopping at epoch {epoch}")
                            break

            if save_interval > 0 and epoch % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(
                        self.config.project.checkpoint_dir,
                        f"phase2_{stage}_epoch{epoch}.pt",
                    )
                )

        return {"best_val_metric": self.best_val_metric}

    def fit(
        self,
        train_loader,
        val_loader,
    ) -> Dict[str, float]:
        """Run warmup, main, and QAT stages sequentially."""
        results = {}

        self.set_stage(self.STAGE_WARMUP)
        results["warmup"] = self._train_single_stage(
            self.STAGE_WARMUP, train_loader, val_loader
        )

        self.set_stage(self.STAGE_MAIN)
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        results["main"] = self._train_single_stage(
            self.STAGE_MAIN, train_loader, val_loader
        )

        self.set_stage(self.STAGE_QAT)
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        results["qat"] = self._train_single_stage(
            self.STAGE_QAT, train_loader, val_loader
        )

        self.logger.close()
        return results

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "stage": self.current_stage,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
        }
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, path)
        self.logger.log(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_metric = state.get("best_val_metric", float("inf"))
        self.current_stage = state.get("stage")
        if self.scaler is not None and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.logger.log(f"Loaded checkpoint from {path}")
