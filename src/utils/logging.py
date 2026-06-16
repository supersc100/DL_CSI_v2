"""Logging utilities."""
import os
from datetime import datetime
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Simple file + tensorboard logger."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        os.makedirs(log_dir, exist_ok=True)
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, "train.log")
        self.use_tensorboard = use_tensorboard
        self.writer = SummaryWriter(self.log_dir) if use_tensorboard else None

    def log(self, message: str, level: str = "INFO") -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if self.writer is not None and isinstance(value, (int, float)):
                self.writer.add_scalar(full_key, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
