"""Configuration loading utilities."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DotDict:
    """Lightweight recursive dict-to-attribute wrapper."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                setattr(self, key, [DotDict(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, DotDict) else item for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"DotDict({self.to_dict()})"


def load_config(path: str = "config.yaml") -> DotDict:
    """Load YAML config and resolve relative paths against project root."""
    if not os.path.isabs(path):
        # Project root is two levels up from src/config.py by default.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = DotDict(raw)

    # Resolve device
    if config.project.device == "auto":
        import torch
        config.project.device = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def update_config_from_cli(config: DotDict, args: Any) -> DotDict:
    """Override config with non-None CLI arguments."""
    for key, value in vars(args).items():
        if value is None:
            continue
        parts = key.split(".")
        node = config
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)
    return config
