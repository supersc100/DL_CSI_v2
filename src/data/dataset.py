"""PyTorch Dataset for FDD UL/DL CSI pairs stored in H5 files."""
import os
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import AngleDelayTransform, complex_to_real_channels


class FddCsiDataset(Dataset):
    """Dataset that loads UL/DL CSI pairs and converts them to angle-delay domain.

    Expected H5 keys:
        h_ul:          [N, num_tx_bs, num_rx_ue, M]
        h_dl:          [N, num_tx_bs, num_rx_ue, M]
        history_ul:    [N, T, num_tx_bs, num_rx_ue, M]
        history_dl:    [N, T, num_tx_bs, num_rx_ue, M]
        large_scale:   [N, num_large_scale]
    """

    def __init__(
        self,
        h5_path: str,
        transform: Optional[AngleDelayTransform] = None,
        load_history: bool = True,
        return_spatial: bool = False,
    ):
        self.h5_path = h5_path
        self.transform = transform or AngleDelayTransform()
        self.load_history = load_history
        self.return_spatial = return_spatial

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            self.num_samples = f["h_ul"].shape[0]
            self.num_tx = f["h_ul"].shape[1]
            self.num_rx = f["h_ul"].shape[2]
            self.num_subcarriers = f["h_ul"].shape[3]
            self.history_window = f["history_ul"].shape[1] if load_history else 0
            self.num_large_scale = f["large_scale"].shape[1]

    def __len__(self) -> int:
        return self.num_samples

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            h_ul = np.array(f["h_ul"][idx])
            h_dl = np.array(f["h_dl"][idx])
            large_scale = np.array(f["large_scale"][idx])
            if self.load_history:
                history_ul = np.array(f["history_ul"][idx])
                history_dl = np.array(f["history_dl"][idx])
            else:
                history_ul = np.zeros(
                    (self.history_window, self.num_tx, self.num_rx, self.num_subcarriers),
                    dtype=np.complex64,
                )
                history_dl = history_ul.copy()

        sample = {
            "h_ul": self._to_tensor(h_ul),
            "h_dl": self._to_tensor(h_dl),
            "history_ul": self._to_tensor(history_ul),
            "history_dl": self._to_tensor(history_dl),
            "large_scale": self._to_tensor(large_scale).float(),
        }

        # Convert current UL to angle-delay domain for model input.
        h_ul_ad, ul_stats = self.transform(sample["h_ul"])
        # Convert target DL to angle-delay domain for supervision.
        h_dl_ad, dl_stats = self.transform(sample["h_dl"])

        # Convert history pairs to angle-delay domain using the same transform.
        history_ul_ad = []
        history_dl_ad = []
        for t in range(self.history_window):
            h_t_ul_ad, _ = self.transform(sample["history_ul"][t])
            h_t_dl_ad, _ = self.transform(sample["history_dl"][t])
            history_ul_ad.append(h_t_ul_ad)
            history_dl_ad.append(h_t_dl_ad)

        sample["h_ul_ad"] = h_ul_ad
        sample["h_dl_ad"] = h_dl_ad
        sample["history_ul_ad"] = torch.stack(history_ul_ad, dim=0)
        sample["history_dl_ad"] = torch.stack(history_dl_ad, dim=0)

        # Keep normalization stats for inverse transform during inference.
        sample["ul_stats"] = ul_stats if ul_stats is not None else torch.zeros(2, *h_ul_ad.shape)
        sample["dl_stats"] = dl_stats if dl_stats is not None else torch.zeros(2, *h_dl_ad.shape)

        if not self.return_spatial:
            # Drop raw spatial fields to save memory.
            sample.pop("h_ul", None)
            sample.pop("h_dl", None)
            sample.pop("history_ul", None)
            sample.pop("history_dl", None)

        return sample


def build_dataloader(
    h5_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[AngleDelayTransform] = None,
    load_history: bool = True,
    **kwargs: Any,
):
    """Create a DataLoader from an H5 dataset."""
    dataset = FddCsiDataset(
        h5_path=h5_path,
        transform=transform,
        load_history=load_history,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=kwargs.get("pin_memory", True),
        persistent_workers=num_workers > 0,
    )
