"""PyTorch Dataset for FDD UL/DL CSI pairs stored in H5 files."""
import os
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import AngleDelayTransform
from src.models.sampling_mask import SamplingMaskGenerator
from src.utils.channel_noise import add_awgn


class FddCsiDataset(Dataset):
    """Dataset that loads UL/DL CSI pairs and converts them to angle-delay domain.

    Expected H5 keys:
        h_ul:          [N, num_tx_bs, num_rx_ue, M]
        h_dl:          [N, num_tx_bs, num_rx_ue, M]
        history_ul:    [N, T, num_tx_bs, num_rx_ue, M]
        history_dl:    [N, T, num_tx_bs, num_rx_ue, M]
        large_scale:   [N, num_large_scale]

    Args:
        h5_path: Path to H5 file.
        transform: AngleDelayTransform instance.
        load_history: If True, load history from H5; otherwise fill with zeros.
            Only used when use_history=True.
        return_spatial: If True, keep raw spatial fields in samples.
        use_history: If False, the dataset does not load, transform, or return
            any history fields. This is used for the no-history ablation.
        use_large_scale: If False, the dataset does not load or return the
            large_scale field. This is used for the no-large-scale ablation.
        phase2_enabled: If True, generate sparse downlink subband samples and
            sampling masks for Phase 2 training/evaluation.
        mask_generator: Optional SamplingMaskGenerator for Phase 2.  If None
            and phase2_enabled is True, a default generator is created.
    """

    def __init__(
        self,
        h5_path: str,
        transform: Optional[AngleDelayTransform] = None,
        load_history: bool = True,
        return_spatial: bool = False,
        use_history: bool = True,
        use_large_scale: bool = True,
        phase2_enabled: bool = False,
        mask_generator: Optional[SamplingMaskGenerator] = None,
    ):
        self.h5_path = h5_path
        self.transform = transform or AngleDelayTransform()
        self.load_history = load_history and use_history
        self.return_spatial = return_spatial or phase2_enabled
        self.use_history = use_history
        self.use_large_scale = use_large_scale
        self.phase2_enabled = phase2_enabled
        self.training = True  # set to False for deterministic validation masks
        # Eval-time AWGN injection (None = clean).  When set, noise is added to
        # the UL input and the DL sparse pilots; the supervision target stays
        # clean.  noise_base_seed makes the per-sample noise reproducible.
        self.snr_db: Optional[float] = None
        self.noise_base_seed: int = 0
        # Phase2 needs large_scale for adaptive sampling even if Stage1 doesn't.
        self._load_large_scale = use_large_scale or phase2_enabled

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            self.num_samples = f["h_ul"].shape[0]
            self.num_tx = f["h_ul"].shape[1]
            self.num_rx = f["h_ul"].shape[2]
            self.num_subcarriers = f["h_ul"].shape[3]
            self.num_large_scale = f["large_scale"].shape[1]
            if self.use_history:
                self.history_window = f["history_ul"].shape[1] if self.load_history else 0
            else:
                self.history_window = 0

        if self.phase2_enabled:
            if mask_generator is None:
                self.mask_generator = SamplingMaskGenerator(
                    num_subcarriers=self.num_subcarriers
                )
            else:
                if mask_generator.num_subcarriers != self.num_subcarriers:
                    raise ValueError(
                        f"mask_generator M={mask_generator.num_subcarriers} "
                        f"does not match dataset M={self.num_subcarriers}"
                    )
                self.mask_generator = mask_generator

    def __len__(self) -> int:
        return self.num_samples

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            h_ul = np.array(f["h_ul"][idx])
            h_dl = np.array(f["h_dl"][idx])
            if self._load_large_scale:
                large_scale = np.array(f["large_scale"][idx])
            if self.use_history and self.load_history:
                history_ul = np.array(f["history_ul"][idx])
                history_dl = np.array(f["history_dl"][idx])
            elif self.use_history:
                history_ul = np.zeros(
                    (self.history_window, self.num_tx, self.num_rx, self.num_subcarriers),
                    dtype=np.complex64,
                )
                history_dl = history_ul.copy()

        sample = {
            "h_ul": self._to_tensor(h_ul),
            "h_dl": self._to_tensor(h_dl),
        }
        if self._load_large_scale:
            sample["large_scale"] = self._to_tensor(large_scale).float()
        if self.use_history:
            sample["history_ul"] = self._to_tensor(history_ul)
            sample["history_dl"] = self._to_tensor(history_dl)

        # Optional eval-time AWGN.  The mask above is generated from the *clean*
        # UL energy so that sampled positions stay fixed across SNR points; noise
        # is only added to the measured signals (UL input, DL sparse pilots).
        noise_gen = None
        if self.snr_db is not None:
            noise_gen = torch.Generator()
            noise_gen.manual_seed(int(self.noise_base_seed) + int(idx))

        # Phase 2: generate sampling mask from UL spatial-frequency energy.
        if self.phase2_enabled:
            mask = self.mask_generator(
                sample["h_ul"],
                large_scale=sample.get("large_scale"),
                training=self.training,
            )
            sample["sampling_mask"] = mask

            # Create sparse downlink CSI by zeroing unsampled subcarriers.
            # Under AWGN, the sampled pilots are noisy observations of h_dl.
            dl_measured = sample["h_dl"]
            if self.snr_db is not None:
                dl_measured = add_awgn(dl_measured, self.snr_db, generator=noise_gen)
            sparse_dl = dl_measured.clone()
            sparse_dl[:, :, ~mask] = 0.0
            sample["sparse_dl"] = sparse_dl

        # Convert current UL to angle-delay domain for model input.  Add noise to
        # the UL estimate (Stage1 input) when an SNR is configured.
        ul_input = sample["h_ul"]
        if self.snr_db is not None:
            ul_input = add_awgn(ul_input, self.snr_db, generator=noise_gen)
        h_ul_ad, ul_stats = self.transform(ul_input)
        # Convert target DL to angle-delay domain for supervision (always clean).
        h_dl_ad, dl_stats = self.transform(sample["h_dl"])

        sample["h_ul_ad"] = h_ul_ad
        sample["h_dl_ad"] = h_dl_ad

        if self.phase2_enabled:
            # Transform sparse DL to angle-delay domain.
            sparse_dl_ad, sparse_dl_stats = self.transform(sample["sparse_dl"])
            sample["sparse_dl_ad"] = sparse_dl_ad
            sample["sparse_dl_stats"] = (
                sparse_dl_stats if sparse_dl_stats is not None
                else torch.zeros(2, *sparse_dl_ad.shape)
            )

        if self.use_history:
            # Convert history pairs to angle-delay domain using the same transform.
            history_ul_ad = []
            history_dl_ad = []
            for t in range(self.history_window):
                h_t_ul_ad, _ = self.transform(sample["history_ul"][t])
                h_t_dl_ad, _ = self.transform(sample["history_dl"][t])
                history_ul_ad.append(h_t_ul_ad)
                history_dl_ad.append(h_t_dl_ad)

            sample["history_ul_ad"] = torch.stack(history_ul_ad, dim=0)
            sample["history_dl_ad"] = torch.stack(history_dl_ad, dim=0)

        # Keep normalization stats for inverse transform during inference.
        sample["ul_stats"] = ul_stats if ul_stats is not None else torch.zeros(2, *h_ul_ad.shape)
        sample["dl_stats"] = dl_stats if dl_stats is not None else torch.zeros(2, *h_dl_ad.shape)

        if not self.return_spatial:
            # Drop raw spatial fields to save memory.
            sample.pop("h_ul", None)
            sample.pop("h_dl", None)
            if self.phase2_enabled:
                sample.pop("sparse_dl", None)
            if self.use_history:
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
    use_history: bool = True,
    use_large_scale: bool = True,
    phase2_enabled: bool = False,
    mask_generator: Optional[SamplingMaskGenerator] = None,
    training: bool = True,
    snr_db: Optional[float] = None,
    noise_base_seed: int = 0,
    **kwargs: Any,
):
    """Create a DataLoader from an H5 dataset."""
    dataset = FddCsiDataset(
        h5_path=h5_path,
        transform=transform,
        load_history=load_history,
        use_history=use_history,
        use_large_scale=use_large_scale,
        phase2_enabled=phase2_enabled,
        mask_generator=mask_generator,
    )
    dataset.training = training
    dataset.snr_db = snr_db
    dataset.noise_base_seed = noise_base_seed
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=kwargs.get("pin_memory", True),
        persistent_workers=num_workers > 0,
    )
