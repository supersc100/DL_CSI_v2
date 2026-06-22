"""Baseline methods for FDD downlink CSI prediction."""
from typing import Dict, Optional

import torch

from src.data.transforms import (
    angle_delay_to_spatial,
    real_channels_to_complex,
    spatial_to_angle_delay,
)


def baseline_copy_ul(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
    transform=None,
) -> Dict[str, torch.Tensor]:
    """Baseline 1: predict DL = current UL (performance lower bound)."""
    pred = current_ul_ad.clone()
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_angle_delay_interp(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
    history_ul_ad: Optional[torch.Tensor] = None,
    history_dl_ad: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Baseline 2: linear interpolation in angle-delay domain from recent history."""
    if history_dl_ad is None or history_dl_ad.shape[1] < 2:
        # Not enough history: fall back to copying current UL.
        return baseline_copy_ul(current_ul_ad, target_dl_ad)
    # Use last two historical DL snapshots to linearly extrapolate next DL.
    h0 = history_dl_ad[:, -2]
    h1 = history_dl_ad[:, -1]
    pred = 2.0 * h1 - h0
    return {"pred_ad": pred, "target_ad": target_dl_ad}


def baseline_tdd_oracle(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Baseline 3: TDD upper bound, UL and DL share identical fast fading.

    In a real TDD oracle dataset h_ul == h_dl after calibration; here we simply
    copy the true DL target as the prediction to measure the dataset upper bound.
    """
    return {"pred_ad": target_dl_ad.clone(), "target_ad": target_dl_ad}


def baseline_no_large_scale(
    current_ul_ad: torch.Tensor,
    target_dl_ad: torch.Tensor,
    history_ul_ad: Optional[torch.Tensor] = None,
    history_dl_ad: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Baseline 4 (ablation): use current UL but no explicit large-scale params.

    The large-scale vector is ignored for this simple baseline.
    """
    return baseline_copy_ul(current_ul_ad, target_dl_ad, None)


def baseline_no_history(
    current_ul_ad: torch.Tensor,
    large_scale: torch.Tensor,
    target_dl_ad: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Baseline 5 (ablation): use current UL + large-scale but no history.

    Implemented as current UL only because the simple baseline cannot consume
    the large-scale vector without a trained model.
    """
    return baseline_copy_ul(current_ul_ad, target_dl_ad, None)


BASELINES = {
    "copy_ul": baseline_copy_ul,
    "angle_delay_interp": baseline_angle_delay_interp,
    "tdd_oracle": baseline_tdd_oracle,
    "no_large_scale": baseline_no_large_scale,
    "no_history": baseline_no_history,
}
