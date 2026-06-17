"""FDD UL/DL channel pair generation using Sionna 2.x.

This module depends on TensorFlow and Sionna and is intentionally isolated so
that training scripts can run on a pure-PyTorch machine if pre-generated H5
files are already available.

Generation idea:
- Downlink and uplink share large-scale parameters (AoD, AoA, delays, powers).
- Small-scale path coefficients are independently drawn for UL and DL.
- We extract ray parameters from a DL CDL realization, then synthesize the UL
  channel by reusing those parameters but resampling the fast-fading gains.
"""
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Sionna helpers
# ---------------------------------------------------------------------------
def _import_sionna():
    """Lazy import of Sionna / TensorFlow."""
    try:
        import sionna
        # import tensorflow as tf
        return sionna
    except ImportError as exc:
        raise ImportError(
            "Sionna/TensorFlow not installed. Install requirements or run on a "
            "machine with `pip install -r requirements.txt`."
        ) from exc


def _scenario_to_cdl(name: str):
    """Map scenario name to Sionna CDL model name."""
    mapping = {
        "UMa": "D",
        "UMi": "C",
        "RMa": "A",
    }
    return mapping.get(name, name)


def _make_array_config(array_cfg: Any, carrier_freq: float):
    """Build a Sionna antenna array description.

    Sionna 2.x API places array utilities under `sionna.channel`. This helper
    tries the common import paths and falls back gracefully.
    """
    sionna = _import_sionna()

    # Try common locations for the AntennaArray class.
    AntennaArray = getattr(sionna.phy.channel.tr38901, "AntennaArray", None)
    if AntennaArray is None:
        AntennaArray = getattr(sionna.phy.channel.tr38881, "AntennaArray", None)
    if AntennaArray is None:
        raise RuntimeError("Could not locate Sionna AntennaArray class.")

    num_elements = int(array_cfg.num_elements)
    polarization = str(array_cfg.polarization)
    spacing = float(array_cfg.spacing)

    if str(array_cfg.type).upper() == "ULA":
        num_rows = 1
        num_cols = num_elements
    else:
        # Simple UPA approximation: square-ish grid.
        num_rows = int(np.sqrt(num_elements))
        num_cols = int(np.ceil(num_elements / num_rows))

    array = AntennaArray(
        num_rows=num_rows,
        num_cols=num_cols,
        polarization=polarization,
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=carrier_freq,
        vertical_spacing=spacing,
        horizontal_spacing=spacing,
    )
    return array


def _make_resource_grid(config: Any):
    """Build a Sionna OFDM resource grid."""
    sionna = _import_sionna()
    ResourceGrid = getattr(sionna.phy.ofdm, "ResourceGrid", None)
    if ResourceGrid is None:
        raise RuntimeError("Could not locate Sionna ResourceGrid class.")

    num_subcarriers = int(config.data.num_subcarriers)
    num_ofdm_symbols = int(config.data.num_ofdm_symbols)
    subcarrier_spacing = float(config.data.subcarrier_spacing)
    num_slots = int(config.data.num_slots)

    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=int(config.data.cyclic_prefix_length),
        num_guard_carriers=[0, 0],
        dc_null=False,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
    )
    return rg, num_slots


# ---------------------------------------------------------------------------
# Ray extraction and manual channel synthesis
# ---------------------------------------------------------------------------
def _extract_ray_params(cdl, tau=None) -> Dict[str, np.ndarray]:
    """Extract large-scale ray parameters from a Sionna CDL object.

    Sionna 2.0 stores ray geometry as private attributes; names may shift
    between minor releases. We try the documented ones and warn on failure.

    Args:
        cdl: Sionna CDL object.
        tau: Optional path delays returned by ``cdl(...)``. Some Sionna
            versions expose delays only through this return value, not as
            an object attribute.
    """
    params: Dict[str, np.ndarray] = {}
    for attr, key in [
        ("_aod", "aod"),
        ("_aoa", "aoa"),
        ("_powers", "powers"),
        ("_tau", "tau"),
        ("_zod", "zod"),
        ("_zoa", "zoa"),
    ]:
        value = getattr(cdl, attr, None)
        if value is not None:
            params[key] = np.array(value)

    if tau is not None:
        # Sionna returns tau as [batch, rx, tx, num_paths]; flatten to 1-D.
        params["tau"] = np.array(tau).reshape(-1)

    if "tau" not in params:
        # `tau` may only exist after channel generation.
        params["tau"] = np.array([])
    return params


def _sample_independent_gains(num_rays: int, rng: np.random.Generator) -> np.ndarray:
    """Generate independent small-scale complex Gaussian gains."""
    real = rng.normal(size=num_rays).astype(np.complex64)
    imag = 1j * rng.normal(size=num_rays).astype(np.complex64)
    return (real + imag) / np.sqrt(2.0)


def _synthesize_cir_from_rays(
    ray_params: Dict[str, np.ndarray],
    num_tx: int,
    num_rx: int,
    carrier_freq: float,
    bandwidth: float,
    num_subcarriers: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize channel impulse response using extracted ray parameters.

    Returns:
        h: complex CIR of shape [num_rx, num_tx, num_paths]
        tau: delay profile of shape [num_paths]
        aoa, aod: angle arrays
    """
    aod = ray_params.get("aod")
    aoa = ray_params.get("aoa")
    powers = ray_params.get("powers")
    tau = ray_params.get("tau")

    if aod is None or aoa is None or powers is None:
        raise RuntimeError(
            "Ray parameters missing; cannot perform manual FDD synthesis."
        )

    # If tau not extracted, build a uniform delay grid scaled by RMS spread.
    if tau is None or tau.size == 0:
        tau = np.linspace(0.0, 1.0 / bandwidth, aod.size)

    # Ensure consistent 1-D ray vectors.
    aod = np.asarray(aod).reshape(-1)
    aoa = np.asarray(aoa).reshape(-1)
    powers = np.asarray(powers).reshape(-1)
    tau = np.asarray(tau).reshape(-1)

    num_rays = min(aod.size, aoa.size, powers.size, tau.size)
    gains = _sample_independent_gains(num_rays, rng)

    # Steering / response vectors using far-field ULA approximation.
    # For a uniform linear array along x with spacing d=lambda/2:
    #   a(theta) = exp(-j*pi*sin(theta)*[0,1,...,N-1])
    tx_idx = np.arange(num_tx)
    rx_idx = np.arange(num_rx)

    wavelength = 3e8 / carrier_freq
    d = wavelength * 0.5

    a_tx = np.exp(-1j * 2.0 * np.pi * d / wavelength * np.sin(np.radians(aod[:num_rays]))[:, None] * tx_idx[None, :])  # [R, Tx]
    a_rx = np.exp(-1j * 2.0 * np.pi * d / wavelength * np.sin(np.radians(aoa[:num_rays]))[:, None] * rx_idx[None, :])  # [R, Rx]

    # Path coefficient = sqrt(power) * gain.
    coeffs = np.sqrt(np.abs(powers[:num_rays])) * gains  # [R]

    # CIR h[rx, tx, ray]
    h = np.einsum("x,xr,xt->rtx", coeffs, a_rx, a_tx).astype(np.complex64)
    return h, tau[:num_rays], aoa[:num_rays], aod[:num_rays]


def _cir_to_frequency_response(
    h: np.ndarray,
    tau: np.ndarray,
    carrier_freq: float,
    subcarrier_spacing: float,
    num_subcarriers: int,
) -> np.ndarray:
    """Convert CIR to OFDM frequency response using exponential delay basis.

    Args:
        h: [num_rx, num_tx, num_paths]
        tau: [num_paths]

    Returns:
        H: [num_rx, num_tx, num_subcarriers]
    """
    subcarrier_indices = np.arange(num_subcarriers) - num_subcarriers // 2
    freqs = subcarrier_indices * subcarrier_spacing  # [M]
    # H(f) = sum_l h_l * exp(-j*2*pi*tau_l*f)
    phase = np.exp(-1j * 2.0 * np.pi * np.outer(tau, freqs))  # [L, M]
    H = np.einsum("rtx,lm->rtm", h, phase)  # [Rx, Tx, M]
    return H.astype(np.complex64)


def _compute_large_scale_features(
    h: np.ndarray,
    tau: np.ndarray,
    aoa: np.ndarray,
    aod: np.ndarray,
    powers: np.ndarray,
    bs_ue_distance: float,
) -> np.ndarray:
    """Compute a fixed-size large-scale parameter vector."""
    # Power statistics in dB.
    power_lin = np.abs(powers)
    power_lin = np.maximum(power_lin, 1e-20)
    avg_path_power_db = 10.0 * np.log10(power_lin.mean())

    # RMS delay spread.
    p_norm = power_lin / power_lin.sum()
    mean_tau = np.sum(p_norm * tau)
    rms_delay_spread = np.sqrt(np.maximum(np.sum(p_norm * (tau - mean_tau) ** 2), 0.0))

    # Angular spreads.
    def _angular_spread(angles: np.ndarray) -> float:
        rad = np.deg2rad(angles)
        sin = np.sum(p_norm * np.sin(rad))
        cos = np.sum(p_norm * np.cos(rad))
        spread = np.sqrt(np.maximum(1.0 - (sin ** 2 + cos ** 2), 0.0))
        return float(np.rad2deg(spread))

    aoa_spread = _angular_spread(aoa)
    aod_spread = _angular_spread(aod)

    # K-factor approximation: dominant path power / remaining power.
    sorted_p = np.sort(power_lin)[::-1]
    k_factor = float(sorted_p[0] / (sorted_p[1:].sum() + 1e-20))

    return np.array(
        [avg_path_power_db, rms_delay_spread, aoa_spread, aod_spread, k_factor, bs_ue_distance],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------
def _generate_one_sample(
    config: Any,
    rng: np.random.Generator,
    synthesize_ul: bool = True,
) -> Optional[Dict[str, Any]]:
    """Generate one FDD UL/DL channel sample."""
    sionna = _import_sionna()

    cdl_name = _scenario_to_cdl(config.data.scenario)
    dl_freq = float(config.data.dl_carrier_freq)
    ul_freq = float(config.data.ul_carrier_freq)
    bs_array = _make_array_config(config.data.bs_array, dl_freq)
    ue_array = _make_array_config(config.data.ue_array, dl_freq)

    num_tx = int(config.data.bs_array.num_elements)
    num_rx = int(config.data.ue_array.num_elements)

    # Number of slots: current slot + history window.
    num_slots = int(config.data.num_slots)
    history_window = int(config.data.history_window)

    # Placeholder distance (path-loss is not the focus; geometry is random).
    distance = float(rng.uniform(50.0, 500.0))

    # --- Downlink channel ---------------------------------------------------
    from sionna.phy.channel.tr38901 import CDL
    cdl_dl = CDL(
        model=cdl_name,
        delay_spread=100e-9,
        carrier_frequency=dl_freq,
        ut_array=ue_array,
        bs_array=bs_array,
        direction="downlink",
        min_speed=0.0,
        max_speed=float(rng.uniform(0.0, 3.0)),
    )

    # Trigger ray generation by calling the CDL once; parameters like _aod,
    # _aoa, and _powers are populated after this call. We do not actually use
    # the path coefficients, but we capture the returned path delays `tau`
    # because Sionna 2.x exposes them through the call return value, not as a
    # CDL attribute.
    sampling_frequency = float(config.data.bandwidth)
    _, tau = cdl_dl(batch_size=1, num_time_steps=1, sampling_frequency=sampling_frequency)

    # Extract DL ray parameters.
    ray_params_dl = _extract_ray_params(cdl_dl, tau=tau)

    # Manually synthesize DL frequency response from extracted rays.
    h_cir_dl, tau_dl, aoa_dl, aod_dl = _synthesize_cir_from_rays(
        ray_params_dl,
        num_tx,
        num_rx,
        dl_freq,
        float(config.data.bandwidth),
        int(config.data.num_subcarriers),
        rng,
    )
    h_dl = _cir_to_frequency_response(
        h_cir_dl,
        tau_dl,
        dl_freq,
        float(config.data.subcarrier_spacing),
        int(config.data.num_subcarriers),
    )

    # --- Uplink channel -----------------------------------------------------
    if synthesize_ul:
        # Reuse DL ray parameters but resample small-scale gains for UL.
        h_cir_ul, tau_ul, aoa_ul, aod_ul = _synthesize_cir_from_rays(
            ray_params_dl,  # shared large-scale
            num_tx,
            num_rx,
            ul_freq,
            float(config.data.bandwidth),
            int(config.data.num_subcarriers),
            rng,
        )
    else:
        # TDD oracle: identical small-scale (use same gains as DL).
        h_cir_ul = h_cir_dl
        tau_ul, aoa_ul, aod_ul = tau_dl, aoa_dl, aod_dl

    h_ul = _cir_to_frequency_response(
        h_cir_ul,
        tau_ul,
        ul_freq,
        float(config.data.subcarrier_spacing),
        int(config.data.num_subcarriers),
    )

    # Canonical shape: [num_tx_bs, num_rx_ue, num_subcarriers]
    h_ul = h_ul.transpose(1, 0, 2)  # Tx, Rx, M
    h_dl = h_dl.transpose(1, 0, 2)

    # Large-scale features.
    large_scale = _compute_large_scale_features(
        h_cir_dl, tau_dl, aoa_dl, aod_dl, ray_params_dl.get("powers", np.ones_like(tau_dl)), distance
    )

    # Time evolution (optional): for history, resample small-scale while keeping
    # large-scale fixed. This produces T correlated slots.
    history_ul = []
    history_dl = []
    for _ in range(history_window):
        h_cir_t, _, _, _ = _synthesize_cir_from_rays(
            ray_params_dl, num_tx, num_rx, ul_freq,
            float(config.data.bandwidth), int(config.data.num_subcarriers), rng,
        )
        h_t_ul = _cir_to_frequency_response(
            h_cir_t, tau_dl, ul_freq,
            float(config.data.subcarrier_spacing), int(config.data.num_subcarriers),
        ).transpose(1, 0, 2)
        history_ul.append(h_t_ul)

        h_cir_t_dl, _, _, _ = _synthesize_cir_from_rays(
            ray_params_dl, num_tx, num_rx, dl_freq,
            float(config.data.bandwidth), int(config.data.num_subcarriers), rng,
        )
        h_t_dl = _cir_to_frequency_response(
            h_cir_t_dl, tau_dl, dl_freq,
            float(config.data.subcarrier_spacing), int(config.data.num_subcarriers),
        ).transpose(1, 0, 2)
        history_dl.append(h_t_dl)

    history_ul = np.stack(history_ul, axis=0).astype(np.complex64)  # [T, Tx, Rx, M]
    history_dl = np.stack(history_dl, axis=0).astype(np.complex64)

    sample = {
        "h_ul": h_ul,
        "h_dl": h_dl,
        "history_ul": history_ul,
        "history_dl": history_dl,
        "large_scale": large_scale,
        "tau": tau_dl.astype(np.float32),
        "aoa": aoa_dl.astype(np.float32),
        "aod": aod_dl.astype(np.float32),
        "powers": np.asarray(ray_params_dl.get("powers", np.ones_like(tau_dl))).astype(np.float32),
    }
    return sample


def generate_dataset(
    config: Any,
    num_samples: int,
    output_path: str,
    seed_offset: int = 0,
    synthesize_ul: bool = True,
) -> None:
    """Generate and save an H5 dataset of FDD UL/DL channel pairs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(int(config.project.seed) + seed_offset)

    num_subcarriers = int(config.data.num_subcarriers)
    num_tx = int(config.data.bs_array.num_elements)
    num_rx = int(config.data.ue_array.num_elements)
    history_window = int(config.data.history_window)
    num_ls = len(config.data.large_scale_params)

    batch_size = int(config.data.batch_size_per_file)

    with h5py.File(output_path, "w") as f:
        # Create resizable datasets.
        ds_h_ul = f.create_dataset(
            "h_ul", shape=(num_samples, num_tx, num_rx, num_subcarriers),
            dtype=np.complex64, chunks=(1, num_tx, num_rx, num_subcarriers)
        )
        ds_h_dl = f.create_dataset(
            "h_dl", shape=(num_samples, num_tx, num_rx, num_subcarriers),
            dtype=np.complex64, chunks=(1, num_tx, num_rx, num_subcarriers)
        )
        ds_history_ul = f.create_dataset(
            "history_ul", shape=(num_samples, history_window, num_tx, num_rx, num_subcarriers),
            dtype=np.complex64, chunks=(1, 1, num_tx, num_rx, num_subcarriers)
        )
        ds_history_dl = f.create_dataset(
            "history_dl", shape=(num_samples, history_window, num_tx, num_rx, num_subcarriers),
            dtype=np.complex64, chunks=(1, 1, num_tx, num_rx, num_subcarriers)
        )
        ds_large_scale = f.create_dataset(
            "large_scale", shape=(num_samples, num_ls), dtype=np.float32
        )
        if config.data.save_ray_info:
            ds_tau = f.create_dataset("tau", shape=(num_samples,), dtype=h5py.vlen_dtype(np.float32))
            ds_aoa = f.create_dataset("aoa", shape=(num_samples,), dtype=h5py.vlen_dtype(np.float32))
            ds_aod = f.create_dataset("aod", shape=(num_samples,), dtype=h5py.vlen_dtype(np.float32))
            ds_powers = f.create_dataset("powers", shape=(num_samples,), dtype=h5py.vlen_dtype(np.float32))

        for i in range(num_samples):
            sample = _generate_one_sample(config, rng, synthesize_ul=synthesize_ul)
            if sample is None:
                warnings.warn(f"Sample {i} generation failed; skipping.")
                continue

            ds_h_ul[i] = sample["h_ul"]
            ds_h_dl[i] = sample["h_dl"]
            ds_history_ul[i] = sample["history_ul"]
            ds_history_dl[i] = sample["history_dl"]
            ds_large_scale[i] = sample["large_scale"]

            if config.data.save_ray_info:
                ds_tau[i] = sample["tau"]
                ds_aoa[i] = sample["aoa"]
                ds_aod[i] = sample["aod"]
                ds_powers[i] = sample["powers"]

            if (i + 1) % batch_size == 0 or i == num_samples - 1:
                print(f"Generated {i + 1}/{num_samples} samples for {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate FDD CSI dataset with Sionna.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--tdd-oracle", action="store_true", help="Use identical UL/DL fast fading (TDD upper bound).")
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)

    num_samples = int(getattr(cfg.data.samples, args.split))
    output_path = getattr(cfg.data, f"h5_{args.split}")
    generate_dataset(cfg, num_samples, output_path, seed_offset=args.seed_offset, synthesize_ul=not args.tdd_oracle)
