"""Unified simulation sweep harness for the two-stage FDD CSI project.

Produces the paper curves described in researchPlan/simPlan.md by sweeping an
x-axis (receive SNR or pilot overhead) and evaluating the trained model plus
baselines.  Noise and sampling overhead are injected at *evaluation* time, so
existing checkpoints are reused without retraining.

Figures:
    nmse_snr          NMSE (dB) vs SNR (dB)            [simPlan 1.1]
    se_snr            Spectral efficiency vs SNR       [simPlan 1.3]
    nmse_overhead     NMSE (dB) vs pilot overhead (%)  [simPlan 1.2]
    sampling_overhead NMSE (dB) vs overhead, per       [simPlan 2.1 fig 4c]
                      sampling strategy

Outputs <output_dir>/results_<figure>.csv and fig_<figure>.png.

Examples:
    python scripts/run_simulation.py --figure nmse_snr \
        --checkpoint outputs/checkpoints/phase2_main_best.pt --num-samples 512
    python scripts/run_simulation.py --figure se_snr \
        --checkpoint outputs/checkpoints/phase2_main_best.pt
    python scripts/run_simulation.py --figure nmse_overhead \
        --checkpoint outputs/checkpoints/phase2_qat_best.pt --quant-curve
    python scripts/run_simulation.py --figure sampling_overhead \
        --checkpoint outputs/checkpoints/phase2_main_best.pt
"""
import argparse
import csv
import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import load_config
from src.data.dataset import build_dataloader
from src.data.transforms import AngleDelayTransform, angle_delay_to_spatial
from src.models.dl_csi_predictor import DlCsiPredictor
from src.models.phase_recovery_network import PhaseRecoveryNetwork
from src.models.quantization import ScalarQuantizer
from src.models.sampling_mask import SamplingMaskGenerator, make_mask_generator, overhead_to_spacing
from src.utils.baselines import BASELINES
from src.utils.metrics import compute_phase2_metrics, nmse, spectral_efficiency
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------
def _sim_cfg(config, key, default):
    sim = getattr(config, "simulation", None)
    if sim is None:
        return default
    return getattr(sim, key, default)


def _resolve_quant_bits(args, config):
    """Resolve quantization bit-width list from CLI and config.

    Priority:
      1. --quant-bits explicit list
      2. --quant-curve flag -> use config default list
      3. None -> no quantization curves
    """
    if args.quant_bits is not None:
        return args.quant_bits
    if args.quant_curve:
        return _sim_cfg(config, "quant_bits_list",
                        [int(config.phase2.quantization.num_bits)])
    return None


def _quant_curve_name(num_bits):
    """Return curve name for a given quantization bit-width."""
    return f"proposed_quant_{num_bits}bit"


def _build_quantizers(bit_widths):
    """Build a dict mapping curve name to ScalarQuantizer."""
    return {_quant_curve_name(b): ScalarQuantizer(num_bits=b) for b in bit_widths}


def parse_args():
    p = argparse.ArgumentParser(description="Run paper simulation sweeps.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--figure",
        required=True,
        choices=["nmse_snr", "se_snr", "nmse_overhead", "sampling_overhead"],
    )
    p.add_argument("--stage", type=int, default=2, choices=[1, 2])
    p.add_argument("--checkpoint", default=None, help="Stage2 (or Stage1) checkpoint.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--snr-list", type=float, nargs="+", default=None)
    p.add_argument("--overhead-list", type=float, nargs="+", default=None)
    p.add_argument("--overhead-strategy", default="nonuniform",
                   choices=["uniform", "nonuniform"],
                   help="Sampling strategy for the nmse_overhead sweep. "
                        "'nonuniform' keeps peak encryption (closer to the trained "
                        "model's input); 'uniform' tracks nominal overhead more "
                        "closely and reaches lower overheads. ('adaptive' cannot be "
                        "swept: it auto-picks spacing from the delay spread.)")
    p.add_argument("--fixed-snr", type=float, default=None,
                   help="SNR (dB) for overhead sweeps. Default from config.")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--quant-curve", action="store_true",
                   help="使用 simulation.quant_bits_list 启用量化曲线。")
    p.add_argument("--quant-bits", type=int, nargs="+", default=None,
                   help="覆盖量化比特宽度，例如 --quant-bits 16 8 4 2。")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------
def build_loader(config, transform, h5_path, mask_generator, snr_db,
                 noise_seed, num_samples, phase2):
    loader = build_dataloader(
        h5_path=str(h5_path),
        batch_size=int(config.phase2.evaluation.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        pin_memory=bool(config.training.pin_memory),
        transform=transform,
        load_history=True,
        use_history=bool(config.model.use_history),
        use_large_scale=bool(config.model.use_large_scale),
        phase2_enabled=phase2,
        mask_generator=mask_generator,
        training=False,
        snr_db=snr_db,
        noise_base_seed=noise_seed,
    )
    if num_samples is not None:
        loader.dataset.num_samples = min(num_samples, loader.dataset.num_samples)
    return loader


def _to_device(batch, device):
    out = {}
    for k in ("h_ul_ad", "sparse_dl_ad", "sampling_mask", "h_dl_ad",
              "large_scale", "history_ul_ad", "history_dl_ad"):
        v = batch.get(k)
        out[k] = v.to(device) if v is not None else None
    return out


# ---------------------------------------------------------------------------
# Stage2 evaluation (one pass over loader)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_model_stage2(model, loader, device, compute_se, se_snr, quantizer=None):
    model.eval()
    agg, n = {}, 0
    for batch in loader:
        b = _to_device(batch, device)
        sparse = b["sparse_dl_ad"]
        if quantizer is not None:
            sparse = quantizer(sparse)
        pred, aux = model(
            b["h_ul_ad"], sparse, b["sampling_mask"],
            large_scale=b["large_scale"],
            history_ul_ad=b["history_ul_ad"], history_dl_ad=b["history_dl_ad"],
        )
        target = b["h_dl_ad"]
        m = compute_phase2_metrics(pred, target, aux["mag_stage1"])
        if compute_se:
            pred_sp = angle_delay_to_spatial(pred)
            tgt_sp = angle_delay_to_spatial(target)
            m.update(spectral_efficiency(pred_sp, tgt_sp, se_snr))
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def eval_baseline_stage2(model, loader, device, name, compute_se, se_snr,
                         full_fb_quantizer=None):
    agg, n = {}, 0
    for batch in loader:
        b = _to_device(batch, device)
        target = b["h_dl_ad"]
        kwargs = {"target_dl_ad": target}
        if name == "magnitude_only":
            s1_kwargs = {}
            if model.stage1.use_large_scale:
                s1_kwargs["large_scale"] = b["large_scale"]
            if model.stage1.use_history:
                s1_kwargs["history_ul_ad"] = b["history_ul_ad"]
                s1_kwargs["history_dl_ad"] = b["history_dl_ad"]
            kwargs["stage1_pred"] = model.stage1(b["h_ul_ad"], **s1_kwargs)
        elif name in ("linear_interp", "dft_interp"):
            kwargs["sparse_dl_ad"] = b["sparse_dl_ad"]
            kwargs["mask"] = b["sampling_mask"]
        elif name == "full_feedback":
            kwargs["quantizer"] = full_fb_quantizer
        elif name == "tdd_oracle":
            kwargs["current_ul_ad"] = b["h_ul_ad"]
        else:
            continue
        pred = BASELINES[name](**kwargs)["pred_ad"]
        m = compute_phase2_metrics(pred, target)
        if compute_se:
            m.update(spectral_efficiency(
                angle_delay_to_spatial(pred), angle_delay_to_spatial(target), se_snr))
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def _update_metrics(agg, metrics, compute_se, pred, target, se_snr):
    """Add a batch of metrics (and optionally SE) to an aggregate dict."""
    for k, v in metrics.items():
        agg[k] = agg.get(k, 0.0) + v
    if compute_se:
        pred_sp = angle_delay_to_spatial(pred)
        tgt_sp = angle_delay_to_spatial(target)
        for k, v in spectral_efficiency(pred_sp, tgt_sp, se_snr).items():
            agg[k] = agg.get(k, 0.0) + v


@torch.no_grad()
def eval_all_stage2(model, loader, device, curves, compute_se=False, se_snr=0.0,
                    quantizers=None, full_fb_quantizer=None):
    """Evaluate multiple Stage2 curves in a **single pass** over ``loader``.

    This avoids re-reading the H5 file and re-running the model for every
    baseline, which is the main bottleneck for ``nmse_snr`` and
    ``nmse_overhead``.

    Supported curve names: proposed, proposed_quant_{N}bit, magnitude_only,
    linear_interp, dft_interp, tdd_oracle, full_feedback.
    """
    model.eval()
    aggs = {c: {} for c in curves}
    n = 0
    for batch in loader:
        b = _to_device(batch, device)
        target = b["h_dl_ad"]
        h_ul = b["h_ul_ad"]
        sparse = b["sparse_dl_ad"]
        mask = b["sampling_mask"]
        large_scale = b["large_scale"]
        history_ul = b["history_ul_ad"]
        history_dl = b["history_dl_ad"]

        # Proposed model (and optionally its quantized-pilot variants).
        if "proposed" in curves or any(c in curves for c in (quantizers or {})):
            pred, aux = model(
                h_ul, sparse, mask,
                large_scale=large_scale,
                history_ul_ad=history_ul, history_dl_ad=history_dl,
            )
            if "proposed" in curves:
                _update_metrics(
                    aggs["proposed"],
                    compute_phase2_metrics(pred, target, aux["mag_stage1"]),
                    compute_se, pred, target, se_snr)
            for q_curve, q_obj in (quantizers or {}).items():
                if q_curve in curves:
                    sparse_q = q_obj(sparse)
                    pred_q, aux_q = model(
                        h_ul, sparse_q, mask,
                        large_scale=large_scale,
                        history_ul_ad=history_ul, history_dl_ad=history_dl,
                    )
                    _update_metrics(
                        aggs[q_curve],
                        compute_phase2_metrics(pred_q, target, aux_q["mag_stage1"]),
                        compute_se, pred_q, target, se_snr)

        # Stage1-magnitude-only baseline (needs one Stage1 forward).
        if "magnitude_only" in curves:
            s1_kwargs = {}
            if model.stage1.use_large_scale:
                s1_kwargs["large_scale"] = large_scale
            if model.stage1.use_history:
                s1_kwargs["history_ul_ad"] = history_ul
                s1_kwargs["history_dl_ad"] = history_dl
            stage1_pred = model.stage1(h_ul, **s1_kwargs)
            pred_mag = BASELINES["magnitude_only"](stage1_pred, target)["pred_ad"]
            _update_metrics(
                aggs["magnitude_only"],
                compute_phase2_metrics(pred_mag, target),
                compute_se, pred_mag, target, se_snr)

        # Interpolation baselines.
        if "linear_interp" in curves:
            pred_li = BASELINES["linear_interp"](sparse, mask, target)["pred_ad"]
            _update_metrics(
                aggs["linear_interp"],
                compute_phase2_metrics(pred_li, target),
                compute_se, pred_li, target, se_snr)
        if "dft_interp" in curves:
            pred_di = BASELINES["dft_interp"](sparse, mask, target)["pred_ad"]
            _update_metrics(
                aggs["dft_interp"],
                compute_phase2_metrics(pred_di, target),
                compute_se, pred_di, target, se_snr)

        # Oracle / ceiling baselines.
        if "tdd_oracle" in curves:
            pred_to = BASELINES["tdd_oracle"](h_ul, target)["pred_ad"]
            _update_metrics(
                aggs["tdd_oracle"],
                compute_phase2_metrics(pred_to, target),
                compute_se, pred_to, target, se_snr)
        if "full_feedback" in curves:
            pred_fb = BASELINES["full_feedback"](target, full_fb_quantizer)["pred_ad"]
            _update_metrics(
                aggs["full_feedback"],
                compute_phase2_metrics(pred_fb, target),
                compute_se, pred_fb, target, se_snr)

        n += 1
    return {c: {k: v / max(n, 1) for k, v in aggs[c].items()} for c in curves}


# ---------------------------------------------------------------------------
# Stage1 evaluation (magnitude NMSE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_model_stage1(model, loader, device):
    model.eval()
    use_h = model.use_history
    use_ls = model.use_large_scale
    agg, n = 0.0, 0
    for batch in loader:
        ul = batch["h_ul_ad"].to(device)
        target = batch["h_dl_ad"].to(device)
        kw = {}
        if use_ls:
            kw["large_scale"] = batch["large_scale"].to(device)
        if use_h:
            kw["history_ul_ad"] = batch["history_ul_ad"].to(device)
            kw["history_dl_ad"] = batch["history_dl_ad"].to(device)
        pred = model(ul, **kw)
        agg += float(nmse(pred.abs(), target.abs()).item())
        n += 1
    return {"magnitude_nmse_db": agg / max(n, 1)}


# ---------------------------------------------------------------------------
# Sampling-mask average overhead (for feedback-bit reporting)
# ---------------------------------------------------------------------------
def measure_overhead(loader):
    counts, M, tot = 0, 0, 0
    for batch in loader:
        mask = batch["sampling_mask"]
        counts += int(mask.sum().item())
        M = mask.shape[-1]
        tot += mask.shape[0]
        break  # first batch is representative
    if tot == 0 or M == 0:
        return 0.0, 0
    avg_sampled = counts / tot
    return avg_sampled / M, int(round(avg_sampled))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def run_nmse_snr(args, config, transform, device, h5_path):
    snr_list = args.snr_list or _sim_cfg(config, "snr_list",
                                         [-10, -5, 0, 5, 10, 15, 20, 25, 30])
    seeds = args.seeds or _sim_cfg(config, "seeds", [42])

    if args.stage == 1:
        return _sweep_stage1_snr(args, config, transform, device, h5_path,
                                 snr_list, seeds)

    model = _load_stage2(args, config, device)
    quant_bits = _resolve_quant_bits(args, config)
    quantizers = _build_quantizers(quant_bits) if quant_bits else None

    curves = ["proposed", "magnitude_only", "linear_interp", "tdd_oracle",
              "full_feedback"]
    if quant_bits:
        for q_curve in reversed(list(quantizers.keys())):
            curves.insert(1, q_curve)

    rows = []
    for snr in snr_list:
        per_curve = {c: [] for c in curves}
        for sd in seeds:
            loader = build_loader(config, transform, h5_path, _eval_mask(config),
                                  snr, sd, args.num_samples, phase2=True)
            results = eval_all_stage2(
                model, loader, device, curves, compute_se=False, se_snr=0.0,
                quantizers=quantizers,
                full_fb_quantizer=quantizers.get(_quant_curve_name(16)) if quantizers else None)
            for c in curves:
                per_curve[c].append(results[c]["nmse_db"])
        rows.extend(_reduce_rows(per_curve, snr, "nmse_db"))
        print(f"[nmse_snr] SNR={snr:>4} dB done")
    return rows, "SNR (dB)", "NMSE (dB)"


def run_se_snr(args, config, transform, device, h5_path):
    snr_list = args.snr_list or _sim_cfg(config, "snr_list",
                                         [-10, -5, 0, 5, 10, 15, 20, 25, 30])
    seeds = args.seeds or _sim_cfg(config, "seeds", [42])
    model = _load_stage2(args, config, device)

    rows = []
    for snr in snr_list:
        prop, perf = [], []
        for sd in seeds:
            loader = build_loader(config, transform, h5_path, _eval_mask(config),
                                  snr, sd, args.num_samples, phase2=True)
            m = eval_model_stage2(model, loader, device, True, snr)
            prop.append(m["se_pred"])
            perf.append(m["se_perfect"])
        rows.extend(_reduce_rows({"proposed": prop, "perfect_csi": perf}, snr, "se"))
        print(f"[se_snr] SNR={snr:>4} dB done")
    return rows, "SNR (dB)", "SE (bps/Hz)"


def run_nmse_overhead(args, config, transform, device, h5_path):
    ov_list = args.overhead_list or _sim_cfg(
        config, "overhead_list", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    seeds = args.seeds or _sim_cfg(config, "seeds", [42])
    snr = args.fixed_snr if args.fixed_snr is not None else _sim_cfg(
        config, "fixed_snr", 10.0)
    model = _load_stage2(args, config, device)
    quant_bits = _resolve_quant_bits(args, config)
    quantizers = _build_quantizers(quant_bits) if quant_bits else None
    M = int(config.data.num_subcarriers)

    # Curves that actually consume the sparse pilots and therefore vary with
    # overhead.  tdd_oracle / full_feedback / magnitude_only do NOT use pilots,
    # so they are not swept: tdd_oracle is drawn as a flat ceiling reference,
    # magnitude_only is the 0%-overhead anchor (phase=0, no DL pilots), and
    # full_feedback is the 100%-overhead anchor (full-band CSI fed back).
    swept = ["proposed", "linear_interp"]
    if quant_bits:
        for q_curve in reversed(list(quantizers.keys())):
            swept.insert(1, q_curve)

    rows = []
    for ov in ov_list:
        # The adaptive strategy auto-picks spacing from the delay spread and
        # ignores base_spacing, which collapses every target overhead to the same
        # actual overhead (single point), so it cannot be swept. Use a
        # spacing-controllable strategy: 'nonuniform' (uniform grid + peak
        # encryption, closer to the trained model's input) or 'uniform' (cleaner
        # nominal-overhead axis, reaches lower overheads). Selected via
        # --overhead-strategy.
        mg = make_mask_generator(args.overhead_strategy, M, target_overhead=ov)
        # Evaluate all overhead-dependent curves in one pass per seed.
        loader_curves = list(swept)
        if "tdd_oracle" not in loader_curves:
            loader_curves.append("tdd_oracle")
        per_curve = {c: [] for c in loader_curves}
        for sd in seeds:
            loader = build_loader(config, transform, h5_path, mg, snr, sd,
                                  args.num_samples, phase2=True)
            actual_ov, _ = measure_overhead(loader)
            results = eval_all_stage2(
                model, loader, device, loader_curves, compute_se=False,
                quantizers=quantizers,
                full_fb_quantizer=quantizers.get(_quant_curve_name(16)) if quantizers else None)
            for c in loader_curves:
                per_curve[c].append(results[c]["nmse_db"])
        x = round(actual_ov * 100, 2)
        rows.extend(_reduce_rows(per_curve, x, "nmse_db"))
        print(f"[nmse_overhead] overhead~{actual_ov*100:.1f}% done")

    # Anchor points (pilot-independent): magnitude_only @ 0%, full_feedback @ 100%.
    anchor_curves = ["magnitude_only", "full_feedback"]
    anchor_results = {c: [] for c in anchor_curves}
    for sd in seeds:
        loader = build_loader(config, transform, h5_path, _eval_mask(config), snr, sd,
                              args.num_samples, phase2=True)
        res = eval_all_stage2(
            model, loader, device, anchor_curves, compute_se=False,
            full_fb_quantizer=quantizers.get(_quant_curve_name(16)) if quantizers else None)
        for c in anchor_curves:
            anchor_results[c].append(res[c]["nmse_db"])
    rows.extend(_reduce_rows({"magnitude_only": anchor_results["magnitude_only"]}, 0.0, "nmse_db"))
    rows.extend(_reduce_rows({"full_feedback": anchor_results["full_feedback"]}, 100.0, "nmse_db"))
    print("[nmse_overhead] anchors: magnitude_only@0%, full_feedback@100% done")
    return rows, "Pilot overhead (%)", "NMSE (dB)"



def run_sampling_overhead(args, config, transform, device, h5_path):
    ov_list = args.overhead_list or _sim_cfg(
        config, "overhead_list", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    seeds = args.seeds or _sim_cfg(config, "seeds", [42])
    snr = args.fixed_snr if args.fixed_snr is not None else _sim_cfg(
        config, "fixed_snr", 10.0)
    model = _load_stage2(args, config, device)
    M = int(config.data.num_subcarriers)

    # Fair ablation (control variables): all curves share the same adaptive base
    # grid (spacing chosen from the RMS delay spread); they differ ONLY in the
    # extra peak encryption strategy:
    #   - no_peak:      no extra encryption (baseline).
    #   - random_peak:  same budget, random peak locations.
    #   - energy_peak:  same budget, peak locations from UL energy (proposed).
    #
    # We keep the curve names as uniform / nonuniform / adaptive for the paper
    # legend, but here "uniform" means "adaptive base with no peaks",
    # "nonuniform" means "adaptive base with random peaks", and "adaptive"
    # means "adaptive base with energy peaks".
    strategy_configs = [
        ("uniform",    {"use_adaptive": True, "use_peaks": False, "peak_mode": "energy"}),
        ("nonuniform", {"use_adaptive": True, "use_peaks": True,  "peak_mode": "random"}),
        ("adaptive",   {"use_adaptive": True, "use_peaks": True,  "peak_mode": "energy"}),
    ]
    rows = []
    for ov in ov_list:
        x_label = round(ov * 100, 2)
        per_curve = {name: [] for name, _ in strategy_configs}
        for name, cfg_kwargs in strategy_configs:
            mg = SamplingMaskGenerator(
                num_subcarriers=M,
                use_adaptive=cfg_kwargs["use_adaptive"],
                base_spacing=overhead_to_spacing(ov, M),
                use_nonuniform_peaks=cfg_kwargs["use_peaks"],
                peak_mode=cfg_kwargs["peak_mode"],
                peak_prob=0.0,
                peak_neighbors=int(config.phase2.sampling.peak_neighbors),
                random_offset=False,
            )
            for sd in seeds:
                loader = build_loader(config, transform, h5_path, mg, snr, sd,
                                      args.num_samples, phase2=True)
                per_curve[name].append(
                    eval_model_stage2(model, loader, device, False, 0.0)["nmse_db"])
        rows.extend(_reduce_rows(per_curve, x_label, "nmse_db"))
        print(f"[sampling_overhead] overhead={x_label}% done")
    return rows, "Pilot overhead (%)", "NMSE (dB)"


def _sweep_stage1_snr(args, config, transform, device, h5_path, snr_list, seeds):
    model = DlCsiPredictor(config).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
    rows = []
    for snr in snr_list:
        vals = []
        for sd in seeds:
            loader = build_loader(config, transform, h5_path, None, snr, sd,
                                  args.num_samples, phase2=False)
            vals.append(eval_model_stage1(model, loader, device)["magnitude_nmse_db"])
        rows.extend(_reduce_rows({"proposed_stage1": vals}, snr, "magnitude_nmse_db"))
        print(f"[nmse_snr/stage1] SNR={snr:>4} dB done")
    return rows, "SNR (dB)", "Magnitude NMSE (dB)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _eval_mask(config):
    """Deterministic adaptive-hybrid mask matching evaluate_phase2 defaults."""
    s = config.phase2.sampling
    return make_mask_generator(
        "adaptive", int(config.data.num_subcarriers),
        base_spacing=int(getattr(s, "base_spacing", 8)),
        peak_neighbors=int(getattr(s, "peak_neighbors", 2)),
    )


def _load_stage2(args, config, device):
    stage1_ckpt = str(config.phase2.stage1_checkpoint) if getattr(
        config.phase2, "stage1_checkpoint", None) else None
    model = PhaseRecoveryNetwork(config, stage1_checkpoint_path=stage1_ckpt)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
    return model.to(device)


def _reduce_rows(per_curve, x, metric):
    rows = []
    for curve, vals in per_curve.items():
        t = torch.tensor(vals, dtype=torch.float64)
        mean = float(t.mean().item())
        std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
        rows.append((curve, x, metric, mean, std))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve", "x", "metric", "mean", "std"])
        w.writerows(rows)
    print(f"Wrote {path}")


def plot_rows(rows, xlabel, ylabel, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc}); skipping PNG.")
        return
    curves = {}
    for curve, x, _metric, mean, std in rows:
        curves.setdefault(curve, []).append((x, mean, std))
    plt.figure(figsize=(7, 5))
    for curve, pts in curves.items():
        pts.sort(key=lambda r: r[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=curve)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Wrote {path}")


FIGURES = {
    "nmse_snr": run_nmse_snr,
    "se_snr": run_se_snr,
    "nmse_overhead": run_nmse_overhead,
    "sampling_overhead": run_sampling_overhead,
}


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.project.seed))
    device = torch.device(str(config.project.device)
                          if torch.cuda.is_available() else "cpu")

    transform = AngleDelayTransform(
        use_2d_antenna_dft=bool(config.preprocess.angle_delay.use_2d_antenna_dft),
        apply_window=bool(config.preprocess.angle_delay.apply_window),
        normalize=bool(config.preprocess.normalize_per_sample),
    )
    h5_path = {"train": config.data.h5_train, "val": config.data.h5_val,
               "test": config.data.h5_test}[args.split]

    out_dir = args.output_dir or _sim_cfg(config, "output_dir", "./outputs/sim")
    os.makedirs(out_dir, exist_ok=True)

    rows, xlabel, ylabel = FIGURES[args.figure](
        args, config, transform, device, h5_path)

    csv_path = os.path.join(out_dir, f"results_{args.figure}.csv")
    write_csv(rows, csv_path)
    if not args.no_plot:
        png_path = os.path.join(out_dir, f"fig_{args.figure}.png")
        plot_rows(rows, xlabel, ylabel, args.figure, png_path)


if __name__ == "__main__":
    main()
