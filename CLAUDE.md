# DL-CSI-v2: FDD Downlink CSI Prediction with CNN + Transformer

This repository implements the research project **"Deep Learning-based Channel State Information Feedback for Massive MIMO"** in two stages:

1. **Stage 1**: predict the downlink CSI **magnitude** from uplink CSI (optionally plus historical UL/DL CSI pairs and large-scale parameters) using dedicated CSI encoders, a lightweight Transformer fusion block, and a regression head.
2. **Stage 2**: recover the downlink CSI **phase** from a small number of downlink subband pilot observations, using the frozen Stage 1 magnitude predictor as a prior. The final output is the full complex downlink CSI:

```
H_pred = |H_stage1| ⊙ exp(j · θ_phase2)
```

Both stages use a CNN + Transformer architecture (no pre-trained LLM) and are trained end-to-end in PyTorch.

---

## 1. Research Goal

In FDD massive MIMO, uplink and downlink channels do **not** share the strong reciprocity of TDD, but they still share the same physical propagation environment (large-scale parameters such as path delays, angles, and cluster powers). This project learns a statistical mapping from:

- current uplink CSI,
- historical UL/DL CSI pairs,
- and explicit large-scale parameters

to the downlink CSI in the **Angle-Delay domain**, where UL/DL reciprocity is stronger.

---

## 2. Project Structure

```text
DL_CSI_v2/
├── CLAUDE.md                          # this file
├── config.yaml                        # full hyper-parameter template
├── requirements.txt                   # Python dependencies
├── src/
│   ├── config.py                      # YAML loader / DotDict wrapper
│   ├── data/
│   │   ├── sionna_generator.py        # Sionna 2.x FDD data generator
│   │   ├── dataset.py                 # PyTorch H5 Dataset + DataLoader
│   │   └── transforms.py              # spatial <-> angle-delay FFT transforms
│   ├── models/
│   │   ├── csi_encoder.py             # CNN encoder for current UL CSI
│   │   ├── temporal_encoder.py        # Transformer over historical UL/DL pairs
│   │   ├── env_encoder.py             # MLP for large-scale parameters
│   │   ├── transformer_fusion.py      # lightweight Transformer over modality tokens
│   │   ├── regression_head.py         # MLP head to predict DL CSI
│   │   ├── dl_csi_predictor.py        # end-to-end Stage 1 model
│   │   ├── sampling_mask.py           # adaptive hybrid subcarrier sampling mask
│   │   ├── quantization.py            # scalar quantizer for limited feedback
│   │   ├── freq_sparse_encoder.py     # CNN encoder for sparse DL subbands
│   │   ├── magnitude_encoder.py       # CNN encoder for Stage 1 magnitude prior
│   │   ├── magnitude_guided_attention.py  # cross-attention between sparse features and magnitude
│   │   ├── frequency_interpolator.py  # Transformer over subcarrier dimension
│   │   ├── phase_head.py              # cos/sin phase prediction head
│   │   └── phase_recovery_network.py  # Stage 2 model
│   ├── training/
│   │   ├── losses.py                  # Stage 1 loss
│   │   ├── trainer.py                 # Stage 1 single-stage trainer
│   │   ├── phase2_losses.py           # Stage 2 combined loss
│   │   └── phase2_trainer.py          # Stage 2 multi-stage trainer (warmup/main/QAT)
│   └── utils/
│       ├── logging.py                 # file + TensorBoard logger
│       ├── metrics.py                 # NMSE / cosine / phase NMSE
│       ├── baselines.py               # baseline methods for both stages
│       └── seed.py                    # random seed utilities
├── scripts/
│   ├── generate_data.py               # generate train/val/test H5 files
│   ├── train.py                       # Stage 1 end-to-end training
│   ├── train_smoke_test.py            # fast smoke test without real data
│   ├── evaluate.py                    # Stage 1 test-set evaluation
│   ├── run_baselines.py               # Stage 1 baseline comparisons
│   ├── sanity_check.py                # Stage 1 shape sanity check
│   ├── train_phase2.py                # Stage 2 training
│   ├── evaluate_phase2.py             # Stage 2 evaluation + baselines
│   └── sanity_check_phase2.py         # Stage 2 shape sanity check
└── tests/
    ├── test_transforms.py             # unit tests for FFT transforms
    └── test_phase2.py                 # unit tests for Phase 2 components
```

---

## 3. High-Level Architecture

### 3.1 Stage 1: Magnitude Prediction

```text
Input
├── current UL CSI (spatial-frequency)
│   └── 2D-DFT (antenna) + IFFT (subcarrier)  -->  Angle-Delay domain
│       └── CsiEncoder (3D CNN + LN + GELU)    -->  feature token [D]
├── historical UL/DL pairs (T=16 slots)
│   └── per-slot CNN + Lightweight Transformer -->  temporal token [D]
└── large-scale parameters (6-10 dims)
    └── Environment Encoder (MLP + SiLU)       -->  env token [D]

        Cross-attention fusion / concat        -->  [B, num_tokens, D]
        TransformerFusion (small Transformer)  -->  [B, num_tokens, H]
        Mean pool                              -->  [B, H]
        Regression Head (MLP)                  -->  predicted DL CSI (Angle-Delay)
        Inverse FFT                            -->  spatial-frequency DL CSI
```

### 3.2 Stage 2: Phase Recovery

Stage 2 is a separate `PhaseRecoveryNetwork` that loads the Stage 1 checkpoint and freezes all Stage 1 parameters. It takes sparse downlink subband observations and recovers the full-band phase:

```text
Stage 1 (frozen)                Sparse DL subbands (zero-padded grid)
       │                                      │
       ▼                                      ▼
  predicted |H_DL|                     FreqSparseEncoder
       │                                      │
       ▼                                      ▼
  MagnitudeEncoder                      sparse features [B, N_s, M, D]
       │                                      │
       └── MagnitudeGuidedAttention ──────────┘
                          │
                          ▼
              FrequencyInterpolator (Transformer over subcarriers)
                          │
                          ▼
                   PhaseHead (cos/sin)
                          │
                          ▼
                   exp(j · θ_phase2)
                          │
       ┌──────────────────┘
       ▼
  H_pred = |H_stage1| ⊙ exp(j · θ_phase2)
```

### 3.3 Key Design Decisions

- **Angle-Delay domain**: DFT along the BS antenna dimension and IFFT along subcarriers using `torch.fft`, so gradients back-propagate through the transform.
- **CNN encoders**: 3D convolutions extract local structure from angle-delay CSI tensors; a lightweight Transformer aggregates historical slots.
- **Transformer fusion**: a small domain-specific Transformer (not a pre-trained LLM) models interactions among the modality tokens `[current_ul, temporal, env]`.
- **End-to-end single-stage training (Stage 1)**: all components are trainable from the start; no staged freezing or adapter tuning is required.
- **Stage 1 loss**: the regression head still predicts complex-valued DL CSI in the angle-delay domain, but the training objective focuses on **magnitude MSE**.  In FDD, UL and DL share large-scale geometry (angles/delays/path powers) yet have independent small-scale phases, so predicting the exact complex channel is infeasible; the magnitude/angle-delay power spectrum is the learnable and useful part.  A small complex-MSE term and an angle-delay L1 consistency term are kept as optional auxiliaries.
- **Stage 2 decoupling**: magnitude and phase are recovered by two separate networks. The magnitude predictor is frozen and only its output magnitude is reused; the phase network consumes sparse downlink subband observations, a sampling mask, the Stage 1 magnitude prior, and the current UL CSI for energy-guided sampling.
- **Magnitude-guided attention**: sparse subband features query against Stage 1 magnitude features so that strong propagation paths receive higher weight during phase interpolation.
- **Phase as cos/sin**: the phase head predicts cos and sin separately and L2-normalizes them to unit magnitude, avoiding `2π` wrapping and `atan2` discontinuities.
- **Quantization-aware training (QAT)**: during the QAT stage, a random fraction (`qat_ratio`) of input batches is passed through a scalar quantizer/dequantizer to simulate limited-feedback noise while still exposing the model to ideal floating-point data.
- **Numerics**: model forward uses `bfloat16`; loss computation is always `float32`. Phase loss uses a stable `1 - cos(Δθ)` metric instead of `asin` to avoid infinite gradients at the float32 boundary.

### 3.4 Why magnitude prediction?

The Sionna data generator shares ray geometry between UL and DL but samples independent small-scale gains for the two links.  After per-sample angle-delay normalization, the optimal complex-MSE predictor is essentially zero (NMSE ≈ 0 dB).  The magnitude spectra, however, remain strongly correlated through the shared path powers.  Training on magnitude MSE therefore yields a usable predictor for angle-delay power profiles, which supports scheduling, beam selection, and feedback-compression tasks.  Full complex prediction requires either (a) correlated UL/DL small-scale fading in the data generator, or (b) a phase-residual feedback branch.  Stage 2 implements option (b) by recovering phase from a small number of downlink subband observations.

---

## 4. Environment Setup

### 4.1 Install Dependencies

```bash
# Create a Python 3.11+ environment
conda create -n dl_csi python=3.11 -y
conda activate dl_csi

# Install PyTorch with CUDA 12.4 (adjust index-url to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Sionna / TensorFlow stack
pip install tensorflow>=2.18.0 sionna>=2.0.1

# Install remaining utilities
pip install -r requirements.txt
```

### 4.2 Sanity Check (No Data Required)

Before generating data, verify that the model shapes are correct with dummy inputs:

```bash
python scripts/sanity_check.py --config config.yaml
```

If the script prints `Shape check passed.`, the encoder/fusion/head dimensions are consistent.

### 4.3 Training Smoke Test (No Data Required)

Before generating real Sionna data, run a fast end-to-end training smoke test that uses in-memory dummy data and shrunk model dims:

```bash
python scripts/train_smoke_test.py --config config.yaml
```

This script:

- creates small in-memory dummy CSI tensors,
- shrinks model dims (`feature_dim=128`, `hidden_dim=128`, smaller CNNs),
- runs 2 epochs of training on the CPU,
- verifies that the full loop (data → model → loss → backward → optimizer → checkpoint) completes without NaN/inf.

If it prints `Training smoke test passed.`, the training pipeline is ready. You can then proceed to generate real data.

You can adjust the smoke-test scale with:

```bash
python scripts/train_smoke_test.py --config config.yaml --samples 32 --batch-size 8 --epochs 3
```

> **Note:** Passing the smoke test does not guarantee convergence on real data; it only verifies that the code runs end-to-end.

### 4.4 Phase 2 Sanity & Smoke Checks

After Stage 1 is trained, verify Stage 2 shapes and the frozen Stage 1 behavior:

```bash
python scripts/sanity_check_phase2.py --config config.yaml
```

For a fast CPU smoke test of the full Stage 2 three-stage loop:

```bash
python scripts/train_phase2.py --config config.yaml --smoke-test --num-samples 128 --epochs 2
```

If both print success messages and no NaN metrics appear, the Stage 2 pipeline is ready.

---

## 5. Data Generation

### 5.1 Generate Standard FDD Splits

```bash
python scripts/generate_data.py --config config.yaml --split train val test
```

This creates:

- `data/processed/train.h5`
- `data/processed/val.h5`
- `data/processed/test.h5`

Each file contains:

| Key | Shape | Description |
|---|---|---|
| `h_ul` | [N, N_tx_bs, N_rx_ue, M] | current uplink CSI |
| `h_dl` | [N, N_tx_bs, N_rx_ue, M] | current downlink CSI (target) |
| `history_ul` | [N, T, N_tx_bs, N_rx_ue, M] | past T UL snapshots |
| `history_dl` | [N, T, N_tx_bs, N_rx_ue, M] | past T DL snapshots |
| `large_scale` | [N, 6] | large-scale parameter vector |
| `tau/aoa/aod/powers` | variable | raw ray geometry (optional, debug) |

Stage 2 reuses the same full-band H5 files. Sparse downlink subbands and binary sampling masks are generated on-the-fly in `FddCsiDataset` (`src/data/dataset.py`), so no separate subband H5 is required.

### 5.2 Generate TDD Oracle Split (Performance Upper Bound)

```bash
python scripts/generate_data.py --config config.yaml --split test --tdd-oracle-split test
```

This writes `data/processed/test_tdd_oracle.h5`, where UL and DL share identical fast fading.

### 5.3 Notes on Sionna Integration

- Sionna is TensorFlow-based; the generator isolates TF/Sionna imports inside `src/data/sionna_generator.py`.
- Ray parameters (`_aod`, `_aoa`, `_powers`, `_tau`) are read from the CDL object after initialization; if a Sionna version moves these attributes, adjust `_extract_ray_params`.
- The generator manually synthesizes CIRs from extracted rays with independent small-scale gains for UL/DL, so the two channels share large-scale geometry but not fast fading.

---

## 6. Training

### 6.1 Stage 1: End-to-End Magnitude Training

Training is single-stage and end-to-end. Adjust `config.yaml` before running.

> **Tip:** Before starting real training, run the fast smoke test in [4.3](#43-training-smoke-test-no-data-required) to verify the full training loop without generating data.

```bash
python scripts/train.py --config config.yaml
```

- All model components are trainable.
- Best checkpoint is saved to `outputs/checkpoints/best.pt`.
- Periodic checkpoints are saved to `outputs/checkpoints/epoch{epoch}.pt`.
- `train.py` supports `--smoke-test`, `--num-samples`, and `--epochs` for quick experiments.

Resume training:

```bash
python scripts/train.py --config config.yaml --resume outputs/checkpoints/epoch5.pt
```

### 6.2 Stage 2: Phase Recovery Training

Stage 2 loads the Stage 1 checkpoint from `phase2.stage1_checkpoint` and freezes all Stage 1 parameters. It then trains the phase-recovery modules through three sequential stages:

1. **Warmup**: freeze the frequency interpolator, train only the sparse encoder, magnitude encoder/attention, and phase head.
2. **Main training**: train all Stage 2 modules while Stage 1 stays frozen.
3. **QAT**: same as main training, but a configurable fraction of batches is quantized/dequantized to simulate limited feedback.

```bash
python scripts/train_phase2.py --config config.yaml
```

- Best checkpoints per stage are saved to `outputs/checkpoints/phase2_{stage}_best.pt`.
- Periodic checkpoints per stage are saved to `outputs/checkpoints/phase2_{stage}_epoch{epoch}.pt`.
- For a quick CPU smoke test: `python scripts/train_phase2.py --config config.yaml --smoke-test --num-samples 128 --epochs 2`.

### 6.3 Resume Stage 2

```bash
python scripts/train_phase2.py --config config.yaml --resume outputs/checkpoints/phase2_main_epoch10.pt
```

---

## 7. Evaluation

### 7.1 Stage 1 Evaluation

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best.pt \
    --split test
```

Outputs:

- `nmse_db`: normalized MSE in dB (lower is better).
- `cosine_similarity`: cosine similarity between flattened complex vectors (higher is better).

### 7.2 Stage 1 Baselines

```bash
python scripts/run_baselines.py --config config.yaml --split test
```

Runs all Stage 1 baselines:

1. `copy_ul` — predict DL = current UL (lower bound).
2. `angle_delay_interp` — linear interpolation of historical DL in angle-delay domain.
3. `tdd_oracle` — perfect TDD reciprocity (upper bound).
4. `no_large_scale` — UL only ablation.
5. `no_history` — UL only ablation (conceptually no history).

### 7.3 Stage 2 Evaluation

```bash
python scripts/evaluate_phase2.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/phase2_main_best.pt \
    --split test
```

Outputs:

- `nmse_db`: full complex CSI NMSE.
- `phase_nmse_db`: phase-only NMSE.
- `magnitude_nmse_db`: magnitude NMSE (should match Stage 1).
- `cosine_similarity`: complex-vector cosine similarity.

Add `--quantized` to evaluate with 16-bit scalar quantization on the sparse subband input:

```bash
python scripts/evaluate_phase2.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/phase2_qat_best.pt \
    --split test \
    --quantized
```

### 7.4 Stage 2 Baselines

`evaluate_phase2.py` also runs Stage 2 baselines configured under `phase2.evaluation.baselines`:

1. `magnitude_only` — use Stage 1 predicted magnitude, phase = 0.
2. `linear_interp` — linear interpolation of sparse subbands along subcarrier dimension.
3. `dft_interp` — delay-domain zero-padding interpolation.
4. `tdd_oracle` — perfect TDD reciprocity.

Planned ablations (`no_magnitude`, `no_ul_guidance`) are supported by constructing the corresponding `PhaseRecoveryNetwork` variant and comparing checkpoints.

### 7.5 Paper Simulation Curves (SNR / overhead sweeps)

`scripts/run_simulation.py` produces the figures in `researchPlan/simPlan.md` by injecting
AWGN and switching pilot overhead / sampling strategy **at evaluation time**, reusing trained
checkpoints (no regeneration / retraining for the SNR and overhead sweeps). See
`researchPlan/simRun.md` for exact commands.

```bash
# NMSE vs SNR (1.1), with 16-bit quantized-pilot curve
python scripts/run_simulation.py --figure nmse_snr \
    --checkpoint outputs/checkpoints/phase2_main_best.pt --quant-curve
# NMSE vs pilot overhead (1.2)
python scripts/run_simulation.py --figure nmse_overhead \
    --checkpoint outputs/checkpoints/phase2_main_best.pt
# Spectral efficiency vs SNR (1.3)
python scripts/run_simulation.py --figure se_snr \
    --checkpoint outputs/checkpoints/phase2_main_best.pt
# Sampling-strategy ablation vs overhead (2.1 fig 4c)
python scripts/run_simulation.py --figure sampling_overhead \
    --checkpoint outputs/checkpoints/phase2_main_best.pt
```

Each figure writes `outputs/sim/results_<figure>.csv` and `fig_<figure>.png`. Sweep grids,
fixed SNR, seeds (error bars), and output dir live in the `simulation:` config section.

- **Noise injection**: `src/utils/channel_noise.add_awgn`, wired into `FddCsiDataset` via the
  `snr_db` / `noise_base_seed` attributes (set by `build_dataloader(..., snr_db=...)`). Noise is
  added to the UL input and the DL sparse pilots; the supervision target stays clean.
- **Spectral efficiency**: `src/utils/metrics.spectral_efficiency` (single-stream eigen-beamforming
  on per-sample-normalized channels, plus a perfect-CSI upper bound).
- **Sampling strategies / overhead**: `src/models/sampling_mask.make_mask_generator`
  (`uniform` / `nonuniform` / `adaptive`) and `overhead_to_spacing`.
  The `sampling_overhead` figure uses a fair ablation: all three curves share the
  same delay-spread-adaptive base grid and differ only in the peak encryption
  strategy (none / random / energy-guided).
- **`use_magnitude` ablation (fig 4a)**: `phase2.model.use_magnitude: false` drops the
  Stage1-magnitude branch in `PhaseRecoveryNetwork`; requires training a separate checkpoint.
- **Config must match the checkpoint**: model dims in `config.yaml` must equal those used at
  training time, or checkpoint loading raises a size mismatch.

---

## 8. Configuration Reference

All hyperparameters live in `config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `project` | seed, device, mixed precision, output directories |
| `data` | scenario, carrier frequencies, antenna arrays, subcarriers, samples |
| `preprocess` | normalization, angle-delay transform options |
| `model` | Stage 1 encoder dims, fusion, Transformer fusion, regression head |
| `training` | Stage 1 epochs, optimizer, scheduler, loss weights, early stopping |
| `evaluation` | Stage 1 test batch size, metrics, baseline list |
| `phase2` | Stage 2 checkpoint, model, sampling, quantization, training stages, baselines |

Before first run, update at least:

- `data.h5_train/val/test` paths (if you move them),
- `model.env_encoder.input_dim` to match `data.large_scale_params` length,
- `phase2.stage1_checkpoint` to point to a trained Stage 1 checkpoint,
- GPU memory-dependent settings: `training.batch_size`, `phase2.model.*` dims.

### 8.1 Phase 2 Configuration Notes

- `phase2.model.feat_dim` controls the size of all Stage 2 modules; it is independent of `model.feature_dim`.
- `phase2.sampling.*` controls the adaptive hybrid subcarrier mask (base spacing, peak encryption, training randomization).
- `phase2.quantization.num_bits` sets the per-real-value bit width (default 16 → 32 bits per complex sample).
- `phase2.training.{warmup,main,qat}` each have their own `epochs`, `lr`, `loss` weights, and early-stopping config.

---

## 9. Reproducibility Notes

- All random seeds are controlled via `project.seed`.
- Sionna/TensorFlow uses its own random state; split-specific `seed_offset`s are applied in `generate_data.py`.
- Mixed precision (`bfloat16`) requires Ampere-or-newer NVIDIA GPU. Disable by setting `project.mixed_precision: false`.
- Data loaders use `persistent_workers=True` when `num_workers > 0`. Set `num_workers: 0` if you encounter multi-process H5 issues on Windows.

---

## 10. Extending the Project

### Add a New Stage 1 Encoder

1. Implement the module in `src/models/`.
2. Import and instantiate it in `src/models/dl_csi_predictor.py`.
3. Add its output to `_prepare_inputs` and update `TransformerFusion.num_tokens` if needed.
4. Update `config.yaml` with the new hyperparameters.

### Replace the Fusion Block

1. Implement the new fusion module in `src/models/` (e.g., a larger Transformer or a Perceiver-style block).
2. Swap `TransformerFusion` for your module inside `DlCsiPredictor`.
3. Update the `model.transformer_fusion` config section accordingly.

### Add a New Stage 1 Baseline

1. Implement the function in `src/utils/baselines.py`.
2. Register it in the `BASELINES` dict.
3. Add handling logic in `scripts/run_baselines.py` if it requires special inputs.

### Extend Stage 2

1. **Add a new Stage 2 module**: implement it in `src/models/`, import in `src/models/phase_recovery_network.py`, and wire it into the forward pass.
2. **Change the sampling strategy**: modify `SamplingMaskGenerator` in `src/models/sampling_mask.py`.
3. **Add a new Stage 2 loss term**: extend `Phase2Loss` in `src/training/phase2_losses.py`.
4. **Add a new Stage 2 baseline**: implement the function in `src/utils/baselines.py` and add its dispatch in `scripts/evaluate_phase2.py`.

---

## 11. Known Limitations / TODO

- The Sionna ray synthesizer assumes far-field ULA array responses. For UPA or dual-polarization arrays, extend `_synthesize_cir_from_rays` with full 2D/3D array responses.
- The current generator does not model UE mobility across slots inside one sample; it resamples fast fading independently per slot.
- Windows multiprocessing with H5 can be fragile; reduce `num_workers` to 0 if DataLoader hangs.
- Stage 2 structural-prior fusion from UL CSI (`use_ul_guidance`) is left as an optional extension; the interface is reserved in `config.yaml` but not implemented.
- Stage 2 ablation baselines (`no_magnitude`, `no_ul_guidance`) require running separate model variants. The `no_magnitude` variant is wired via `phase2.model.use_magnitude: false` (train a separate checkpoint); `no_ul_guidance` remains a reserved interface.
- Iso-overhead fairness in the sampling-strategy comparison (`sampling_overhead`): peak encryption *adds* a fixed neighbor budget, so `nonuniform`/`adaptive` sample more points than `uniform` at the same nominal overhead. A budget-constrained (peak-replacing) mask would be needed for a strict equal-overhead comparison.

---

## 12. Citation

If you use this code, please cite the underlying datasets and models appropriately (3GPP TR 38.901, Sionna).

```bibtex
@software{dl_csi_v2,
  title = {DL-CSI-v2: CNN+Transformer based FDD Downlink CSI Prediction},
  author = {Sun, Ce},
  year = {2026},
}
```
