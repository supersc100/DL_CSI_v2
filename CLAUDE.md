# DL-CSI-v2: FDD Downlink CSI Prediction with CNN + Transformer

This repository implements the first stage of the research project **"Deep Learning-based Channel State Information Feedback for Massive MIMO"**:
predicting FDD downlink CSI from uplink CSI using dedicated CSI encoders, a lightweight Transformer fusion block, and a regression head. All components are trained end-to-end in a single stage.

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
│   │   └── dl_csi_predictor.py        # end-to-end model (encoders + fusion + head)
│   ├── training/
│   │   ├── losses.py                  # MSE + angle-delay L1 loss
│   │   └── trainer.py                 # single-stage trainer
│   └── utils/
│       ├── logging.py                 # file + TensorBoard logger
│       ├── metrics.py                 # NMSE (dB) and cosine similarity
│       └── baselines.py               # baseline methods
├── scripts/
│   ├── generate_data.py               # generate train/val/test H5 files
│   ├── train.py                       # single-stage end-to-end training
│   ├── train_smoke_test.py            # fast smoke test without real data
│   ├── evaluate.py                    # test-set evaluation
│   ├── run_baselines.py               # run baseline comparisons
│   └── sanity_check.py                # shape sanity check without real data
└── tests/
    └── test_transforms.py             # unit tests for FFT transforms
```

---

## 3. High-Level Architecture

```text
Input
├── current UL CSI (spatial-frequency)
│   └── 2D-DFT (antenna) + IFFT (subcarrier)  -->  Angle-Delay domain
│       └── CsiEncoder (3D CNN + LN + GELU)    -->  feature token [D]
├── historical UL/DL pairs (T=16 slots)
│   └── per-slot CNN + Lightweight Transformer -->  temporal token [D]
└── large-scale parameters (6-10 dims)
    └── Environment Encoder (MLP + SiLU)       -->  env token [D]

        Cross-attention fusion / concat        -->  [B, 3, D]
        TransformerFusion (small Transformer)  -->  [B, 3, H]
        Mean pool                              -->  [B, H]
        Regression Head (MLP)                  -->  predicted DL CSI (Angle-Delay)
        Inverse FFT                            -->  spatial-frequency DL CSI
```

### 3.1 Key Design Decisions

- **Angle-Delay domain**: DFT along the BS antenna dimension and IFFT along subcarriers using `torch.fft`, so gradients back-propagate through the transform.
- **CNN encoders**: 3D convolutions extract local structure from angle-delay CSI tensors; a lightweight Transformer aggregates historical slots.
- **Transformer fusion**: a small domain-specific Transformer (not a pre-trained LLM) models interactions among the three modality tokens `[current_ul, temporal, env]`.
- **End-to-end single-stage training**: all components are trainable from the start; no staged freezing or adapter tuning is required.
- **Loss**: complex MSE in angle-delay domain plus an L1 magnitude consistency loss to preserve large-scale structure.
- **Numerics**: model forward uses `bfloat16`; loss computation is always `float32`.

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

Training is single-stage and end-to-end. Adjust `config.yaml` before running.

> **Tip:** Before starting real training, run the fast smoke test in [4.3](#43-training-smoke-test-no-data-required) to verify the full training loop without generating data.

### 6.1 End-to-End Training

```bash
python scripts/train.py --config config.yaml
```

- All model components are trainable.
- Best checkpoint is saved to `outputs/checkpoints/best.pt`.
- Periodic checkpoints are saved to `outputs/checkpoints/epoch{epoch}.pt`.

### 6.2 Resume Training

`train.py` supports `--resume <checkpoint.pt>`:

```bash
python scripts/train.py --config config.yaml --resume outputs/checkpoints/epoch5.pt
```

---

## 7. Evaluation

### 7.1 Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best.pt \
    --split test
```

Outputs:

- `nmse_db`: normalized MSE in dB (lower is better).
- `cosine_similarity`: cosine similarity between flattened complex vectors (higher is better).

### 7.2 Run Baselines

```bash
python scripts/run_baselines.py --config config.yaml --split test
```

Runs all five baselines:

1. `copy_ul` — predict DL = current UL (lower bound).
2. `angle_delay_interp` — linear interpolation of historical DL in angle-delay domain.
3. `tdd_oracle` — perfect TDD reciprocity (upper bound).
4. `no_large_scale` — UL only ablation.
5. `no_history` — UL only ablation (conceptually no history).

To run only selected baselines:

```bash
python scripts/run_baselines.py --config config.yaml --split test \
    --baselines copy_ul angle_delay_interp tdd_oracle
```

---

## 8. Configuration Reference

All hyperparameters live in `config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `project` | seed, device, mixed precision, output directories |
| `data` | scenario, carrier frequencies, antenna arrays, subcarriers, samples |
| `preprocess` | normalization, angle-delay transform options |
| `model` | encoder dims, fusion, Transformer fusion, regression head |
| `training` | epochs, optimizer, scheduler, loss weights, early stopping |
| `evaluation` | test batch size, metrics, baseline list |

Before first run, update at least:

- `data.h5_train/val/test` paths (if you move them),
- `model.env_encoder.input_dim` to match `data.large_scale_params` length,
- GPU memory-dependent settings: `training.batch_size`, `model.regression_head.hidden_dim`.

---

## 9. Reproducibility Notes

- All random seeds are controlled via `project.seed`.
- Sionna/TensorFlow uses its own random state; split-specific `seed_offset`s are applied in `generate_data.py`.
- Mixed precision (`bfloat16`) requires Ampere-or-newer NVIDIA GPU. Disable by setting `project.mixed_precision: false`.
- Data loaders use `persistent_workers=True` when `num_workers > 0`. Set `num_workers: 0` if you encounter multi-process H5 issues on Windows.

---

## 10. Extending the Project

### Add a New Encoder

1. Implement the module in `src/models/`.
2. Import and instantiate it in `src/models/dl_csi_predictor.py`.
3. Add its output to `_prepare_inputs` and update `TransformerFusion.num_tokens` if needed.
4. Update `config.yaml` with the new hyperparameters.

### Replace the Fusion Block

1. Implement the new fusion module in `src/models/` (e.g., a larger Transformer or a Perceiver-style block).
2. Swap `TransformerFusion` for your module inside `DlCsiPredictor`.
3. Update the `model.transformer_fusion` config section accordingly.

### Add a New Baseline

1. Implement the function in `src/utils/baselines.py`.
2. Register it in the `BASELINES` dict.
3. Add handling logic in `scripts/run_baselines.py` if it requires special inputs.

---

## 11. Known Limitations / TODO

- The Sionna ray synthesizer assumes far-field ULA array responses. For UPA or dual-polarization arrays, extend `_synthesize_cir_from_rays` with full 2D/3D array responses.
- The current generator does not model UE mobility across slots inside one sample; it resamples fast fading independently per slot.
- Windows multiprocessing with H5 can be fragile; reduce `num_workers` to 0 if DataLoader hangs.

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
