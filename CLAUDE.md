# DL-CSI-v2: FDD Downlink CSI Prediction with DeepSeek-LLM + LoRA

This repository implements the first stage of the research project **"Deep Learning-based Channel State Information Feedback for Massive MIMO"**:
predicting FDD downlink CSI from uplink CSI using a frozen DeepSeek-R1-Distill-Qwen-1.5B backbone, continuous-value input embeddings, LoRA adapters, and dedicated CSI encoders.

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
│   │   ├── embedding_projection.py    # continuous feature -> LLM hidden space
│   │   ├── regression_head.py         # MLP head to predict DL CSI
│   │   └── dl_csi_predictor.py        # end-to-end model (encoders + LLM + LoRA + head)
│   ├── training/
│   │   ├── losses.py                  # MSE + angle-delay L1 loss
│   │   └── trainer.py                 # generic warmup / LoRA trainer
│   └── utils/
│       ├── logging.py                 # file + TensorBoard logger
│       ├── metrics.py                 # NMSE (dB) and cosine similarity
│       └── baselines.py               # baseline methods
├── scripts/
│   ├── generate_data.py               # generate train/val/test H5 files
│   ├── train_warmup.py                # stage 1: train encoders + head
│   ├── train_lora.py                  # stage 2: LoRA end-to-end fine-tuning
│   ├── train_full.py                  # optional stage 3: full / projection-only finetune
│   ├── evaluate.py                    # test-set evaluation
│   ├── run_baselines.py               # run baseline comparisons
│   └── sanity_check.py                # shape sanity check without real weights/data
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
        Continuous Embedding Projection        -->  [B, 3, H_llm]
        DeepSeek-R1-Distill-Qwen-1.5B (frozen)
        LoRA adapters (Q/K/V/FFN)
        Regression Head (MLP)                  -->  predicted DL CSI (Angle-Delay)
        Inverse FFT                            -->  spatial-frequency DL CSI
```

### 3.1 Key Design Decisions

- **Angle-Delay domain**: DFT along the BS antenna dimension and IFFT along subcarriers using `torch.fft`, so gradients back-propagate through the transform.
- **Continuous embeddings**: the LLM's token embedding is bypassed by feeding `inputs_embeds` directly. A learned `Linear(feature_dim -> llm_hidden_dim)` projects local CSI features into the LLM's semantic space.
- **Frozen LLM + LoRA**: the entire DeepSeek/Qwen backbone is frozen; only LoRA matrices in Q/K/V/FFN are trainable during stage 2.
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

# Install LLM stack
pip install transformers>=5.5.4 peft>=0.19.0 accelerate bitsandbytes

# Install Sionna / TensorFlow stack
pip install tensorflow>=2.18.0 sionna>=2.0.1

# Install remaining utilities
pip install -r requirements.txt
```

### 4.2 Download DeepSeek Weights (Offline)

The project expects weights under `./models/deepseek-1_5b`.

```bash
# Using HuggingFace CLI (recommended)
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir ./models/deepseek-1_5b \
    --local-dir-use-symlinks False

# Or use git-lfs (requires git-lfs installed)
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ./models/deepseek-1_5b
```

Verify the directory contains `config.json`, `model.safetensors`, `tokenizer.json`, etc.

### 4.3 Sanity Check (No Weights/Data Required)

Before downloading the full LLM or generating data, verify that the model shapes are correct with a tiny dummy backbone:

```bash
python scripts/sanity_check.py --config config.yaml --skip-llm
```

If the script prints `Shape check passed.`, the encoder/projection/head dimensions are consistent.

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

Training is split into two stages. Adjust `config.yaml` before running.

### 6.1 Stage 1: Warmup (Encoders + Regression Head)

```bash
python scripts/train_warmup.py --config config.yaml
```

- The LLM and LoRA adapters are frozen.
- Only `CsiEncoder`, `TemporalEncoder`, `EnvironmentEncoder`, `ContinuousEmbeddingProjection`, and `RegressionHead` are trained.
- Best checkpoint is saved to `outputs/checkpoints/best_warmup.pt`.

### 6.2 Stage 2: LoRA End-to-End Fine-Tuning

```bash
python scripts/train_lora.py \
    --config config.yaml \
    --warmup-checkpoint outputs/checkpoints/best_warmup.pt
```

- Base LLM weights stay frozen; LoRA adapters become trainable.
- Local encoders remain trainable so the whole pipeline can co-adapt.
- Best checkpoint is saved to `outputs/checkpoints/best_lora.pt`.

### 6.3 Optional Stage 3: Full Fine-Tuning

Use this only if you have enough GPU memory. Enable in `config.yaml`:

```yaml
training:
  full_finetune:
    enabled: true
    epochs: 10
    lr: 5.0e-6
```

Then run:

```bash
python scripts/train_full.py \
    --config config.yaml \
    --lora-checkpoint outputs/checkpoints/best_lora.pt
```

### 6.4 Resume Training

Both `train_warmup.py` and `train_lora.py` support `--resume <checkpoint.pt>`.

---

## 7. Evaluation

### 7.1 Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint outputs/checkpoints/best_lora.pt \
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
| `model` | encoder dims, LLM path, fusion, regression head |
| `lora` | rank, alpha, target modules, dropout |
| `training` | warmup / LoRA / full-finetune schedules, loss weights, early stopping |
| `evaluation` | test batch size, metrics, baseline list |

Before first run, update at least:

- `data.h5_train/val/test` paths (if you move them),
- `model.llm_path` (should point to `./models/deepseek-1_5b` or your offline weights),
- `model.env_encoder.input_dim` to match `data.large_scale_params` length,
- GPU memory-dependent settings: `training.batch_size`, `lora.r`.

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
3. Add its output to `_prepare_inputs` and increase `model.num_virtual_tokens` if needed.
4. Update `config.yaml` with the new hyperparameters.

### Change the LLM Backbone

1. Download the new weights to a local directory.
2. Update `model.llm_path` and `model.llm_hidden_dim` to match the new model.
3. Update `lora.target_modules` to the attention/FFN projection names of the new architecture.

### Add a New Baseline

1. Implement the function in `src/utils/baselines.py`.
2. Register it in the `BASELINES` dict.
3. Add handling logic in `scripts/run_baselines.py` if it requires special inputs.

---

## 11. Known Limitations / TODO

- The Sionna ray synthesizer assumes far-field ULA array responses. For UPA or dual-polarization arrays, extend `_synthesize_cir_from_rays` with full 2D/3D array responses.
- `modules_to_save` in LoRA config is left empty because a custom regression head is used instead of `lm_head`.
- The current generator does not model UE mobility across slots inside one sample; it resamples fast fading independently per slot.
- Windows multiprocessing with H5 can be fragile; reduce `num_workers` to 0 if DataLoader hangs.

---

## 12. Citation

If you use this code, please cite the underlying datasets and models appropriately (3GPP TR 38.901, Sionna, DeepSeek-R1-Distill-Qwen).

```bibtex
@software{dl_csi_v2,
  title = {DL-CSI-v2: DeepSeek-LLM based FDD Downlink CSI Prediction},
  author = {Sun, Ce},
  year = {2026},
}
```
