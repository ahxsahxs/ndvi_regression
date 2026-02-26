# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

A deep learning framework for vegetation reflectance forecasting using Sentinel-2 satellite imagery, E-OBS weather data, and land cover classification. The model predicts reflectance *deltas* (changes from a Best Available Pixel baseline) over a 100-day horizon using a ConvLSTM encoder with a Fourier harmonics decoder.

**Dataset**: GreenEarthNet (NetCDF files). Default paths are in `src/config.py`.

## Environment Setup

```bash
conda create -n tf-gpu python=3.12 cuda tensorflow xarray earthnet -c conda-forge -c nvidia/label/cuda-12.5.0
```

## Common Commands

### Training
```bash
# From scratch
python src/train.py --train_dir /path/to/train --val_dir /path/to/val --epochs 500 --batch_size 1 --learning_rate 3e-4

# Resume from checkpoint
python src/train.py --resume_from checkpoints/model_at_100.keras --initial_epoch 100 --epochs 500
```

### Analysis & Visualization
```bash
# Generate prediction images (images/predictions/)
python scripts/visualize.py --split val_chopped

# Signal diagnostics (images/diagnostics/)
python scripts/diagnose_signal.py --split val_chopped --max_samples 30

# EarthNet official scoring
python scripts/score_earthnet.py --model final --split val --limit 10

# Run all analysis steps (loads model/data once)
python scripts/run_analysis.py --split val_chopped
python scripts/run_analysis.py --steps 1 2   # select specific steps

# Validate dataset ranges
python scripts/check_dataset.py
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

## Architecture

### Model (`src/build_model.py`)

**Inputs** (5 named tensors — `cloudmask_sequence` was removed when `CloudAwareGatingLayer` was dropped):
- `sentinel2_sequence`: `(10, 128, 128, 4)` — BAP-filled B02, B03, B04, B8A reflectance
- `landcover_map`: `(128, 128, 10)` — one-hot ESA WorldCover
- `temporal_metadata`: `(3,)` — normalized year, sin/cos DOY
- `weather_sequence`: `(20, 21)` — 7 E-OBS vars × 3 aggregations
- `target_start_doy`: `(1,)` — normalized start DOY for the forecast window

**Encoder**: 3× ConvLSTM2D (32→48→64 filters) with group norm + skip connections. Input is the BAP-filled S2 sequence concatenated with tiled landcover — no cloud gating (gating was zeroing BAP-filled clean pixels).

**Decoder**:
- `FourierCoefficientHead` (no L2, `coeff_scale` init=0.1) predicts harmonic coefficients (a_k, b_k) per pixel per band
- `FourierSynthesisLayer` synthesises temporal deltas using `keras.ops` (not `tf.*`) for Keras 3 graph tracing
- DC offset: separate `Conv2D` from `latent_fused`, broadcast across time via `Reshape`+`UpSampling3D`
- Per-timestep weather residual: `TimeDistributed(Dense)` on `weather_sequence`, broadcast spatially

**Output**: `(20, 128, 128, 4)` reflectance deltas

### Data Pipeline (`src/dataset.py`)

`DatasetGenerator` loads NetCDF files and produces `(x_dict, y)` pairs:

- **Input frames**: Every 5th day, starting at day 4, for 10 frames (50-day window)
- **Target frames**: 20 frames starting at day 54 (100-day forecast horizon)
- **BAP composite**: Fills cloudy pixels by iterating backwards through the sequence
- **Normalization**: Global percentile-based (2nd–98th) across bands/timesteps
- **Weather**: 21-day rolling climatology detrended; 3 aggregations per variable (min anomaly, max anomaly, mean climatology)
- **Target structure**: 9 channels packed as `[cloudmask(1), deltas(4), BAP(4)]`; unpacked in loss functions

### Loss Function (`src/losses.py`)

`DeltaRegressionLoss`:
- **Weighted log-cosh regression** on masked (clear-sky only) reflectance deltas — primary objective. Per-band weights `[1.0, 1.0, 1.5, 2.0]` (B02, B03, B04, B8A) correct the slope imbalance (NIR was most under-predicted at slope 0.601).
- **Edge penalty** — preserves spatial structure
- **Asymmetric under-prediction penalty** — scales loss by 1.5× when `|pred| < |true|`
- **NDVI reconstruction loss** — penalises `|NDVI(BAP+pred) − NDVI(BAP+true)|` on clear pixels. Computed on full reflectance (BAP available in `y_true[..., 5:9]`) to avoid division-by-zero on small deltas. Constrains the NIR/Red inter-band ratio and fixes the purple colour bias.
- **Batch diversity loss** — penalises `mean(ReLU(std_true − std_pred))` across the batch axis. Zero when predictions match or exceed target variability; positive when the model collapses toward the mean.

Training defaults: `regression_weight=5.0`, `edge_weight=0.1`, `ndvi_weight=0.5`, `diversity_weight=0.5`, `band_weights=(1.0, 1.0, 1.5, 2.0)`.

### Training Pipeline (`src/train.py`)

- **Optimizer**: AdamW with `clipnorm=5.0` (was 1.0 — too aggressive for 3-layer ConvLSTM)
- **LR schedule**: Linear warmup (5 epochs) via `WarmupScheduler`, then `ReduceLROnPlateau`
- **Callbacks**: `ModelCheckpoint` (saves per epoch to `checkpoints/`), `EarlyStopping`
- Shuffle buffer of 15 samples; input dict adaptation maps dataset keys to model input names

## Key Design Decisions

- **Delta prediction**: Model predicts changes from BAP baseline, not absolute reflectance. Full reflectance = `BAP + delta`.
- **Fourier harmonics decoder**: Non-monotonic, DOY-conditioned. DC offset term (separate `Conv2D`) enables sustained trend predictions beyond pure oscillatory harmonics.
- **Cloud masking in loss only**: The cloud mask (`y_true[..., 0:1]`) gates the regression loss — clear pixels only. The encoder receives the full BAP-filled sequence without spatial zeroing.
- **`keras.ops` required in custom layers**: Using `tf.*` ops on KerasTensors during Keras 3 model construction breaks graph connectivity (`inputs not connected to outputs`). All `FourierSynthesisLayer` and `FourierCoefficientHead` ops use `keras.ops.*`.
- **No L2 on decoder layers**: L2 on `FourierCoefficientHead` was driving coefficients to zero (mode collapse). AdamW's built-in weight decay is sufficient.
- **Spectral balance loss uses true-reflectance denominator**: `(nir_pred − red_pred) / (nir_true + red_true + ε)`. The denominator has no dependency on model parameters so the gradient only flows through the numerator — no division-by-prediction explosion. This replaced the earlier NDVI ratio `(nir_pred − red_pred) / (nir_pred + red_pred + ε)` which caused NaN gradients when predicted reflectances were near zero.
- **Diversity loss uses `sqrt(variance + eps)` not `reduce_std`**: `tf.math.reduce_std` gradient is `(x − mean) / (N · std)`. At batch_size=1, std=0 exactly, so the gradient is `0/0 = NaN`. TF's chain rule then computes `relu'(0) × NaN = NaN`. Using `sqrt(variance + 1e-8)` keeps the denominator finite at all times.

## Changelog

### 2026-02-25
**Loss function overhaul (`src/losses.py`) — fixes colour bias and mode collapse**
- **(A) Per-band regression weights** `[1.0, 1.0, 1.5, 2.0]`: replaced uniform band mean in log-cosh with a normalised weighted sum. NIR (slope 0.601) and Red (slope 0.761) were systematically under-predicted; higher weights increase their gradient pressure. Applied to both regression and edge penalty terms.
- **(B) NDVI reconstruction loss** (`ndvi_weight=0.5`): adds `mean(|NDVI_pred − NDVI_true|)` on clear-sky pixels, where NDVI is computed from full reflectance `BAP + delta`. Directly constrains the NIR/Red inter-band ratio and addresses the purple colour tint observed in delta prediction images.
- **(C) Batch diversity loss** (`diversity_weight=0.5`): adds `mean(ReLU(std_true − std_pred))` across the batch dimension. Fires only when cross-sample prediction variance collapses below target variance (diversity ratios were 0.16–0.32 across bands). Inactive at batch_size=1.

**Loss function bug fixes (`src/losses.py`) — NaN during training**
- **Root cause 1 — `tf.math.reduce_std` NaN gradient**: `reduce_std` computes `sqrt(variance)` whose backward pass is `(x − mean) / (N · std)`. At batch_size=1 (the default), `std = 0` exactly and the gradient is `0/0 = NaN`. TF's chain rule then evaluates `relu'(0) × NaN = 0 × NaN = NaN` (IEEE 754), corrupting all gradients. Fixed by replacing with `sqrt(reduce_variance + 1e-8)`.
- **Root cause 2 — NDVI denominator gradient explosion**: `(nir_pred − red_pred) / (nir_pred + red_pred + ε)` has gradient `∂/∂nir_pred = 2·red_pred / (sum + ε)²`. With `ε=1e-6` and both bands near zero (common early in training), this reached ~500/pixel, causing NaN in the optimizer's internal accumulators before `clipnorm` could act. Fixed by using true-reflectance as denominator: `denom = nir_true + red_true + 1e-3` — zero model dependency, gradient bounded by `1/denom`.
- **Root cause 3 — `log(cosh(x))` overflow (latent)**: `tf.math.cosh` hits float32-inf at `|x| > 88.7`, then `inf * 0` (cloudy mask) = NaN. Fixed with the stable identity `|x| + softplus(−2|x|) − log(2)`, which is finite for any finite input.
- **Added `y_pred` NaN guard**: `tf.where(is_finite(y_pred), y_pred, 0)` at the top of `call()` prevents a single bad batch from corrupting optimizer state.

**Analysis scripts**
- `scripts/run_analysis.py`: added steps 3 (Signal Diagnostics) and 4 (Parameter Analysis) to the shared-resource runner. Scripts now expose callable entry points (`diagnose()`, `analyze_parameters()`) so the model and dataset are loaded only once.
- `scripts/parameter_analysis.py`: `plot_cloud_cover_analysis` and `write_collapse_summary` updated to handle missing `cloudmask_sequence` input (removed with `CloudAwareGatingLayer`); input cloud panels replaced with an N/A annotation.
