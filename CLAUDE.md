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
- **Log-Cosh regression** on masked (clear-sky only) reflectance deltas — primary objective
- **Edge penalty** — preserves spatial structure
- **Asymmetric under-prediction penalty** — scales loss by 1.5× when `|pred| < |true|`

Training defaults: `regression_weight=5.0`, `edge_weight=0.1`.

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
