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
# From scratch (batch_size=4 default)
python src/train.py --train_dir /path/to/train --val_dir /path/to/val --epochs 500 --learning_rate 3e-4

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

**Inputs** (5 named tensors):
- `sentinel2_sequence`: `(10, 128, 128, 4)` — BAP-filled B02, B03, B04, B8A reflectance
- `landcover_map`: `(128, 128, 10)` — one-hot ESA WorldCover
- `temporal_metadata`: `(10, 3)` — per-frame normalized year, sin/cos DOY
- `weather_sequence`: `(20, 21)` — 7 E-OBS vars × 3 aggregations
- `target_start_doy`: `(1,)` — normalized start DOY for the forecast window

**Encoder**: 3× ConvLSTM2D (32→48→64 filters) with group norm + skip connections. Input is BAP-filled S2 concatenated with spatially-tiled per-frame DOY encoding (7 channels total). Landcover is no longer in the encoder — it was wasting recurrent capacity (16.4M redundant values tiled across 10 timesteps).

**Decoder**:
- `FourierCoefficientHead` (no L2, no coeff_scale, 4 harmonics) predicts harmonic coefficients (a_k, b_k) per pixel per band from `latent_fused`. Convolutions output at full scale — no learnable amplitude gate.
- `FourierSynthesisLayer` synthesises temporal deltas using `keras.ops` (not `tf.*`) for Keras 3 graph tracing.
- Weather FiLM modulation: `harmonics * (1 + gamma)` where `gamma = TimeDistributed(Dense(4, zeros_init))(weather_sequence)`. Scale-only (no shift) so weather can't create signal from nothing. Zero-init starts as identity. Applied before DC offset so weather modulates oscillatory harmonics, not the sustained trend.
- DC offset: separate `Conv2D` from `latent_fused`, zero-initialised, broadcast across time via `Reshape`+`UpSampling3D`. Added after FiLM. Starts silent so harmonics lead.

**Latent fusion**: `latent(64) + skip(48) + context(48) + landcover(10) = 170 channels`. Landcover joins at the decoder stage where it informs per-pixel coefficient prediction without burdening the recurrent encoder.

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

`DeltaRegressionLoss` — 3 components with horizon-weighted temporal masking:
- **Weighted log-cosh regression** on masked (clear-sky only) reflectance deltas — primary objective. Per-band weights `[0.75, 0.75, 1.25, 3.5]` (B02, B03, B04, B8A) correct the slope imbalance (NIR most under-predicted).
- **NDVI reconstruction loss** — penalises `|NDVI(BAP+pred) − NDVI(BAP+true)|` on clear pixels. Computed on full reflectance (BAP available in `y_true[..., 5:9]`). Constrains the NIR/Red inter-band ratio.
- **Temporal smoothness** — 2nd-order Huber on all-band delta curvature. Penalises non-physical acceleration, allows linear trends.

**Horizon weighting**: Both regression and NDVI losses are weighted by a linear temporal ramp from `w_min=0.5` to `w_max=2.0` across the 20 forecast steps, normalised to mean 1.0. Later timesteps (larger deltas, harder predictions) receive ~3.2× the gradient of early timesteps. This prevents the model from "coasting" on easy early predictions while being conservative at long horizons.

Removed: edge penalty, spatial diversity, temporal MAD excess (anti-tint), cross-band correlation, asymmetric under-prediction penalty. These were treating symptoms of the architectural problem (spatially-uniform shortcut paths dominating the Fourier harmonics).

Training defaults: `regression_weight=5.0`, `ndvi_weight=1.5`, `temporal_weight=0.1`, `band_weights=(0.75, 0.75, 1.25, 3.5)`, `horizon_w_min=0.5`, `horizon_w_max=2.0`.

### Training Pipeline (`src/train.py`)

- **Optimizer**: AdamW with `clipnorm=5.0` (was 1.0 — too aggressive for 3-layer ConvLSTM)
- **LR schedule**: Linear warmup (5 epochs) via `WarmupScheduler`, then `ReduceLROnPlateau`
- **Callbacks**: `ModelCheckpoint` (saves per epoch to `checkpoints/`), `EarlyStopping`
- Shuffle buffer of 15 samples; `adapt_inputs` (from `src/adapt_inputs.py`) maps dataset keys to model input names

## Key Design Decisions

- **Delta prediction**: Model predicts changes from BAP baseline, not absolute reflectance. Full reflectance = `BAP + delta`.
- **Fourier harmonics decoder**: Non-monotonic, DOY-conditioned. DC offset term (separate `Conv2D`) enables sustained trend predictions beyond pure oscillatory harmonics.
- **Cloud masking in loss only**: The cloud mask (`y_true[..., 0:1]`) gates the regression loss — clear pixels only. The encoder receives the full BAP-filled sequence without spatial zeroing.
- **`keras.ops` required in custom layers**: Using `tf.*` ops on KerasTensors during Keras 3 model construction breaks graph connectivity (`inputs not connected to outputs`). All `FourierSynthesisLayer` and `FourierCoefficientHead` ops use `keras.ops.*`.
- **No L2 on decoder layers**: L2 on `FourierCoefficientHead` was driving coefficients to zero (mode collapse). AdamW's built-in weight decay is sufficient.
- **Spectral balance loss uses true-reflectance denominator**: `(nir_pred − red_pred) / (nir_true + red_true + ε)`. The denominator has no dependency on model parameters so the gradient only flows through the numerator — no division-by-prediction explosion. This replaced the earlier NDVI ratio `(nir_pred − red_pred) / (nir_pred + red_pred + ε)` which caused NaN gradients when predicted reflectances were near zero.
- **Weather FiLM modulation (not additive residual)**: Previous per-timestep `TimeDistributed(Dense)` additive residual was a spatially-uniform shortcut dominating the Fourier path. Replaced with scale-only FiLM: `harmonics * (1 + gamma)` where gamma is zero-initialised. Scale-only means `gamma * 0 = 0` — weather can modulate existing harmonic signal but can't inject its own. Applied before DC offset. Weather also enters through the global context vector fused into `latent_fused`.
- **Landcover in decoder, not encoder**: Landcover is static — tiling it across 10 ConvLSTM steps wastes 16.4M values per sample of recurrent capacity on unchanging data. Moved to `latent_fused` concat where it directly informs per-pixel coefficient prediction.
- **Per-frame DOY in encoder**: Each ConvLSTM frame now receives its own DOY encoding (year, sin_doy, cos_doy), giving the encoder explicit temporal positioning. Previously only the last timestep's metadata was used (via context Dense), so the encoder had no per-frame season awareness.
- **Shared `adapt_inputs` utility (`src/adapt_inputs.py`)**: Input dict adaptation (mapping dataset keys to model input names) extracted from 6 duplicated copies into a single shared module.
- **No coeff_scale on harmonics**: The learnable amplitude scale (init=0.1) was removed — it had shrunk to 0.066 during training, actively suppressing the harmonic signal. Convolutions now output at full scale.
- **DC offset zero-initialised**: Prevents the DC path from dominating early training while harmonics are still learning. DC was accounting for 22–68% of prediction variance per band with default glorot init.

## Next Steps

- Train from scratch on full 14K dataset with batch_size=4 and verify:
  - Under-prediction improvement (target: mean slope > 0.7, NIR slope > 0.5)
  - Diversity ratios improved (target: > 0.2 for all bands)
  - Horizon weighting effect visible (late-horizon MAE closer to early-horizon)
  - 4th harmonic contributing meaningful amplitude (not noise)

## Changelog

### 2026-02-28
**Horizon-weighted loss, NIR weight increase, batch size, 4th harmonic**

After training on the full 14K dataset (13 epochs), the model shows healthy internals (no overfitting, no pink tint, spatial structure learned, harmonics dominating over DC) but two persistent problems: significant under-prediction (mean slope 0.59, NIR worst at 0.377) and worsened diversity collapse (ratios 0.05–0.20). Root cause: symmetric log-cosh averaged uniformly over timesteps encourages conservative "mean-seeking" predictions, especially at later timesteps where deltas are largest.

**Loss changes (`src/losses.py`, `src/train.py`):**
- **(A) Horizon-weighted temporal loss**: Both regression and NDVI losses are now weighted by a linear ramp from `w_min=0.5` to `w_max=2.0` across the 20 forecast steps, normalised to mean 1.0. The last timestep receives ~3.2× the gradient of the first. This directly penalises the "coast on easy early, hedge on hard late" strategy that was causing under-prediction of large deltas at long horizons. The weighting is applied via `weighted_clear = clear_sq * horizon_w` and normalised by `n_clear_w` to properly account for the interaction with cloud masking.
- **(B) NIR band weight 2.25 → 3.5**: NIR slope was stuck at 0.377 despite 2.25× weight while Blue/Green improved substantially. At 3.5×, NIR gets ~4.7× the gradient of Blue/Green — aggressive but justified by the 2× gap between NIR and Blue slopes.

**Training changes (`src/train.py`):**
- **(C) Batch size 1 → 4**: Reduces gradient noise (14K samples / 4 = 3,500 steps/epoch), enables smoother convergence, and reduces per-epoch wall-clock time by ~3–4×.

**Architecture changes (`src/build_model.py`):**
- **(D) n_harmonics 3 → 4**: With 14K training samples and healthy gradient flow, the model can utilise more spectral components. Complex phenology (double cropping, drought recovery) benefits from the additional harmonic. The Fourier coefficient output grows from 24 to 32 values per pixel (n_bands × 2K).

### 2026-02-27 (b)
**Architecture improvements — fix spatially-uniform prediction dominance**

Three architectural weaknesses compounded the pink tint: weather temporal structure destroyed by flattening `(20,21)→Dense(32)`, landcover wasting recurrent capacity (static map tiled across 10 ConvLSTM steps = 16.4M redundant values), and no temporal positioning in encoder (ConvLSTM had no per-frame DOY awareness).

**Model changes (`src/build_model.py`):**
- **(A) Per-frame DOY encoding in encoder**: `temporal_metadata` shape changed `(3,)→(10,3)`. Full per-frame `[year, sin_doy, cos_doy]` tiled spatially and concatenated with S2 as encoder input (7 channels, was 14). For the context Dense(16), last timestep extracted via `Cropping1D`.
- **(B) Landcover moved from encoder to decoder**: Removed `lc_reshape`/`lc_tile` from encoder input. Landcover now joins at `latent_fused` concat: `64+48+48+10=170 channels` (was 160). Static information informs per-pixel coefficients without burdening the recurrent path.
- **(C) Weather FiLM modulation**: Scale-only FiLM applied to harmonics before DC offset: `harmonics * (1 + gamma)` where `gamma = TimeDistributed(Dense(4, zeros_init))(weather_sequence)`. Zero-init = identity at start. Scale-only = can't create signal from nothing. Replaces the removed additive weather residual with a modulation that preserves spatial variation.

**Code quality (`src/adapt_inputs.py`):**
- **(D) Extract shared `adapt_inputs`**: Duplicated in 6 files → single `src/adapt_inputs.py`. Now passes full `(B,10,3)` temporal metadata (was `[:,-1,:]`). Updated imports in `train.py`, `visualize.py`, `diagnose_signal.py`, `diagnose_tint.py`, `parameter_analysis.py`, `score_earthnet.py`.

Total params: 592,860 (down from 595,036).

### 2026-02-27 (a)
**Architecture & loss simplification — fix pink tint root cause**

Diagnostics revealed the pink tint is architectural: two spatially-uniform paths (DC offset at 22–68% of variance, weather residual at 56–75% of harmonic magnitude) were drowning the only spatially-varying path (Fourier harmonics, suppressed by coeff_scale shrinking from 0.1→0.066). Diversity ratios 0.18–0.38 confirmed predictions were spatially flat. The 7-component loss was treating symptoms of this architectural problem.

**Model changes (`src/build_model.py`):**
- **(A) Remove weather residual**: The `TimeDistributed(Dense)` + `LearnableScale` weather shortcut path removed entirely. Weather context already enters through the global context vector fused into `latent_fused`. The short gradient path (weather→Dense→output) made it learn faster than ConvLSTM→Fourier, creating spatially-uniform predictions.
- **(B) Remove coeff_scale from FourierCoefficientHead**: The learnable amplitude scale (init=0.1) had shrunk to 0.066 during training — actively suppressing the harmonic signal. Convolutions now output at full scale.
- **(C) Reduce n_harmonics 6 → 3**: Harmonics k=4,5,6 had amplitudes 0.013–0.015 (effectively noise). Only k=1 had meaningful signal (0.062). Fewer parameters means the remaining ones can carry more weight.
- **(D) Zero-initialise DC offset**: Default glorot init gave DC immediate signal while harmonics started suppressed. Zero-init lets harmonics lead.

**Loss simplification (`src/losses.py`, `src/train.py`):**
- **(E) Drop 4 loss components**: Removed edge penalty (0.2% of total — negligible), spatial diversity (treating symptom), temporal MAD excess / anti-tint (treating symptom), cross-band correlation (treating symptom).
- **(F) Drop asymmetric under-prediction penalty**: Model was already over-predicting at P50 (0.070 vs 0.039 target) — this penalty was counterproductive.
- **(G) Keep 3 components**: Regression (weighted log-cosh, per-band), NDVI consistency, temporal smoothness. Clean gradient signal without competing objectives.

**Diagnostic script fixes (`scripts/diagnose_signal.py`):**
- **(H) `plot_coefficient_analysis` updated**: Removed references to `coeff_scale` (no longer exists). Now plots kernel weight distribution and per-band kernel spread.

### 2026-02-26 (b)
**Reduce correlation and NDVI loss weights — fix over-prediction and weather shortcut re-emergence**

Training with `corr_weight=1.0` and `ndvi_weight=3.0` made the model worse: NIR slope dropped from 0.655→0.437, predictions flipped from under- to over-prediction (1.43× target magnitude), and the weather residual shortcut re-emerged (7.68× harmonics for Blue). Root cause: correlation loss consumed 45.3% of total loss, making correlation matching the dominant objective. The weather residual (88-param Dense, spatially uniform) was the fastest path to change inter-band correlations, re-enabling the shortcut that zero-init was meant to prevent.

- **(A) `corr_weight` 1.0 → 0.1 (`src/losses.py`, `src/train.py`)**: Reduces correlation from 45% to ~8% of total loss. Correlation matching becomes a gentle guiding signal rather than the primary objective, preventing the weather residual from being the fastest optimisation path.
- **(B) `ndvi_weight` 3.0 → 1.5 (`src/losses.py`, `src/train.py`)**: At 3.0×, NDVI loss combined with correlation was overwhelming regression. At 1.5×, NDVI contributes ~12% — still meaningful pressure on NIR/Red ratio without starving regression of gradient.
- **(C) Diagnostics label update (`scripts/diagnose_tint.py`)**: Weight labels in `plot_loss_components` updated to match new defaults.

Projected loss balance: Regression ~41%, Diversity ~33%, NDVI ~12%, Correlation ~8%, Tint ~6%.

### 2026-02-26 (a)
**Pre-full-training tuning — fixes shortcut learning, diversity collapse, and band imbalance**

Diagnostics on the 1,369-sample model (plots 04, 12, 20, 22, 24) identified three structural problems that more data alone cannot fix:

- **(A) Weather residual shortcut learning fix (`src/build_model.py`)**: The `TimeDistributed(Dense)` weather residual (88 parameters) was converging instantly and contributing 22× the magnitude of Fourier harmonics for Blue/Green (plot 24). This made the model bypass the ConvLSTM→Fourier path entirely. Fix: zero-initialise the Dense kernel + bias, and gate the output through a new `LearnableScale` layer (init=0.05). The weather residual now starts silent and grows only when it genuinely reduces error beyond what harmonics provide.
- **(B) Aggressive band weights for Red/NIR (`src/losses.py`, `src/train.py`)**: Per-band slopes after training were Blue=0.891, Green=0.878, Red=0.766, NIR=0.636 (plot 04). Previous weights `(0.76, 0.87, 0.90, 1.47)` hadn't closed the gap. New weights `(0.75, 0.75, 1.25, 2.25)` give NIR 3× the gradient of Blue/Green. Combined with the diversity fix, the per-band multiplier actually takes effect (predicting near-zero × any weight is still near-zero — diversity must break first).
- **(C) Diversity weight 0.5 → 2.0 (`src/losses.py`, `src/train.py`)**: Diversity was 12.8% of total loss (plot 23) but diversity ratios were still collapsed at 0.16–0.35 (plot 12). The 5.0× regression weight rewards near-zero predictions when the model is uncertain, overwhelming diversity at 0.5. At weight=2.0, the diversity contribution (~40% of regression) is strong enough to break the "predict zero" attractor.

**Cross-band correlation loss + NDVI weight increase — fixes Red/NIR under-prediction and NDVI NSE**

Red (slope 0.779) and NIR (0.655) remain severely under-predicted despite band weights. Root cause: independent per-band Fourier coefficients decorrelate bands (predicted Blue-NIR corr = −0.32 vs target +0.31). This destroys NDVI NSE (mean 0.055).

- **(D) Cross-band temporal correlation loss (`src/losses.py`)**: New component 7 — computes per-pixel Pearson correlation between all 6 band pairs across the temporal axis (cloud-masked), penalises `MSE(corr_pred − corr_true)`. Pixels with <3 clear timesteps excluded. Numerical safety: `eps=1e-3` in sqrt (gradient bounded at 15.8), `clip_by_value(-1, 1)` on correlations. `corr_weight=1.0` (estimated ~0.20 raw loss). Reuses `n_clear_t`, `pred_t_mean_exp`, `true_t_mean_exp` from section 5b.
- **(E) NDVI weight 1.0 → 3.0 (`src/losses.py`, `src/train.py`)**: NDVI NSE is the primary EarthNet evaluation metric but NDVI loss was only 18% of total. At 3.0×, NDVI contributes ~35% — comparable to regression — directly increasing gradient pressure on the NIR/Red ratio.
- **(F) Diagnostics update (`scripts/diagnose_tint.py`)**: `plot_loss_components` updated with correct band weights `(0.75, 0.75, 1.25, 2.25)`, correlation bar added to plot 23, weight labels updated to match new defaults.

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
