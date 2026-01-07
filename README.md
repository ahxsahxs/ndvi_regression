# Satellite NDVI Forecasting with Parametric ConvLSTM

A deep learning framework for predicting vegetation dynamics using multi-spectral Sentinel-2 imagery, weather data, and a parametric curve decoder.

## Overview

This project implements a spatiotemporal encoder-decoder model that forecasts vegetation changes over a 60-day horizon using:
- **Historical Sentinel-2 observations** (18 frames over 90 days)
- **E-OBS climate variables** (7 weather features with climatology-based processing)
- **ESA WorldCover land classification**

The model uses ConvLSTM layers to extract spatiotemporal patterns and predicts reflectance deltas via a parametric polynomial curve that passes through the origin.

---

## Dataset

### Source: GreenEarthNet
The project uses NetCDF (`.nc`) files containing multi-modal Earth observation data.

### Input Features

| Feature | Shape | Description |
|---------|-------|-------------|
| **Sentinel-2 Bands** | `(18, 128, 128, 4)` | B02 (Blue), B03 (Green), B04 (Red), B8A (NIR) |
| **Cloud Mask** | `(18, 128, 128, 1)` | Binary mask (1 = cloudy/invalid, 0 = clear) |
| **Land Cover** | `(128, 128, 10)` | One-hot encoded ESA WorldCover classes |
| **Temporal Metadata** | `(3,)` | Normalized year, sin(day-of-year), cos(day-of-year) |
| **Weather Features** | `(12, 21)` | E-OBS climate variables (7 vars × 3 aggregations) |

### Target Output

| Component | Shape | Description |
|-----------|-------|-------------|
| **Cloud Mask** | `(12, 128, 128, 1)` | Validity mask for target frames |
| **Reflectance Deltas** | `(12, 128, 128, 4)` | Change from Best Available Pixel (BAP) |
| **BAP Composite** | `(12, 128, 128, 4)` | Reference image for reconstruction |

---

## Data Transformation Pipeline

### 1. Sentinel-2 Preprocessing
```
Raw Observations → NaN Detection → Cloud Masking → Temporal Slicing (every 5 days)
```
- **Temporal sampling**: Extracts every 5th day from day 4 (input) and day 94 (target)
- **NaN handling**: Missing values are merged into the cloud mask and replaced with zeros
- **BAP composite**: Best Available Pixel method iterates backwards to find cloud-free pixels

### 2. Temporal Feature Engineering
- **Year normalization**: `(year - 2017) / 4` for 2017–2021 range
- **Cyclic DOY encoding**: Sine and cosine of `2π × day_of_year / days_in_year`

### 3. Weather Feature Processing (E-OBS)

| Variable | Range | Description |
|----------|-------|-------------|
| `eobs_tg` | -20 to 45°C | Mean temperature |
| `eobs_hu` | 0–100% | Relative humidity |
| `eobs_pp` | 950–1050 hPa | Sea level pressure |
| `eobs_qq` | 0–400 W/m² | Global radiation |
| `eobs_rr` | 0–50 mm | Precipitation |
| `eobs_tn` | -30 to 35°C | Minimum temperature |
| `eobs_tx` | -10 to 50°C | Maximum temperature |

**Climatology-based detrending**:
1. Compute 21-day rolling mean (climatology)
2. Calculate anomalies: `value - climatology`
3. For each 5-day forecast step, extract:
   - `min_detrend`: Minimum anomaly (normalized to [-1, 1])
   - `max_detrend`: Maximum anomaly (normalized to [-1, 1])
   - `mean_clima`: Mean climatology (normalized to [0, 1])

---

## Model Architecture

### Encoder (ConvLSTM)
```
Input Concatenation: [Sentinel-2 (4) + CloudMask (1) + LandCover (10)] = 15 channels
    ↓
ConvLSTM2D(32 filters, 3×3 kernel, return_sequences=True) + BatchNorm
    ↓
ConvLSTM2D(64 filters, 3×3 kernel, return_sequences=False) + BatchNorm
    ↓
Context Volume: (Batch, 128, 128, 64)
```

### Multi-modal Fusion
```
Temporal Embedding: Dense(16) on last timestep metadata
Weather Embedding:  Dense(16) → Mean over output steps
    ↓
Spatial Broadcasting: Tile embeddings to (H, W) and concatenate with context
    ↓
Fused Context: (Batch, 128, 128, 96)  [64 + 16 + 16]
```

### Parametric Decoder
Instead of a recurrent decoder, the model predicts **polynomial coefficients** for a parametric curve:

```
Conv2D(64, 3×3, ReLU) + BatchNorm
    ↓
Conv2D(12, 1×1)  → Coefficients (4 bands × 3 degrees)
    ↓
ParametricEvalLayer → Evaluates cubic curve at 12 output timesteps
```

**Curve definition** (passes through origin):
```math
y(t) = w_3 t^3 + w_2 t^2 + w_1 t
```

Evaluation uses Horner's method for numerical stability.

---

## Loss Function: kNDVI Loss

The `kNDVILoss` combines regression accuracy with vegetation index prediction.

### Components

1. **Masked Huber Loss** (on reflectance deltas):
   ```python
   L_reg = HuberLoss(true_deltas, pred_deltas) × (1 - cloudmask)
   ```

2. **kNDVI Loss** (kernel NDVI using RBF):
   ```python
   k(n, r) = exp(-(n - r)² / 2σ²)
   kNDVI = (1 - k) / (1 + k)
   L_kndvi = |kNDVI_true - kNDVI_pred| × (1 - cloudmask)
   ```

### Combined Loss
```python
L_total = α × L_reg + β × L_kndvi
```
Default: `α=0.5`, `β=1.0`, `σ=0.5`

### Reconstruction
Full reflectance is reconstructed as `BAP + deltas`, allowing kNDVI computation on absolute values while training on relative changes.

---

## Project Structure

```
bspline_ndvi/
├── src/
│   ├── build_model.py   # Model architecture & ParametricEvalLayer
│   ├── dataset.py       # DatasetGenerator with weather processing
│   ├── losses.py        # MaskedHuberLoss & kNDVILoss
│   ├── train.py         # Training pipeline with CLI
│   ├── config.py        # Dataset paths
│   └── visualize.py     # Visualization utilities
├── checkpoints/         # Saved model weights
├── logs/                # TensorBoard logs
└── tests/               # Unit tests
```

---

## Usage

### Training
```bash
python src/train.py \
    --train_dir /path/to/train/data \
    --val_dir /path/to/val/data \
    --epochs 500 \
    --batch_size 2 \
    --learning_rate 1e-3
```

### Resume from Checkpoint
```bash
python src/train.py \
    --resume_from checkpoints/model_at_100.keras \
    --initial_epoch 100
```

### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--train_dir` | config path | Training data directory |
| `--val_dir` | config path | Validation data directory |
| `--epochs` | 500 | Total training epochs |
| `--batch_size` | 1 | Batch size |
| `--learning_rate` | 1e-3 | Initial learning rate |
| `--checkpoint_dir` | `checkpoints/` | Model save directory |

---

## Requirements

- Python 3.9+
- TensorFlow/Keras 2.x
- NumPy
- xarray
- pandas
- glob
