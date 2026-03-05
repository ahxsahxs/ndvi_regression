#!/usr/bin/env python3
"""
Tint diagnostics — band spectral imbalance in Fourier predictions.

The "oscillating tint" problem: Blue/Green Fourier harmonics oscillate at a
different amplitude or phase than Red/NIR, producing a systematic colour sweep
across the forecast horizon (warm → neutral → purple or similar).

Outputs:
  images/diagnostics/
    19_temporal_band_trajectories.png  – Per-band delta time series vs targets
    20_cross_band_correlation.png      – Pairwise temporal correlation between bands
    21_harmonic_imbalance.png          – Mean a_k / b_k per band per harmonic
    22_dc_vs_harmonic_variance.png     – DC offset contribution vs Fourier harmonics
    23_loss_components.png             – Runtime value of each loss term
    24_weather_residual_magnitude.png  – Weather residual amplitude over time

Usage:
    conda run -n tf-gpu python scripts/diagnose_tint.py --split val_chopped --max_samples 20
"""

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_model import load_model
from dataset import DatasetGenerator
from config import DATASET_PATH, VALIDATION_PATH
from adapt_inputs import adapt_inputs

DIAG_DIR = os.path.join(PROJECT_ROOT, "images", "diagnostics")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.keras")

BAND_NAMES = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B8A (NIR)"]
BAND_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:brown"]



# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_records(model, dataset, max_samples):
    """Run inference and collect per-sample arrays including intermediate outputs."""
    # Build submodels to extract intermediate outputs
    fourier_head_out = None
    fourier_synth_out = None
    dc_out = None
    weather_res_out = None

    for layer in model.layers:
        if layer.name == "fourier_head":
            fourier_head_out = layer.output
        elif layer.name == "fourier_synthesis":
            fourier_synth_out = layer.output
        elif layer.name == "dc_add":
            # output after DC is added but before weather residual
            dc_add_out = layer.output
        elif layer.name == "weather_corr_tile":
            weather_res_out = layer.output

    extra_outputs = []
    output_keys = []

    if fourier_head_out is not None:
        extra_outputs.append(fourier_head_out)
        output_keys.append("coeffs")
    if fourier_synth_out is not None:
        extra_outputs.append(fourier_synth_out)
        output_keys.append("harmonics")
    try:
        if dc_add_out is not None:
            extra_outputs.append(dc_add_out)
            output_keys.append("dc_add")
    except NameError:
        pass
    if weather_res_out is not None:
        extra_outputs.append(weather_res_out)
        output_keys.append("weather_res")

    if extra_outputs:
        extractor = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.output] + extra_outputs
        )
    else:
        extractor = None

    records = []
    count = 0
    for x_batch, y_batch in dataset:
        if extractor is not None:
            outputs = extractor(x_batch, training=False)
            pred = outputs[0].numpy()
            extras = {k: outputs[i + 1].numpy() for i, k in enumerate(output_keys)}
        else:
            pred = model(x_batch, training=False).numpy()
            extras = {}

        records.append({
            "mask": y_batch[..., 0:1].numpy(),
            "true_deltas": y_batch[..., 1:5].numpy(),
            "bap": y_batch[..., 5:9].numpy(),
            "pred_deltas": pred,
            **extras,
        })
        count += pred.shape[0]
        if count >= max_samples:
            break

    return records


# ---------------------------------------------------------------------------
# Plot 19: Temporal band trajectories
# ---------------------------------------------------------------------------

def plot_temporal_trajectories(records, save_dir, n_samples=5):
    """
    For selected samples, plot the predicted delta time series per band
    overlaid on the true targets. Shows phase/amplitude mismatch between bands.

    A tint-free prediction has all 4 bands following the target in the same
    direction at each timestep. Tint shows as one band systematically leading
    or lagging the others.
    """
    n_samples = min(n_samples, len(records))
    T = records[0]["pred_deltas"].shape[1]
    days = [(t + 1) * 5 + 50 for t in range(T)]

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 3.5 * n_samples),
                             gridspec_kw={"wspace": 0.05})
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, rec in enumerate(records[:n_samples]):
        # Average over spatial dims (clear pixels only) to get (T, 4) trajectories
        mask = 1.0 - rec["mask"][0, :, :, :, 0]  # (T, H, W)
        n_clear = mask.sum(axis=(1, 2)) + 1e-6    # (T,)

        pred = rec["pred_deltas"][0]   # (T, H, W, 4)
        true = rec["true_deltas"][0]

        pred_mean = (pred * mask[..., np.newaxis]).sum(axis=(1, 2)) / n_clear[:, np.newaxis]  # (T, 4)
        true_mean = (true * mask[..., np.newaxis]).sum(axis=(1, 2)) / n_clear[:, np.newaxis]

        # Left: True targets
        ax = axes[row, 0]
        for b in range(4):
            ax.plot(days, true_mean[:, b], "-", color=BAND_COLORS[b],
                    label=BAND_NAMES[b], linewidth=2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Mean delta", fontsize=10)
        if row == 0:
            ax.set_title("Target Deltas", fontsize=12, fontweight="bold")
        if row == n_samples - 1:
            ax.set_xlabel("Forecast day", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.annotate(f"Sample {row}", xy=(0.01, 0.97), xycoords="axes fraction",
                    va="top", fontsize=9)

        # Right: Predictions
        ax = axes[row, 1]
        for b in range(4):
            ax.plot(days, pred_mean[:, b], "-", color=BAND_COLORS[b],
                    label=BAND_NAMES[b], linewidth=2)
            ax.plot(days, true_mean[:, b], "--", color=BAND_COLORS[b],
                    alpha=0.4, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if row == 0:
            ax.set_title("Predicted Deltas (dashed = target)", fontsize=12, fontweight="bold")
        if row == n_samples - 1:
            ax.set_xlabel("Forecast day", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Temporal Band Trajectories (spatial mean over clear pixels)\n"
                 "Tint = bands oscillate out of phase with each other",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "19_temporal_band_trajectories.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 19_temporal_band_trajectories.png")


# ---------------------------------------------------------------------------
# Plot 20: Cross-band temporal correlation
# ---------------------------------------------------------------------------

def plot_cross_band_correlation(records, save_dir):
    """
    For each sample compute the pairwise Pearson correlation between the 4 band
    time series (20 timesteps). Average across samples and compare to target.

    In real vegetation, all bands co-move (+0.6 to +0.9 correlation).
    Tint shows as Blue having low or negative correlation with NIR/Red.
    """
    pred_corrs = []
    true_corrs = []

    for rec in records:
        mask = 1.0 - rec["mask"][0, :, :, :, 0]  # (T, H, W)
        pred = rec["pred_deltas"][0]  # (T, H, W, 4)
        true = rec["true_deltas"][0]

        n_clear = mask.sum(axis=(1, 2)) + 1e-6    # (T,)
        pred_traj = (pred * mask[..., np.newaxis]).sum(axis=(1, 2)) / n_clear[:, np.newaxis]  # (T, 4)
        true_traj = (true * mask[..., np.newaxis]).sum(axis=(1, 2)) / n_clear[:, np.newaxis]

        # Correlation matrix (4, 4)
        if pred_traj.std() > 1e-8:
            pred_corrs.append(np.corrcoef(pred_traj.T))  # (4, 4)
        if true_traj.std() > 1e-8:
            true_corrs.append(np.corrcoef(true_traj.T))

    if not pred_corrs:
        print("  ⚠ Cross-band correlation: insufficient variance, skipping.")
        return {}

    mean_pred_corr = np.mean(pred_corrs, axis=0)  # (4, 4)
    mean_true_corr = np.mean(true_corrs, axis=0) if true_corrs else np.eye(4)

    short_names = ["Blue", "Green", "Red", "NIR"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def plot_corr_matrix(ax, matrix, title, vmin=-1, vmax=1):
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(short_names, fontsize=11)
        ax.set_yticklabels(short_names, fontsize=11)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black" if abs(matrix[i, j]) < 0.7 else "white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_corr_matrix(axes[0], mean_true_corr, "Target: Cross-band Correlation")
    plot_corr_matrix(axes[1], mean_pred_corr, "Predicted: Cross-band Correlation")

    # Difference panel
    diff = mean_pred_corr - mean_true_corr
    plot_corr_matrix(axes[2], diff, "Difference (Pred − Target)\n(negative = pred correlation too low)",
                     vmin=-1, vmax=1)

    fig.suptitle("Cross-band Temporal Correlation (averaged across samples)\n"
                 "Tint = Blue has low/negative correlation with Red/NIR in predictions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "20_cross_band_correlation.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 20_cross_band_correlation.png")

    return {
        "pred_blue_nir_corr": float(mean_pred_corr[0, 3]),
        "true_blue_nir_corr": float(mean_true_corr[0, 3]),
        "pred_blue_red_corr": float(mean_pred_corr[0, 2]),
        "true_blue_red_corr": float(mean_true_corr[0, 2]),
    }


# ---------------------------------------------------------------------------
# Plot 21: Per-harmonic spectral imbalance
# ---------------------------------------------------------------------------

def plot_harmonic_imbalance(records, save_dir):
    """
    For each harmonic k, plot the mean a_k and b_k per band as a grouped bar chart.

    Systematic differences in the mean coefficient values across bands reveal
    which harmonic is causing the tint. A balanced decoder would have all bands
    with similar mean coefficient magnitudes.
    """
    all_coeffs = [r["coeffs"] for r in records if r.get("coeffs") is not None]
    if not all_coeffs:
        print("  ⚠ No coefficient data, skipping plot 21.")
        return {}

    # all_coeffs: list of (B, H, W, n_bands, 2K)
    coeffs = np.concatenate(all_coeffs, axis=0)  # (N, H, W, n_bands, 2K)
    N, H, W, n_bands, n_coeffs = coeffs.shape
    n_harmonics = n_coeffs // 2

    # Subsample spatial pixels
    flat = coeffs.reshape(N * H * W, n_bands, n_coeffs)
    if flat.shape[0] > 20000:
        idx = np.random.choice(flat.shape[0], 20000, replace=False)
        flat = flat[idx]

    # Mean and std of a_k and b_k per band per harmonic
    mean_a = np.zeros((n_harmonics, n_bands))
    mean_b = np.zeros((n_harmonics, n_bands))
    std_a = np.zeros((n_harmonics, n_bands))
    std_b = np.zeros((n_harmonics, n_bands))
    for k in range(n_harmonics):
        for b in range(n_bands):
            mean_a[k, b] = flat[:, b, 2 * k].mean()
            std_a[k, b] = flat[:, b, 2 * k].std()
            mean_b[k, b] = flat[:, b, 2 * k + 1].mean()
            std_b[k, b] = flat[:, b, 2 * k + 1].std()

    fig, axes = plt.subplots(n_harmonics, 2, figsize=(14, 2.8 * n_harmonics),
                             gridspec_kw={"hspace": 0.5})
    if n_harmonics == 1:
        axes = axes[np.newaxis, :]

    x = np.arange(n_bands)
    w = 0.2

    for k in range(n_harmonics):
        # a_k panel
        ax = axes[k, 0]
        for b in range(n_bands):
            ax.bar(b, mean_a[k, b], color=BAND_COLORS[b], alpha=0.8,
                   label=BAND_NAMES[b] if k == 0 else "")
            ax.errorbar(b, mean_a[k, b], yerr=std_a[k, b], fmt="none",
                        color="black", capsize=3, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels(["Blue", "Green", "Red", "NIR"], fontsize=9)
        ax.set_ylabel("Mean coefficient")
        ax.set_title(f"k={k+1}: mean a_k (cosine)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if k == 0:
            ax.legend(fontsize=7, ncol=2)

        # b_k panel
        ax = axes[k, 1]
        for b in range(n_bands):
            ax.bar(b, mean_b[k, b], color=BAND_COLORS[b], alpha=0.8)
            ax.errorbar(b, mean_b[k, b], yerr=std_b[k, b], fmt="none",
                        color="black", capsize=3, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels(["Blue", "Green", "Red", "NIR"], fontsize=9)
        ax.set_title(f"k={k+1}: mean b_k (sine)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Harmonic Mean Coefficients per Band (error bars = ±1 std)\n"
                 "Imbalance = systematic offset between bands for the same harmonic",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "21_harmonic_imbalance.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 21_harmonic_imbalance.png")

    # Identify the most imbalanced harmonic
    # Imbalance = std across bands of the mean coefficient
    imbalance_a = mean_a.std(axis=1)  # (n_harmonics,)
    imbalance_b = mean_b.std(axis=1)
    worst_k = int(np.argmax(imbalance_a + imbalance_b)) + 1

    return {
        "worst_imbalance_harmonic": worst_k,
        "mean_a": mean_a.tolist(),
        "mean_b": mean_b.tolist(),
    }


# ---------------------------------------------------------------------------
# Plot 22: DC offset vs harmonic variance contribution
# ---------------------------------------------------------------------------

def plot_dc_vs_harmonic_variance(records, save_dir):
    """
    Decompose the predicted output into its Fourier harmonics component and
    DC offset contribution. Measures what fraction of the prediction variance
    comes from each component.

    If DC dominates, the model is using the constant offset as a lazy prediction
    strategy and the Fourier harmonics are under-utilised. If harmonics dominate
    and DC ≈ 0, sustained trends cannot be captured.
    """
    harmonics_list = [r.get("harmonics") for r in records if r.get("harmonics") is not None]
    dc_add_list = [r.get("dc_add") for r in records if r.get("dc_add") is not None]

    if not harmonics_list or not dc_add_list:
        print("  ⚠ DC/harmonic intermediate outputs not available, skipping plot 22.")
        print("    (layer 'fourier_synthesis' or 'dc_add' not found in model)")
        return {}

    harmonics = np.concatenate(harmonics_list, axis=0)   # (N, T, H, W, 4)
    dc_add = np.concatenate(dc_add_list, axis=0)          # (N, T, H, W, 4)
    # DC offset contribution: dc_add - harmonics = just the DC term broadcast
    dc_only = dc_add - harmonics                          # (N, T, H, W, 4)

    # Variance per component (across T and spatial for each sample then average)
    harm_var = harmonics.var(axis=(1, 2, 3))  # (N, 4)
    dc_var = dc_only.var(axis=(1, 2, 3))
    pred_var = dc_add.var(axis=(1, 2, 3))

    # Fraction of total variance
    harm_frac = harm_var / (pred_var + 1e-8)  # (N, 4)
    dc_frac = dc_var / (pred_var + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    short_names = ["Blue", "Green", "Red", "NIR"]

    # Panel 1: Mean variance fraction per band
    ax = axes[0]
    x = np.arange(4)
    w = 0.35
    harm_mean = harm_frac.mean(axis=0)
    dc_mean = dc_frac.mean(axis=0)
    ax.bar(x - w / 2, harm_mean, w, label="Fourier harmonics", color="tab:blue", alpha=0.8)
    ax.bar(x + w / 2, dc_mean, w, label="DC offset", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Fraction of prediction variance")
    ax.set_title("Variance Fraction: Harmonics vs DC Offset\nper Band",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for i in range(4):
        ax.text(i - w / 2, harm_mean[i] + 0.01, f"{harm_mean[i]:.2f}",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, dc_mean[i] + 0.01, f"{dc_mean[i]:.2f}",
                ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Distribution of DC variance fraction across samples
    ax = axes[1]
    for b in range(4):
        ax.hist(dc_frac[:, b], bins=20, alpha=0.5, label=BAND_NAMES[b],
                color=BAND_COLORS[b], density=True)
    ax.set_xlabel("DC offset variance fraction per sample")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of DC Offset Contribution\nAcross Samples",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Mean harmonic trajectory over time (absolute signal level)
    ax = axes[2]
    T = harmonics.shape[1]
    days = [(t + 1) * 5 + 50 for t in range(T)]
    harm_t = np.abs(harmonics).mean(axis=(0, 2, 3))  # (T, 4)
    dc_t = np.abs(dc_only).mean(axis=(0, 2, 3))
    for b in range(4):
        ax.plot(days, harm_t[:, b], "-", color=BAND_COLORS[b],
                label=f"{BAND_NAMES[b].split('(')[1].rstrip(')')} harm", linewidth=2)
        ax.plot(days, dc_t[:, b], "--", color=BAND_COLORS[b], alpha=0.5,
                label=f"{BAND_NAMES[b].split('(')[1].rstrip(')')} DC", linewidth=1)
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Mean |delta| (reflectance units)")
    ax.set_title("Harmonic (solid) vs DC offset (dashed)\nMean Absolute Magnitude over Time",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("DC Offset vs Fourier Harmonics Contribution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "22_dc_vs_harmonic_variance.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 22_dc_vs_harmonic_variance.png")

    return {
        "mean_harmonic_var_frac": harm_mean.tolist(),
        "mean_dc_var_frac": dc_mean.tolist(),
    }


# ---------------------------------------------------------------------------
# Plot 23: Runtime loss component values
# ---------------------------------------------------------------------------

def plot_loss_components(model, dataset, save_dir):
    """
    Run one forward pass and report the actual contribution of each loss term.

    Answers: Is the tint penalty (tint_weight × total_tint) large enough to
    matter, or is it negligible compared to the regression term?

    Requires the compiled loss to be DeltaRegressionLoss.
    """
    print("  Computing loss components (1 batch)...")
    x_batch, y_batch = next(iter(dataset))
    pred = model(x_batch, training=False)

    y_true = y_batch
    y_pred = pred

    # Manually recompute each component so we don't need the compiled loss
    cloudmask = y_true[..., 0:1]
    true_deltas = y_true[..., 1:5]
    bap = y_true[..., 5:9]
    pred_deltas = y_pred

    clear = 1.0 - cloudmask
    clear_sq = tf.squeeze(clear, axis=-1)
    n_clear = tf.reduce_sum(clear_sq) + 1e-6

    band_weights = (0.75, 0.75, 1.25, 3.5)
    bw = tf.constant(band_weights, dtype=tf.float32)

    def stable_log_cosh(x):
        ax = tf.abs(x)
        return ax + tf.math.softplus(-2.0 * ax) - tf.math.log(2.0)

    def weighted_log_cosh(error):
        per_band = stable_log_cosh(error) * bw
        return tf.reduce_sum(per_band, axis=-1) / tf.reduce_sum(bw)

    # 1. Regression
    error = pred_deltas - true_deltas
    reg_loss = weighted_log_cosh(error)
    under_mask = tf.cast(tf.abs(pred_deltas) < tf.abs(true_deltas), tf.float32)
    asym_mult = tf.reduce_sum(under_mask * bw, axis=-1) / tf.reduce_sum(bw)
    reg_loss = reg_loss * (1.0 + 0.5 * asym_mult)
    total_reg = float(tf.reduce_sum(reg_loss * clear_sq) / n_clear)

    # 2. Edge
    shape = tf.shape(true_deltas)
    H, W = shape[2], shape[3]
    true_flat = tf.reshape(true_deltas, [-1, H, W, shape[4]])
    pred_flat = tf.reshape(pred_deltas, [-1, H, W, shape[4]])
    w_flat = tf.reshape(clear, [-1, H, W, 1])
    true_dy, true_dx = tf.image.image_gradients(true_flat)
    pred_dy, pred_dx = tf.image.image_gradients(pred_flat)
    edge_y = weighted_log_cosh(pred_dy - true_dy)
    edge_x = weighted_log_cosh(pred_dx - true_dx)
    w_flat_sq = tf.squeeze(w_flat, axis=-1)
    total_edge = float(tf.reduce_sum((edge_y + edge_x) * w_flat_sq) /
                       (tf.reduce_sum(w_flat_sq) + 1e-6))

    # 3. NDVI
    ndvi_eps = 0.05
    full_true = tf.maximum(bap + true_deltas, 0.0)
    full_pred = tf.maximum(bap + pred_deltas, 0.0)
    red_t, nir_t = full_true[..., 2:3], full_true[..., 3:4]
    red_p, nir_p = full_pred[..., 2:3], full_pred[..., 3:4]
    ndvi_true = (nir_t - red_t) / (nir_t + red_t + ndvi_eps)
    ndvi_pred = (nir_p - red_p) / (nir_p + red_p + ndvi_eps)
    ndvi_err = tf.squeeze(stable_log_cosh(ndvi_pred - ndvi_true), axis=-1)
    total_ndvi = float(tf.reduce_sum(ndvi_err * clear_sq) / n_clear)

    # 4. Spatial diversity
    n_clear_spatial = tf.reduce_sum(clear, axis=[2, 3]) + 1e-6
    pred_masked = pred_deltas * clear
    true_masked = true_deltas * clear
    pred_mean = tf.reduce_sum(pred_masked, axis=[2, 3]) / n_clear_spatial
    true_mean = tf.reduce_sum(true_masked, axis=[2, 3]) / n_clear_spatial
    pred_mean_exp = pred_mean[:, :, tf.newaxis, tf.newaxis, :]
    true_mean_exp = true_mean[:, :, tf.newaxis, tf.newaxis, :]
    pred_mad = tf.reduce_sum(tf.abs(pred_deltas - pred_mean_exp) * clear, axis=[2, 3]) / n_clear_spatial
    true_mad = tf.reduce_sum(tf.abs(true_deltas - true_mean_exp) * clear, axis=[2, 3]) / n_clear_spatial
    total_diversity = float(tf.reduce_mean(tf.nn.relu(true_mad - pred_mad)))

    # 5a. Temporal smoothness
    d_pred = pred_deltas[:, 1:] - pred_deltas[:, :-1]
    dd_pred = d_pred[:, 1:] - d_pred[:, :-1]
    total_temporal = float(tf.reduce_mean(tf.keras.losses.huber(
        tf.zeros_like(dd_pred), dd_pred, delta=0.02)))

    # 5b. Tint (temporal MAD excess)
    n_clear_t = tf.reduce_sum(clear, axis=1) + 1e-6
    pred_t_mean = tf.reduce_sum(pred_deltas * clear, axis=1) / n_clear_t
    true_t_mean = tf.reduce_sum(true_deltas * clear, axis=1) / n_clear_t
    pred_t_mad = tf.reduce_sum(
        tf.abs(pred_deltas - pred_t_mean[:, tf.newaxis]) * clear, axis=1) / n_clear_t
    true_t_mad = tf.reduce_sum(
        tf.abs(true_deltas - true_t_mean[:, tf.newaxis]) * clear, axis=1) / n_clear_t
    total_tint = float(tf.reduce_mean(tf.nn.relu(pred_t_mad - true_t_mad)))

    # 7. Cross-band temporal correlation matching
    pred_t_mean_exp = pred_t_mean[:, tf.newaxis]  # (B, 1, H, W, 4)
    true_t_mean_exp = true_t_mean[:, tf.newaxis]
    pred_centered = (pred_deltas - pred_t_mean_exp) * clear
    true_centered = (true_deltas - true_t_mean_exp) * clear
    pred_var = tf.reduce_sum(pred_centered ** 2, axis=1) / n_clear_t
    true_var = tf.reduce_sum(true_centered ** 2, axis=1) / n_clear_t
    pred_std_cb = tf.sqrt(pred_var + 1e-3)
    true_std_cb = tf.sqrt(true_var + 1e-3)
    pixel_valid = tf.cast((n_clear_t - 1e-6) >= 3.0, tf.float32)
    BAND_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    corr_mse_sum = 0.0
    for i, j in BAND_PAIRS:
        cov_pred = tf.reduce_sum(
            pred_centered[..., i:i+1] * pred_centered[..., j:j+1], axis=1
        ) / n_clear_t
        cov_true = tf.reduce_sum(
            true_centered[..., i:i+1] * true_centered[..., j:j+1], axis=1
        ) / n_clear_t
        denom_pred = pred_std_cb[..., i:i+1] * pred_std_cb[..., j:j+1] + 1e-3
        denom_true = true_std_cb[..., i:i+1] * true_std_cb[..., j:j+1] + 1e-3
        corr_pred_ij = tf.clip_by_value(cov_pred / denom_pred, -1.0, 1.0)
        corr_true_ij = tf.clip_by_value(cov_true / denom_true, -1.0, 1.0)
        corr_mse_sum += tf.reduce_sum(
            tf.square(corr_pred_ij - corr_true_ij) * pixel_valid
        )
    n_valid = tf.reduce_sum(pixel_valid) + 1e-6
    total_corr = float(corr_mse_sum / (n_valid * 6.0))

    # Weighted contributions
    weights = {"Regression\n(×5.0)": 5.0, "Edge\n(×0.1)": 0.1,
               "NDVI\n(×1.5)": 1.5, "Diversity\n(×2.0)": 2.0,
               "Temporal\n(×0.1)": 0.1, "Tint\n(×0.5)": 0.5,
               "Correlation\n(×0.1)": 0.1}
    raw = [total_reg, total_edge, total_ndvi, total_diversity, total_temporal, total_tint, total_corr]
    labels = list(weights.keys())
    weighted = [r * w for r, w in zip(raw, weights.values())]
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Raw values
    ax = axes[0]
    bars = ax.bar(labels, raw, color=colors, alpha=0.8)
    ax.set_ylabel("Raw loss value", fontsize=12)
    ax.set_title("Raw Loss Component Values\n(before weighting)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, raw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(raw) * 0.01,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Weighted values
    ax = axes[1]
    total = sum(weighted)
    bars = ax.bar(labels, weighted, color=colors, alpha=0.8)
    ax.set_ylabel("Weighted contribution (loss units)", fontsize=12)
    ax.set_title(f"Weighted Loss Contributions\n(total = {total:.5f})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, weighted):
        frac = val / (total + 1e-8)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(weighted) * 0.01,
                f"{frac:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Runtime Loss Component Breakdown (1 batch)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "23_loss_components.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 23_loss_components.png")

    return {k.replace("\n", "_"): w for k, w in zip(labels, weighted)}


# ---------------------------------------------------------------------------
# Plot 24: Weather residual magnitude
# ---------------------------------------------------------------------------

def plot_weather_residual_magnitude(records, save_dir):
    """
    Compare the per-timestep weather residual magnitude to the Fourier harmonics
    output. If weather_residual ≈ 0, the Dense layer has not learned to modulate
    individual forecast steps.
    """
    weather_list = [r.get("weather_res") for r in records if r.get("weather_res") is not None]
    harmonics_list = [r.get("harmonics") for r in records if r.get("harmonics") is not None]

    if not weather_list:
        print("  ⚠ Weather residual intermediate output not available, skipping plot 24.")
        print("    (layer 'weather_corr_tile' not found in model graph)")
        return {}

    weather_res = np.concatenate(weather_list, axis=0)   # (N, T, H, W, 4)
    harmonics = np.concatenate(harmonics_list, axis=0)   # (N, T, H, W, 4)

    T = weather_res.shape[1]
    days = [(t + 1) * 5 + 50 for t in range(T)]

    # Mean absolute value over space and samples
    weather_abs = np.abs(weather_res).mean(axis=(0, 2, 3))   # (T, 4)
    harm_abs = np.abs(harmonics).mean(axis=(0, 2, 3))         # (T, 4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Weather residual magnitude per timestep per band
    ax = axes[0]
    for b in range(4):
        ax.plot(days, weather_abs[:, b], "-o", color=BAND_COLORS[b],
                label=BAND_NAMES[b], linewidth=2, markersize=4)
    ax.set_xlabel("Forecast day", fontsize=12)
    ax.set_ylabel("Mean |weather residual|", fontsize=12)
    ax.set_title("Weather Residual Magnitude per Timestep",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Ratio: weather residual / harmonic output
    ax = axes[1]
    ratio = weather_abs / (harm_abs + 1e-8)
    for b in range(4):
        ax.plot(days, ratio[:, b], "-o", color=BAND_COLORS[b],
                label=BAND_NAMES[b], linewidth=2, markersize=4)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.6, label="10% threshold")
    ax.set_xlabel("Forecast day", fontsize=12)
    ax.set_ylabel("|weather residual| / |harmonics|", fontsize=12)
    ax.set_title("Weather Residual as Fraction of Harmonic Output\n"
                 "(< 0.1 = negligible; > 0.3 = actively contributing)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Band-averaged summary
    ax = axes[2]
    mean_weather = weather_abs.mean(axis=0)  # (4,)
    mean_harm = harm_abs.mean(axis=0)
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w / 2, mean_harm, w, label="Harmonics", color="tab:blue", alpha=0.8)
    ax.bar(x + w / 2, mean_weather, w, label="Weather residual", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Blue", "Green", "Red", "NIR"])
    ax.set_ylabel("Mean |delta| (reflectance units)", fontsize=12)
    ax.set_title("Harmonics vs Weather Residual\nBand-Averaged Magnitude",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for i in range(4):
        r = mean_weather[i] / (mean_harm[i] + 1e-8)
        ax.text(i + w / 2, mean_weather[i] + max(mean_harm.max(), mean_weather.max()) * 0.01,
                f"{r:.2f}×", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Weather Residual Contribution (per-timestep Dense layer)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "24_weather_residual_magnitude.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 24_weather_residual_magnitude.png")

    return {
        "mean_weather_res": float(weather_abs.mean()),
        "mean_harmonic": float(harm_abs.mean()),
        "weather_fraction": float(weather_abs.mean() / (harm_abs.mean() + 1e-8)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def diagnose_tint(model, generator, split="val_chopped", max_samples=20):
    """Run tint diagnostics with a pre-loaded model and generator."""
    os.makedirs(DIAG_DIR, exist_ok=True)

    dataset = generator.get_dataset()
    dataset = dataset.batch(1).map(adapt_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"\nCollecting predictions ({max_samples} samples) ...")
    records = collect_records(model, dataset, max_samples)
    print(f"  Collected: {len(records)} samples")

    print("\nGenerating tint diagnostic plots ...")
    plot_temporal_trajectories(records, DIAG_DIR, n_samples=5)
    plot_cross_band_correlation(records, DIAG_DIR)
    plot_harmonic_imbalance(records, DIAG_DIR)
    plot_dc_vs_harmonic_variance(records, DIAG_DIR)
    plot_loss_components(model, dataset, DIAG_DIR)
    plot_weather_residual_magnitude(records, DIAG_DIR)

    print(f"\nAll tint diagnostics saved to: {DIAG_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Tint diagnostic — band spectral imbalance.")
    parser.add_argument("--split", type=str, default="val_chopped",
                        choices=["train", "val_chopped", "val"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model_path = args.model or MODEL_PATH
    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, compile=False)
    print(f"  Parameters: {model.count_params():,}")

    data_path = VALIDATION_PATH if args.split in ("val_chopped", "val") else DATASET_PATH
    print(f"Loading dataset from {data_path} (split={args.split}) ...")
    generator = DatasetGenerator(data_path)

    diagnose_tint(model, generator, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
