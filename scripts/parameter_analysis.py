#!/usr/bin/env python3
"""
Parameter analysis script for diagnosing mode collapse.

Investigates the root causes of the model collapsing to near-zero predictions
by examining gradient flow, Fourier coefficient distributions, loss component
magnitudes, and prediction diversity.

Outputs:
  images/diagnostics/
    10_gradient_norms.png         – Gradient norm per layer (identifies dead layers)
    11_per_harmonic_coeffs.png    – Actual a_k/b_k coefficient distributions
    12_prediction_diversity.png   – Prediction std across samples (mode collapse check)
    13_l2_vs_regression_loss.png  – L2 penalty vs regression loss magnitudes
    14_kndvi_dead_code.png        – What kNDVI loss WOULD be (demonstrating it's 0)
    15_target_delta_by_lc.png     – Target delta distributions by land cover class
    16_cloud_cover_analysis.png   – Cloud fraction in inputs and targets
    17_fourier_basis.png          – Fourier basis functions for sample DOYs
    18_collapse_summary.txt       – Quantitative summary of collapse indicators

Usage:
    conda run -n tf-gpu python scripts/parameter_analysis.py --split val_chopped --max_samples 20
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_model import load_model
from dataset import DatasetGenerator
from config import DATASET_PATH, VALIDATION_PATH

DIAG_DIR = os.path.join(PROJECT_ROOT, "images", "diagnostics")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.keras")

BAND_NAMES = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B8A (NIR)"]
LC_NAMES = [
    "Tree cover", "Shrubland", "Grassland", "Cropland",
    "Built-up", "Bare/sparse", "Snow/Ice", "Water", "Wetland", "Mangrove",
]


def adapt_inputs(x, y):
    t_meta = x["time"][:, -1, :]
    return {
        "sentinel2_sequence": x["sentinel2"],
        "landcover_map": x["landcover"],
        "temporal_metadata": t_meta,
        "weather_sequence": x["weather"],
        "target_start_doy": x["target_start_doy"],
    }, y


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def collect_data(model, dataset, max_samples):
    """Collect model inputs, predictions, targets, and intermediate outputs."""
    records = []
    count = 0
    for x_batch, y_batch in dataset:
        pred = model(x_batch, training=False)

        # Extract Fourier coefficients from intermediate layer
        fourier_head = None
        for layer in model.layers:
            if "fourier_head" in layer.name:
                fourier_head = layer
                break

        coeffs = None
        if fourier_head is not None:
            # Build submodel to extract coefficients
            coeff_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=fourier_head.output
            )
            coeffs = coeff_model(x_batch, training=False).numpy()  # (B, H, W, bands, 2K)

        records.append({
            "mask": y_batch[..., 0:1].numpy(),
            "true_deltas": y_batch[..., 1:5].numpy(),
            "bap": y_batch[..., 5:9].numpy(),
            "pred_deltas": pred.numpy(),
            "cloudmask_in": x_batch["cloudmask_sequence"].numpy() if "cloudmask_sequence" in x_batch else None,
            "landcover": x_batch["landcover_map"].numpy(),
            "target_doy": x_batch["target_start_doy"].numpy(),
            "coeffs": coeffs,
        })
        count += pred.shape[0]
        if count >= max_samples:
            break

    return records


# ---------------------------------------------------------------------------
# Plot 10: Gradient Norms per Layer
# ---------------------------------------------------------------------------

def plot_gradient_norms(model, dataset, save_dir):
    """
    Compute gradient norms per layer for one batch.

    A near-zero gradient norm in an early layer means that layer receives
    no training signal — a key indicator of vanishing gradients or
    loss saturation at the output.
    """
    print("  Computing gradient norms (1 batch)...")
    x_batch, y_batch = next(iter(dataset))

    loss_fn = tf.keras.losses.MeanAbsoluteError()

    with tf.GradientTape() as tape:
        pred = model(x_batch, training=True)
        # Use MAE on deltas as a proxy loss (avoids needing compiled loss)
        true_deltas = y_batch[..., 1:5]
        mask = 1.0 - y_batch[..., 0:1]
        masked_pred = pred * mask
        masked_true = true_deltas * mask
        loss_val = tf.reduce_mean(tf.abs(masked_true - masked_pred))

    grads = tape.gradient(loss_val, model.trainable_variables)

    # Compute norm per variable and group by layer prefix
    layer_norms = {}
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None:
            continue
        # Extract layer name from variable name: "layer_name/weight:0" -> "layer_name"
        parts = var.name.split("/")
        layer_name = parts[0] if len(parts) > 1 else var.name
        norm = float(tf.norm(grad).numpy())
        if layer_name not in layer_norms:
            layer_norms[layer_name] = []
        layer_norms[layer_name].append(norm)

    # Average norm per layer
    layer_avg = {k: np.mean(v) for k, v in layer_norms.items()}

    # Sort by norm (ascending) to highlight dead layers
    sorted_layers = sorted(layer_avg.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_layers]
    norms = [item[1] for item in sorted_layers]

    # Color by magnitude: red=near-zero (< 1e-4), orange=weak, green=healthy
    colors = []
    for n in norms:
        if n < 1e-5:
            colors.append("red")
        elif n < 1e-3:
            colors.append("orange")
        else:
            colors.append("tab:green")

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), norms, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Gradient L2 Norm (log scale)", fontsize=12)
    ax.set_title("Gradient Norms per Layer\n"
                 "(red=near-zero <1e-5, orange=weak <1e-3, green=healthy)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=1e-5, color="red", linestyle="--", alpha=0.5, label="Dead threshold")
    ax.axvline(x=1e-3, color="orange", linestyle="--", alpha=0.5, label="Weak threshold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "10_gradient_norms.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 10_gradient_norms.png")

    dead_layers = [k for k, v in layer_avg.items() if v < 1e-5]
    weak_layers = [k for k, v in layer_avg.items() if 1e-5 <= v < 1e-3]
    return {"dead_layers": dead_layers, "weak_layers": weak_layers,
            "layer_norms": layer_avg, "proxy_loss": float(loss_val.numpy())}


# ---------------------------------------------------------------------------
# Plot 11: Per-Harmonic Coefficient Distributions
# ---------------------------------------------------------------------------

def plot_per_harmonic_coefficients(records, model, save_dir):
    """
    Show the actual a_k, b_k values predicted for each harmonic.

    If all harmonics have near-zero values, the FourierCoefficientHead
    is collapsing. If only high-frequency harmonics are near-zero, the
    model may be overly smooth.
    """
    # Collect coefficients across all samples
    all_coeffs = [r["coeffs"] for r in records if r["coeffs"] is not None]
    if not all_coeffs:
        print("  ⚠ No coefficient data available, skipping plot 11.")
        return {}

    # all_coeffs: list of (B, H, W, n_bands, 2K)
    coeffs = np.concatenate(all_coeffs, axis=0)  # (N, H, W, n_bands, 2K)

    n_bands = coeffs.shape[3]
    n_coeffs = coeffs.shape[4]  # 2K
    n_harmonics = n_coeffs // 2

    # Subsample spatial pixels for efficiency
    N, H, W = coeffs.shape[:3]
    flat = coeffs.reshape(N * H * W, n_bands, n_coeffs)
    if flat.shape[0] > 10000:
        idx = np.random.choice(flat.shape[0], 10000, replace=False)
        flat = flat[idx]

    fig, axes = plt.subplots(n_harmonics, 2, figsize=(14, n_harmonics * 2.5))
    if n_harmonics == 1:
        axes = axes[np.newaxis, :]

    harmonic_stats = {}
    for k in range(n_harmonics):
        a_k = flat[:, :, 2 * k].ravel()    # cosine coefficients, all bands
        b_k = flat[:, :, 2 * k + 1].ravel()  # sine coefficients, all bands

        ax_a = axes[k, 0]
        ax_b = axes[k, 1]

        for b in range(n_bands):
            a_band = flat[:, b, 2 * k]
            b_band = flat[:, b, 2 * k + 1]
            ax_a.hist(a_band, bins=50, alpha=0.5, label=BAND_NAMES[b], density=True)
            ax_b.hist(b_band, bins=50, alpha=0.5, label=BAND_NAMES[b], density=True)

        ax_a.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax_a.set_title(f"Harmonic k={k+1}: a_k (cosine)", fontsize=11, fontweight="bold")
        ax_a.set_xlabel("Coefficient value")
        ax_a.set_ylabel("Density")
        ax_a.legend(fontsize=7)
        ax_a.grid(True, alpha=0.3)

        ax_b.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax_b.set_title(f"Harmonic k={k+1}: b_k (sine)", fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Coefficient value")
        ax_b.legend(fontsize=7)
        ax_b.grid(True, alpha=0.3)

        harmonic_stats[f"k{k+1}_a_std"] = float(a_k.std())
        harmonic_stats[f"k{k+1}_b_std"] = float(b_k.std())
        harmonic_stats[f"k{k+1}_amplitude"] = float(
            np.sqrt(a_k ** 2 + b_k ** 2).mean())

    fig.suptitle("Per-Harmonic Fourier Coefficient Distributions\n"
                 "(near-zero = harmonic not contributing)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "11_per_harmonic_coeffs.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 11_per_harmonic_coeffs.png")
    return harmonic_stats


# ---------------------------------------------------------------------------
# Plot 12: Prediction Diversity (Mode Collapse Check)
# ---------------------------------------------------------------------------

def plot_prediction_diversity(records, save_dir):
    """
    Measure whether predictions vary meaningfully across samples.

    If the per-pixel std of predictions is near zero across all samples,
    the model outputs the same prediction regardless of input (mode collapse).
    A healthy model should have prediction std comparable to target std.
    """
    all_pred = np.concatenate([r["pred_deltas"] for r in records], axis=0)
    all_true = np.concatenate([r["true_deltas"] for r in records], axis=0)
    all_mask = np.concatenate([r["mask"] for r in records], axis=0)

    valid = (1.0 - all_mask[..., 0]).astype(bool)  # (N, T, H, W)

    # Prediction variance across samples (axis=0) for each pixel-timestep
    # Shape: (N, T, H, W, 4) -> std over N
    pred_std_per_pixel = all_pred.std(axis=0)   # (T, H, W, 4)
    true_std_per_pixel = all_true.std(axis=0)   # (T, H, W, 4)

    # Mean std over space/time/bands
    mean_pred_std = float(pred_std_per_pixel.mean())
    mean_true_std = float(true_std_per_pixel.mean())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Std of predictions vs targets across samples
    ax = axes[0]
    pred_stds = pred_std_per_pixel.ravel()
    true_stds = true_std_per_pixel.ravel()
    bins = np.linspace(0, max(pred_stds.max(), true_stds.max()) * 0.5, 60)
    ax.hist(pred_stds, bins=bins, alpha=0.6, label=f"Predicted (mean={mean_pred_std:.4f})",
            color="tab:red", density=True)
    ax.hist(true_stds, bins=bins, alpha=0.6, label=f"Target (mean={mean_true_std:.4f})",
            color="tab:blue", density=True)
    ax.set_xlabel("Std across samples", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Cross-Sample Variability\n(pred std << target std = mode collapse)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-band cross-sample std ratio
    ax = axes[1]
    ratios = []
    for b in range(4):
        p_std = float(all_pred[..., b].std(axis=0).mean())
        t_std = float(all_true[..., b].std(axis=0).mean())
        ratios.append(p_std / (t_std + 1e-8))
    ax.bar(range(4), ratios, color=["tab:blue", "tab:green", "tab:red", "tab:brown"], alpha=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels([b.split("(")[1].rstrip(")") for b in BAND_NAMES])
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.6, label="Perfect diversity match")
    ax.set_ylabel("Pred std / Target std", fontsize=12)
    ax.set_title("Diversity Ratio per Band\n(1.0 = matched diversity, < 1 = collapsed)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate ratios
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Per-timestep prediction std vs target std
    ax = axes[2]
    n_steps = all_pred.shape[1]
    pred_std_t = [all_pred[:, t].std() for t in range(n_steps)]
    true_std_t = [all_true[:, t].std() for t in range(n_steps)]
    days = [(t + 1) * 5 + 50 for t in range(n_steps)]
    ax.plot(days, pred_std_t, "o-", color="tab:red", label="Prediction std", linewidth=2)
    ax.plot(days, true_std_t, "o-", color="tab:blue", label="Target std", linewidth=2)
    ax.set_xlabel("Forecast day", fontsize=12)
    ax.set_ylabel("Std (across samples)", fontsize=12)
    ax.set_title("Temporal Diversity Profile", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Mode Collapse Diagnosis: Prediction Diversity vs Target Diversity",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "12_prediction_diversity.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 12_prediction_diversity.png")

    return {
        "mean_pred_std": mean_pred_std,
        "mean_true_std": mean_true_std,
        "diversity_ratio": mean_pred_std / (mean_true_std + 1e-8),
        "per_band_ratios": ratios,
    }


# ---------------------------------------------------------------------------
# Plot 13: L2 Regularization vs Regression Loss Magnitude
# ---------------------------------------------------------------------------

def plot_l2_vs_regression_loss(model, dataset, save_dir):
    """
    Compare the magnitude of L2 regularization losses vs regression loss.

    If L2 >> regression, the model is being pulled toward zero weights
    faster than the data gradient can push them outward. This directly
    causes mode collapse toward zero-amplitude predictions.
    """
    print("  Computing L2 vs regression loss (1 batch)...")
    x_batch, y_batch = next(iter(dataset))

    pred = model(x_batch, training=False)
    true_deltas = y_batch[..., 1:5]
    mask = 1.0 - y_batch[..., 0:1]

    # Log-Cosh regression loss (same as ImprovedkNDVILoss)
    error = pred - true_deltas
    log_cosh_per_pixel = tf.math.log(tf.math.cosh(error))
    masked_loss = log_cosh_per_pixel * mask
    reg_loss = float(tf.reduce_mean(masked_loss).numpy())

    # L2 regularization losses (Keras tracks these per-layer)
    l2_losses = {}
    total_l2 = 0.0
    for layer in model.layers:
        layer_l2 = 0.0
        for weight in layer.weights:
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                if 'fourier' in layer.name or 'encoder' in layer.name or 'weather' in layer.name or 'temporal' in layer.name:
                    l2_val = float(tf.reduce_sum(weight ** 2).numpy()) * 1e-4
                    layer_l2 += l2_val
        if layer_l2 > 0:
            l2_losses[layer.name] = layer_l2
            total_l2 += layer_l2

    # Also get Keras tracked losses
    keras_reg_losses = model.losses
    keras_total_l2 = sum(float(l.numpy()) for l in keras_reg_losses) if keras_reg_losses else 0.0

    # Weighted losses as in training
    weighted_reg = 5.0 * reg_loss
    weighted_edge_estimate = 0.1 * reg_loss * 0.5  # edge is roughly similar scale

    labels = ["Regression\n(×5.0 weight)", "Edge est.\n(×0.1)", "L2 Regularization\n(Keras tracked)"]
    values = [weighted_reg, weighted_edge_estimate, keras_total_l2]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Bar chart of loss components
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    ax.set_ylabel("Loss magnitude", fontsize=12)
    ax.set_title("Loss Component Magnitudes\n(if L2 >> Regression, collapse is expected)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Per-layer L2 breakdown
    ax = axes[1]
    if l2_losses:
        layer_names = list(l2_losses.keys())
        layer_vals = [l2_losses[n] for n in layer_names]
        short_names = [n.replace("fourier_", "f_").replace("encoder_", "enc_") for n in layer_names]
        ax.barh(range(len(short_names)), layer_vals, color="tab:red", alpha=0.7)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xlabel("L2 penalty (estimated)", fontsize=12)
        ax.set_title("Per-Layer L2 Regularization\n(drives weights toward zero)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, f"Keras tracked L2: {keras_total_l2:.6f}\n\nraw reg loss: {reg_loss:.6f}",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title("L2 Regularization", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "13_l2_vs_regression_loss.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 13_l2_vs_regression_loss.png")

    return {
        "raw_reg_loss": reg_loss,
        "weighted_reg_loss": weighted_reg,
        "keras_total_l2": keras_total_l2,
        "l2_to_regression_ratio": keras_total_l2 / (reg_loss + 1e-8),
    }


# ---------------------------------------------------------------------------
# Plot 14: kNDVI Dead Code Verification
# ---------------------------------------------------------------------------

def plot_kndvi_dead_code(records, save_dir):
    """
    Compute what the kNDVI loss WOULD be if it were active.

    The ImprovedkNDVILoss.call() never applies kndvi_weight — it's dead code.
    This plot shows the magnitude of the kNDVI signal that is being ignored,
    demonstrating that enabling it would provide meaningful gradient.
    """
    all_true_d = np.concatenate([r["true_deltas"] for r in records], axis=0)
    all_pred_d = np.concatenate([r["pred_deltas"] for r in records], axis=0)
    all_bap = np.concatenate([r["bap"] for r in records], axis=0)
    all_mask = np.concatenate([r["mask"] for r in records], axis=0)

    valid = (1.0 - all_mask[..., 0]).astype(bool)

    # Reconstruct full reflectance
    true_full = all_bap + all_true_d
    pred_full = all_bap + all_pred_d

    # kNDVI: (1 - k) / (1 + k), k = RBF kernel on (NIR, RED)
    sigma = 1.0
    for_each = {}
    kndvi_diffs = []
    kndvi_by_timestep = []
    n_steps = all_true_d.shape[1]

    for t in range(n_steps):
        true_red = true_full[:, t, :, :, 2]  # Band index 2 = B04 (Red)
        true_nir = true_full[:, t, :, :, 3]  # Band index 3 = B8A (NIR)
        pred_red = pred_full[:, t, :, :, 2]
        pred_nir = pred_full[:, t, :, :, 3]

        def kndvi(nir, red):
            k = np.exp(-(nir - red) ** 2 / (2 * sigma ** 2))
            return (1 - k) / (1 + k + 1e-6)

        true_k = kndvi(true_nir, true_red)
        pred_k = kndvi(pred_nir, pred_red)
        diff = np.abs(true_k - pred_k)
        w = valid[:, t]
        if w.sum() > 0:
            kndvi_by_timestep.append(float((diff * w).sum() / w.sum()))
        else:
            kndvi_by_timestep.append(0.0)
        kndvi_diffs.append(diff[valid[:, t]])

    all_diffs = np.concatenate(kndvi_diffs) if kndvi_diffs else np.array([0.0])
    mean_kndvi_loss = float(all_diffs.mean())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: kNDVI loss distribution
    ax = axes[0]
    bins = np.linspace(0, min(1.0, np.percentile(all_diffs, 99)), 60)
    ax.hist(all_diffs, bins=bins, color="tab:purple", alpha=0.7, density=True)
    ax.axvline(x=mean_kndvi_loss, color="red", linestyle="--",
               label=f"Mean = {mean_kndvi_loss:.4f}")
    ax.set_xlabel("|kNDVI_true - kNDVI_pred|", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("kNDVI Loss Distribution\n(this signal is IGNORED in training)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: kNDVI loss vs timestep
    ax = axes[1]
    days = [(t + 1) * 5 + 50 for t in range(n_steps)]
    ax.plot(days, kndvi_by_timestep, "o-", color="tab:purple", linewidth=2)
    ax.set_xlabel("Forecast day", fontsize=12)
    ax.set_ylabel("Mean |kNDVI loss|", fontsize=12)
    ax.set_title("kNDVI Error vs Forecast Horizon\n(gradients that should be flowing back)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 3: Regression loss vs kNDVI loss comparison
    ax = axes[2]
    # Regression loss using log-cosh
    reg_errors = np.abs(all_true_d[valid[:, np.newaxis, np.newaxis, np.newaxis].repeat(all_true_d.shape[1], axis=1)
                                    if False else ...].ravel()[:len(all_diffs)])
    log_cosh_vals = np.log(np.cosh(np.concatenate([r["true_deltas"] for r in records])[
        np.concatenate([r["mask"] for r in records])[..., 0] < 0.5] -
        np.concatenate([r["pred_deltas"] for r in records])[
        np.concatenate([r["mask"] for r in records])[..., 0] < 0.5]).ravel()
    )

    labels = [
        f"Regression\n(log-cosh)\nmean={np.abs(log_cosh_vals).mean():.4f}",
        f"kNDVI loss\n(IGNORED)\nmean={mean_kndvi_loss:.4f}",
    ]
    vals = [np.abs(log_cosh_vals).mean(), mean_kndvi_loss]
    ax.bar(labels, vals, color=["tab:blue", "tab:purple"], alpha=0.7)
    ax.set_ylabel("Mean loss magnitude", fontsize=12)
    ax.set_title("Regression vs kNDVI Loss Magnitudes\n(kNDVI is non-trivial but disabled)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "14_kndvi_dead_code.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 14_kndvi_dead_code.png")

    return {"mean_kndvi_loss_if_active": mean_kndvi_loss}


# ---------------------------------------------------------------------------
# Plot 15: Target Delta Distribution by Land Cover
# ---------------------------------------------------------------------------

def plot_target_deltas_by_landcover(records, save_dir):
    """
    Show target delta distributions per land cover class.

    Classes with high delta variance (Cropland, Tree cover) should be
    hardest to predict. Near-zero mean deltas suggest BAP baseline is close
    to the actual values, making the model's zero-prediction strategy viable.
    """
    all_true = np.concatenate([r["true_deltas"] for r in records], axis=0)
    all_pred = np.concatenate([r["pred_deltas"] for r in records], axis=0)
    all_mask = np.concatenate([r["mask"] for r in records], axis=0)
    all_lc = np.concatenate([r["landcover"] for r in records], axis=0)

    valid = (1.0 - all_mask[..., 0]).astype(bool)
    lc_class = np.argmax(all_lc, axis=-1)  # (N, H, W)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    lc_stats = {}
    for cls_idx, cls_name in enumerate(LC_NAMES):
        ax = axes[cls_idx]

        # Get valid pixels for this land cover class
        cls_pixel_mask = (lc_class == cls_idx)  # (N, H, W)
        combined_mask = valid * cls_pixel_mask[:, np.newaxis]  # (N, T, H, W) broadcast

        true_vals = all_true[combined_mask].ravel()
        pred_vals = all_pred[combined_mask].ravel()

        if len(true_vals) < 100:
            ax.text(0.5, 0.5, f"Insufficient\ndata\n(n={len(true_vals)})",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(cls_name, fontsize=10, fontweight="bold")
            continue

        # Subsample
        if len(true_vals) > 5000:
            idx = np.random.choice(len(true_vals), 5000, replace=False)
            true_vals = true_vals[idx]
            pred_vals = pred_vals[idx]

        bins = np.linspace(-0.3, 0.3, 50)
        ax.hist(true_vals, bins=bins, alpha=0.5, label="Target", color="tab:blue", density=True)
        ax.hist(pred_vals, bins=bins, alpha=0.5, label="Predicted", color="tab:red", density=True)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        true_mean = float(true_vals.mean())
        true_std = float(true_vals.std())
        pred_std = float(pred_vals.std())
        ax.set_title(f"{cls_name}\nμ={true_mean:.3f} σ_t={true_std:.3f} σ_p={pred_std:.3f}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        lc_stats[cls_name] = {
            "true_mean": true_mean, "true_std": true_std,
            "pred_std": pred_std, "n_pixels": len(true_vals)
        }

    fig.suptitle("Target Delta Distributions by Land Cover Class\n"
                 "(μ near zero = BAP baseline is close to targets = zero-prediction viable for the model)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "15_target_delta_by_lc.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 15_target_delta_by_lc.png")
    return lc_stats


# ---------------------------------------------------------------------------
# Plot 16: Cloud Cover Analysis
# ---------------------------------------------------------------------------

def plot_cloud_cover_analysis(records, save_dir):
    """
    Analyse cloud fraction in the target sequence.

    Input cloudmask is no longer available since the encoder uses BAP-filled
    sequences without CloudAwareGating. Only the target clear-sky mask (used
    to gate the regression loss) is analysed here.
    """
    all_mask_out = np.concatenate([r["mask"] for r in records], axis=0)
    # shape: (N, T_out, H, W, 1)

    target_cloud_per_step = all_mask_out[..., 0].mean(axis=(0, 2, 3))  # (T_out,)
    target_cloud_per_sample = all_mask_out[..., 0].mean(axis=(1, 2, 3))  # (N,)

    # Check whether input cloudmask records exist (legacy support)
    has_cloud_in = records[0].get("cloudmask_in") is not None

    if has_cloud_in:
        all_cloud_in = np.concatenate([r["cloudmask_in"] for r in records], axis=0)
        cloud_frac_per_step = all_cloud_in[..., 0].mean(axis=(0, 2, 3))
        cloud_frac_per_sample = all_cloud_in[..., 0].mean(axis=(1, 2, 3))
        n_cols = 3
    else:
        n_cols = 2

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    col = 0

    if has_cloud_in:
        # Panel: Cloud fraction per input timestep
        ax = axes[col]; col += 1
        t_in = [(t + 1) * 5 + 4 for t in range(all_cloud_in.shape[1])]
        ax.bar(t_in, cloud_frac_per_step, color="tab:gray", alpha=0.7)
        ax.axhline(y=0.5, color="red", linestyle="--", label="50% cloud cover")
        ax.set_xlabel("Day of input sequence", fontsize=12)
        ax.set_ylabel("Mean cloud fraction", fontsize=12)
        ax.set_title("Cloud Cover in Input Sequence", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel: Cloud fraction distribution per sample
        ax = axes[col]; col += 1
        ax.hist(cloud_frac_per_sample, bins=30, color="tab:blue", alpha=0.7)
        ax.axvline(x=cloud_frac_per_sample.mean(), color="red", linestyle="--",
                   label=f"Mean = {cloud_frac_per_sample.mean():.2f}")
        ax.set_xlabel("Mean cloud fraction per sample", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Input Cloud Cover\nAcross Samples", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        # Panel: N/A note for input cloudmask
        ax = axes[col]; col += 1
        ax.text(0.5, 0.5,
                "Input cloudmask not available.\nEncoder uses BAP-filled sequences\n(CloudAwareGating removed).",
                ha="center", va="center", fontsize=13, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))
        ax.set_title("Cloud Cover in Input Sequence", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Panel: Target cloud fraction per output step
    ax = axes[col]
    t_out = [(t + 1) * 5 + 54 for t in range(all_mask_out.shape[1])]
    ax.bar(t_out, target_cloud_per_step, color="tab:orange", alpha=0.7)
    ax.axhline(y=0.5, color="red", linestyle="--", label="50% cloud cover")
    ax.set_xlabel("Forecast day", fontsize=12)
    ax.set_ylabel("Mean clear-sky fraction", fontsize=12)
    ax.set_title("Clear-sky Fraction in Target Sequence\n"
                 "(regression loss computed on clear pixels only)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "16_cloud_cover_analysis.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 16_cloud_cover_analysis.png")

    result = {
        "mean_target_clear_frac": float(target_cloud_per_step.mean()),
    }
    if has_cloud_in:
        result["mean_input_cloud_frac"] = float(cloud_frac_per_sample.mean())
        result["max_input_cloud_frac"] = float(cloud_frac_per_sample.max())
    return result


# ---------------------------------------------------------------------------
# Plot 17: Fourier Basis Visualization
# ---------------------------------------------------------------------------

def plot_fourier_basis(records, save_dir, n_harmonics=6):
    """
    Visualise the Fourier design matrix for representative DOY values.

    Shows what temporal patterns the basis can express, and demonstrates
    that without a DC offset, the basis sums to zero over a full cycle.
    """
    target_doys = [r["target_doy"].ravel()[0] for r in records]
    example_doys = [float(d) for d in target_doys[:3]]  # Use first 3 samples
    # Also show a canonical summer / winter case
    example_doys = sorted(set([0.1, 0.4, 0.7] + example_doys[:2]))[:5]

    T = 20
    step_interval = 5
    omega = 2 * np.pi / 365.0

    fig, axes = plt.subplots(len(example_doys), 2, figsize=(16, len(example_doys) * 3))
    if len(example_doys) == 1:
        axes = axes[np.newaxis, :]

    for row, doy_norm in enumerate(example_doys):
        start_day = doy_norm * 365.0
        days = np.arange(1, T + 1) * step_interval  # [5, 10, ..., 100]
        doy_t = start_day + days  # absolute DOY for each forecast step

        # Build basis matrix (T, 2K)
        phi = np.zeros((T, 2 * n_harmonics))
        for k in range(n_harmonics):
            freq = (k + 1) * omega
            phi[:, 2 * k] = np.cos(freq * doy_t)
            phi[:, 2 * k + 1] = np.sin(freq * doy_t)

        # Panel 1: Basis functions per harmonic
        ax = axes[row, 0]
        for k in range(min(n_harmonics, 4)):  # Show first 4 harmonics
            ax.plot(days, phi[:, 2 * k], "-", label=f"cos k={k+1}", alpha=0.8)
            ax.plot(days, phi[:, 2 * k + 1], "--", label=f"sin k={k+1}", alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Forecast offset (days)")
        ax.set_ylabel("Basis value")
        ax.set_title(f"Fourier Basis (start DOY≈{int(doy_norm*365)})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Sum of all basis functions (illustrates DC offset absence)
        ax = axes[row, 1]
        basis_sum = phi.sum(axis=1)  # Sum of all K cosines + K sines
        ax.plot(days, basis_sum, "k-", linewidth=2, label="Sum of all basis")
        ax.fill_between(days, 0, basis_sum, alpha=0.2)
        ax.axhline(y=0, color="red", linestyle="--", label="Zero line (DC offset)")
        ax.set_xlabel("Forecast offset (days)")
        ax.set_ylabel("Summed basis value")
        ax.set_title(f"Basis Sum (start DOY≈{int(doy_norm*365)})\nA constant offset would allow non-zero mean predictions",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fourier Basis Functions for Sample DOYs\n"
                 "(Note: no DC term means basis cannot represent sustained trends)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "17_fourier_basis.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 17_fourier_basis.png")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def write_collapse_summary(grad_stats, harmonic_stats, diversity_stats, l2_stats,
                            kndvi_stats, cloud_stats, save_dir):
    lines = []
    lines.append("=" * 70)
    lines.append("  MODE COLLAPSE PARAMETER ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1. GRADIENT FLOW")
    lines.append(f"   Proxy regression loss (MAE on clear pixels): {grad_stats.get('proxy_loss', 0):.5f}")
    dead = grad_stats.get("dead_layers", [])
    weak = grad_stats.get("weak_layers", [])
    lines.append(f"   Dead layers (grad norm < 1e-5): {len(dead)}")
    for l in dead:
        lines.append(f"     - {l}")
    lines.append(f"   Weak layers (grad norm < 1e-3): {len(weak)}")
    for l in weak[:5]:
        lines.append(f"     - {l}")
    lines.append("")

    lines.append("2. FOURIER COEFFICIENT HEALTH")
    if harmonic_stats:
        for k in range(1, 7):
            a_std = harmonic_stats.get(f"k{k}_a_std", 0)
            b_std = harmonic_stats.get(f"k{k}_b_std", 0)
            amp = harmonic_stats.get(f"k{k}_amplitude", 0)
            status = "OK" if amp > 0.01 else "NEAR-ZERO"
            lines.append(f"   Harmonic k={k}: a_std={a_std:.4f}, b_std={b_std:.4f}, "
                         f"mean_amplitude={amp:.4f}  [{status}]")
    lines.append("")

    lines.append("3. MODE COLLAPSE INDICATORS")
    dr = diversity_stats.get("diversity_ratio", 0)
    lines.append(f"   Prediction std / Target std: {dr:.4f}  (1.0 = good, << 1.0 = collapsed)")
    lines.append(f"   Mean prediction std: {diversity_stats.get('mean_pred_std', 0):.5f}")
    lines.append(f"   Mean target std:     {diversity_stats.get('mean_true_std', 0):.5f}")
    ratios = diversity_stats.get("per_band_ratios", [])
    for b, r in enumerate(ratios):
        lines.append(f"   Band {b} diversity ratio: {r:.4f}")
    lines.append("")

    lines.append("4. L2 REGULARIZATION vs REGRESSION LOSS")
    lines.append(f"   Raw regression loss (log-cosh): {l2_stats.get('raw_reg_loss', 0):.6f}")
    lines.append(f"   Weighted regression (×5.0):     {l2_stats.get('weighted_reg_loss', 0):.6f}")
    lines.append(f"   Total L2 regularization:        {l2_stats.get('keras_total_l2', 0):.6f}")
    ratio = l2_stats.get("l2_to_regression_ratio", 0)
    lines.append(f"   L2/Regression ratio:            {ratio:.4f}  "
                 f"({'L2 DOMINATES' if ratio > 0.1 else 'OK'})")
    lines.append("")

    lines.append("5. kNDVI AUXILIARY LOSS (DEAD CODE)")
    lines.append(f"   Mean kNDVI loss if active: {kndvi_stats.get('mean_kndvi_loss_if_active', 0):.5f}")
    lines.append("   Status: kNDVI computation is MISSING from ImprovedkNDVILoss.call()")
    lines.append("   The kndvi_weight tf.Variable is never referenced in call().")
    lines.append("")

    lines.append("6. CLOUD COVER")
    if "mean_input_cloud_frac" in cloud_stats:
        lines.append(f"   Mean input cloud fraction:  {cloud_stats['mean_input_cloud_frac']:.3f}")
        lines.append(f"   Max input cloud fraction:   {cloud_stats['max_input_cloud_frac']:.3f}")
    else:
        lines.append("   Input cloud fraction: N/A (encoder uses BAP-filled sequences, no cloudmask input)")
    target_clear = cloud_stats.get("mean_target_clear_frac", cloud_stats.get("mean_target_cloud_frac", 0))
    lines.append(f"   Mean target clear-sky fraction: {target_clear:.3f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  ROOT CAUSES (in priority order)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  [BUG-1] kNDVI auxiliary loss is dead code in ImprovedkNDVILoss.call().")
    lines.append("          The kNDVI computation block is missing from the call() method.")
    lines.append("          FIX: Add kNDVI loss computation and apply kndvi_weight in call().")
    lines.append("")
    lines.append("  [BUG-2] L2 regularization on all FourierCoefficientHead conv layers")
    lines.append("          drives Fourier coefficients toward zero (mode collapse).")
    lines.append("          FIX: Remove L2 from FourierCoefficientHead or reduce lambda to 1e-6.")
    lines.append("")
    lines.append("  [ARCH-1] No DC offset in Fourier representation.")
    lines.append("           Cannot represent sustained trends (monotonic increase/decrease).")
    lines.append("           FIX: Add a learnable bias per pixel per band as a DC term.")
    lines.append("")
    lines.append("  [ARCH-2] CloudAwareGating zeros BAP-filled values for cloudy timesteps,")
    lines.append("           making the encoder input very sparse when cloud cover is high.")
    lines.append("           FIX: Apply gating only to S2 channels, not to landcover.")
    lines.append("           Or use the BAP-filled inputs without gating (BAP is already clean).")
    lines.append("")
    lines.append("  [ARCH-3] Weather features collapsed to a single global context vector.")
    lines.append("           Per-timestep weather anomalies cannot modulate per-step predictions.")
    lines.append("           FIX: Use weather as per-timestep additive correction on output.")
    lines.append("")
    lines.append("  [TRAIN-1] gradient clipnorm=1.0 may be too aggressive for a 3-layer")
    lines.append("            ConvLSTM. Signals may not reach the first encoder layer.")
    lines.append("            FIX: Increase clipnorm to 5.0 or use clipvalue instead.")
    lines.append("")

    report = "\n".join(lines)
    path = os.path.join(save_dir, "18_collapse_summary.txt")
    with open(path, "w") as f:
        f.write(report)
    print("  ✓ 18_collapse_summary.txt")
    print()
    print(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_parameters(model, generator, split="val_chopped", max_samples=20):
    """Run parameter analysis with a pre-loaded model and generator."""
    os.makedirs(DIAG_DIR, exist_ok=True)

    dataset = generator.get_dataset()
    dataset = dataset.batch(1).map(adapt_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"\nCollecting predictions ({max_samples} samples) ...")
    records = collect_data(model, dataset, max_samples)
    print(f"  Collected: {len(records)} samples")

    print("\nGenerating diagnostic plots ...")
    grad_stats = plot_gradient_norms(model, dataset, DIAG_DIR)
    harmonic_stats = plot_per_harmonic_coefficients(records, model, DIAG_DIR)
    diversity_stats = plot_prediction_diversity(records, DIAG_DIR)
    l2_stats = plot_l2_vs_regression_loss(model, dataset, DIAG_DIR)
    kndvi_stats = plot_kndvi_dead_code(records, DIAG_DIR)
    lc_stats = plot_target_deltas_by_landcover(records, DIAG_DIR)
    cloud_stats = plot_cloud_cover_analysis(records, DIAG_DIR)
    plot_fourier_basis(records, DIAG_DIR)

    print("\nWriting summary ...")
    write_collapse_summary(grad_stats, harmonic_stats, diversity_stats, l2_stats,
                           kndvi_stats, cloud_stats, DIAG_DIR)

    print(f"\nAll diagnostics saved to: {DIAG_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Parameter analysis for mode collapse diagnosis.")
    parser.add_argument("--split", type=str, default="val_chopped",
                        choices=["train", "val_chopped", "val"])
    parser.add_argument("--max_samples", type=int, default=20,
                        help="Number of samples to use for analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: final_model.keras)")
    args = parser.parse_args()

    os.makedirs(DIAG_DIR, exist_ok=True)

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

    analyze_parameters(model, generator, split=args.split, max_samples=args.max_samples)

    print(f"\nAll diagnostics saved to: {DIAG_DIR}")


if __name__ == "__main__":
    main()
