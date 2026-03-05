#!/usr/bin/env python3
"""
Diagnostic script for delta signal quality.

Analyses how prediction error relates to delta magnitude, time step,
land-cover class, and spectral band.

Outputs:
  images/diagnostics/
    01_error_vs_time.png       – MAE per timestep by land-cover class
    02_amplitude_histogram.png – |target| vs |predicted| delta distributions
    03_coefficient_analysis.png – Fourier coefficient & coeff_scale analysis
    04_per_band_scatter.png    – True vs predicted deltas per band
    06_summary.txt             – Quantitative summary

Usage:
    conda run -n tf-gpu python scripts/diagnose_signal.py --split val_chopped --max_samples 30
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
from adapt_inputs import adapt_inputs

# Output directory
DIAG_DIR = os.path.join(PROJECT_ROOT, "images", "diagnostics")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.keras")

# ESA WorldCover class names (indices 0-9, matching one-hot after dropping class 10)
LC_NAMES = [
    "Tree cover",      # 20
    "Shrubland",       # 30
    "Grassland",       # 40
    "Cropland",        # 50
    "Built-up",        # 60
    "Bare/sparse",     # 70
    "Snow/Ice",        # 80
    "Water",           # 90
    "Wetland",         # 95
    "Mangrove",        # 100
]

# Highlight classes: cropland (idx 3 = class 50) and tree cover (idx 0 = class 20)
HIGHLIGHT_CLASSES = {
    "Cropland": 3,
    "Tree cover": 0,
    "Grassland": 2,
}

BAND_NAMES = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B8A (NIR)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_predictions(model, dataset, max_samples):
    """Run model inference and collect arrays."""
    all_true_deltas = []
    all_pred_deltas = []
    all_masks = []
    all_landcover = []

    count = 0
    for x_batch, y_batch in dataset:
        pred = model(x_batch, training=False)

        # y_batch: (B, T, H, W, 9) -> mask(1) + deltas(4) + bap(4)
        mask = y_batch[..., 0:1].numpy()       # (B, T, H, W, 1)
        true_d = y_batch[..., 1:5].numpy()     # (B, T, H, W, 4)
        pred_d = pred.numpy()                   # (B, T, H, W, 4)
        lc = x_batch["landcover_map"].numpy()   # (B, H, W, 10)

        all_true_deltas.append(true_d)
        all_pred_deltas.append(pred_d)
        all_masks.append(mask)
        all_landcover.append(lc)

        count += true_d.shape[0]
        if max_samples and count >= max_samples:
            break

    true_d = np.concatenate(all_true_deltas, axis=0)
    pred_d = np.concatenate(all_pred_deltas, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    lcs = np.concatenate(all_landcover, axis=0)

    return true_d, pred_d, masks, lcs


# ---------------------------------------------------------------------------
# Plot 1: Error vs Time
# ---------------------------------------------------------------------------

def plot_error_vs_time(true_d, pred_d, masks, lcs, save_dir):
    """MAE per timestep, overall and per highlighted land-cover class."""
    n_steps = true_d.shape[1]
    valid = (1.0 - masks[..., 0])  # (N, T, H, W), 1=clear

    # Dominant land-cover per pixel
    lc_class = np.argmax(lcs, axis=-1)  # (N, H, W)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Overall MAE per timestep ---
    ax = axes[0]
    mae_per_step = []
    for t in range(n_steps):
        w = valid[:, t]
        err = np.abs(true_d[:, t] - pred_d[:, t]).mean(axis=-1)  # avg over bands
        if w.sum() > 0:
            mae_per_step.append((err * w).sum() / w.sum())
        else:
            mae_per_step.append(0)
    days = [(t + 1) * 5 + 50 for t in range(n_steps)]
    ax.plot(days, mae_per_step, "o-", color="tab:blue", linewidth=2, markersize=6)
    ax.set_xlabel("Day from sequence start", fontsize=12)
    ax.set_ylabel("MAE (reflectance delta)", fontsize=12)
    ax.set_title("Overall MAE vs Forecast Horizon", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- Right: Per land-cover class ---
    ax = axes[1]
    colors = {"Cropland": "tab:orange", "Tree cover": "tab:green", "Grassland": "tab:olive"}
    for cls_name, cls_idx in HIGHLIGHT_CLASSES.items():
        mae_cls = []
        for t in range(n_steps):
            cls_mask = (lc_class == cls_idx).astype(np.float32)  # (N, H, W)
            w = valid[:, t] * cls_mask
            err = np.abs(true_d[:, t] - pred_d[:, t]).mean(axis=-1)
            if w.sum() > 0:
                mae_cls.append((err * w).sum() / w.sum())
            else:
                mae_cls.append(0)
        ax.plot(days, mae_cls, "o-", label=cls_name, color=colors[cls_name],
                linewidth=2, markersize=5)
    ax.set_xlabel("Day from sequence start", fontsize=12)
    ax.set_ylabel("MAE (reflectance delta)", fontsize=12)
    ax.set_title("MAE vs Forecast Horizon by Land Cover", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "01_error_vs_time.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 01_error_vs_time.png")
    return mae_per_step


# ---------------------------------------------------------------------------
# Plot 2: Amplitude Histogram
# ---------------------------------------------------------------------------

def plot_amplitude_histogram(true_d, pred_d, masks, save_dir):
    """Distribution of |true delta| vs |pred delta| for clear pixels."""
    valid = (1.0 - masks[..., 0]).astype(bool)  # (N, T, H, W)

    true_abs = np.abs(true_d[valid])   # all valid pixels, all bands
    pred_abs = np.abs(pred_d[valid])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overlaid histogram
    ax = axes[0]
    bins = np.linspace(0, 0.4, 80)
    ax.hist(true_abs.ravel(), bins=bins, alpha=0.55, label="Target |δ|",
            color="tab:blue", density=True)
    ax.hist(pred_abs.ravel(), bins=bins, alpha=0.55, label="Predicted |δ|",
            color="tab:red", density=True)
    ax.set_xlabel("|Delta| (reflectance units)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Delta Amplitude Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Quantile comparison
    ax = axes[1]
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    true_q = np.quantile(true_abs, quantiles)
    pred_q = np.quantile(pred_abs, quantiles)
    x_pos = np.arange(len(quantiles))
    w = 0.35
    ax.bar(x_pos - w / 2, true_q, w, label="Target", color="tab:blue", alpha=0.7)
    ax.bar(x_pos + w / 2, pred_q, w, label="Predicted", color="tab:red", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"P{int(q*100)}" for q in quantiles])
    ax.set_ylabel("|Delta|", fontsize=12)
    ax.set_title("Amplitude Quantile Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate ratio
    for i in range(len(quantiles)):
        if pred_q[i] > 0:
            ratio = true_q[i] / pred_q[i]
            ax.annotate(f"{ratio:.1f}×", (x_pos[i], max(true_q[i], pred_q[i])),
                       ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "02_amplitude_histogram.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 02_amplitude_histogram.png")

    return {
        "true_quantiles": dict(zip(quantiles, true_q)),
        "pred_quantiles": dict(zip(quantiles, pred_q)),
        "true_mean_abs": float(true_abs.mean()),
        "pred_mean_abs": float(pred_abs.mean()),
    }


# ---------------------------------------------------------------------------
# Plot 3: Coefficient Analysis
# ---------------------------------------------------------------------------

def plot_coefficient_analysis(model, save_dir):
    """Analyse Fourier coefficient conv kernel weights."""
    # Extract the FourierCoefficientHead layer
    head = None
    for layer in model.layers:
        if "fourier_head" in layer.name:
            head = layer
            break
    if head is None:
        print("  ⚠ Could not find fourier_head layer — skipping coefficient analysis.")
        return {}

    n_bands = head.n_bands
    n_coeffs = head.n_coeffs

    # Get the final conv layer weights
    conv_weights = head.conv_coeffs.get_weights()
    kernel = conv_weights[0]  # (1, 1, hidden, n_bands*2K)
    bias = conv_weights[1] if len(conv_weights) > 1 else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Distribution of conv_coeffs kernel weights
    ax = axes[0]
    ax.hist(kernel.ravel(), bins=60, color="tab:purple", alpha=0.7)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.set_title("Coefficient Conv Kernel Distribution", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-band kernel weight std (proxy for amplitude capacity)
    ax = axes[1]
    # kernel shape: (1, 1, hidden, n_bands * n_coeffs) — reshape to per-band
    kernel_flat = kernel.reshape(-1, n_bands, n_coeffs)
    band_std = np.std(kernel_flat, axis=(0, 2))
    ax.bar(range(n_bands), band_std, color=[
        "tab:blue", "tab:green", "tab:red", "tab:brown"], alpha=0.7)
    ax.set_xticks(range(n_bands))
    ax.set_xticklabels([b.split("(")[1].rstrip(")") for b in BAND_NAMES])
    ax.set_ylabel("Kernel weight std")
    ax.set_title("Per-Band Coefficient Kernel Spread",
                fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "03_coefficient_analysis.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 03_coefficient_analysis.png")

    return {
        "kernel_std": float(kernel.std()),
    }


# ---------------------------------------------------------------------------
# Plot 4: Per-band Scatter
# ---------------------------------------------------------------------------

def plot_per_band_scatter(true_d, pred_d, masks, save_dir, max_points=50000):
    """2D scatter / hexbin: true delta vs predicted delta per band."""
    valid = (1.0 - masks[..., 0]).astype(bool)  # (N, T, H, W)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ["Blues", "Greens", "Reds", "YlOrBr"]

    slope_info = {}
    for b in range(4):
        ax = axes[b]
        t_vals = true_d[valid][..., b].ravel()
        p_vals = pred_d[valid][..., b].ravel()

        # Subsample for hexbin
        if len(t_vals) > max_points:
            idx = np.random.choice(len(t_vals), max_points, replace=False)
            t_vals = t_vals[idx]
            p_vals = p_vals[idx]

        ax.hexbin(t_vals, p_vals, gridsize=60, cmap=colors[b], mincnt=1)
        lim = max(np.abs(t_vals).max(), np.abs(p_vals).max()) * 1.05
        lim = min(lim, 0.5)  # cap for readability
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True delta")
        ax.set_ylabel("Pred delta")
        ax.set_title(BAND_NAMES[b], fontsize=13, fontweight="bold")
        ax.set_aspect("equal")

        # Linear fit — slope < 1 indicates under-prediction
        if len(t_vals) > 10:
            slope = np.polyfit(t_vals, p_vals, 1)[0]
            ax.text(0.05, 0.92, f"slope = {slope:.3f}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            slope_info[BAND_NAMES[b]] = float(slope)

    fig.suptitle("True vs Predicted Deltas (slope < 1 = under-prediction)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "04_per_band_scatter.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_per_band_scatter.png")
    return slope_info


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def write_summary(mae_per_step, amp_stats, coeff_stats, slope_info, save_dir):
    """Write a text summary with quantitative findings and recommendations."""
    lines = []
    lines.append("=" * 70)
    lines.append("  WEAK SIGNAL DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append("")

    # 1. Error vs time
    lines.append("1. ERROR VS FORECAST HORIZON")
    lines.append(f"   MAE at t=1 (day 55):  {mae_per_step[0]:.5f}")
    lines.append(f"   MAE at t=20 (day 150): {mae_per_step[-1]:.5f}")
    if mae_per_step[0] > 0:
        lines.append(f"   Growth ratio (last/first): {mae_per_step[-1]/mae_per_step[0]:.2f}×")
    lines.append("")

    # 2. Amplitude gap
    lines.append("2. AMPLITUDE GAP")
    lines.append(f"   Mean |target delta|:    {amp_stats['true_mean_abs']:.5f}")
    lines.append(f"   Mean |predicted delta|: {amp_stats['pred_mean_abs']:.5f}")
    if amp_stats['pred_mean_abs'] > 0:
        gap_ratio = amp_stats['true_mean_abs'] / amp_stats['pred_mean_abs']
        lines.append(f"   Under-prediction ratio: {gap_ratio:.2f}× (target is {gap_ratio:.1f}× larger)")
    lines.append("   Quantile comparison (target / predicted):")
    for q in amp_stats["true_quantiles"]:
        tq = amp_stats["true_quantiles"][q]
        pq = amp_stats["pred_quantiles"][q]
        ratio_str = f"{tq/pq:.2f}×" if pq > 0 else "∞"
        lines.append(f"     P{int(q*100):02d}: {tq:.5f} / {pq:.5f}  = {ratio_str}")
    lines.append("")

    # 3. Coefficient analysis
    if coeff_stats:
        lines.append("3. FOURIER COEFFICIENT ANALYSIS")
        lines.append(f"   Conv kernel weight std:   {coeff_stats['kernel_std']:.4f}")
        lines.append("")

    # 4. Per-band slopes
    if slope_info:
        lines.append("4. PER-BAND REGRESSION SLOPES (1.0 = perfect, <1 = under-prediction)")
        for band, slope in slope_info.items():
            status = "OK" if slope > 0.8 else "UNDER-PREDICTING" if slope > 0.3 else "SEVERELY UNDER-PREDICTING"
            lines.append(f"   {band}: slope = {slope:.3f}  [{status}]")
        lines.append("")

    # 5. Assessment
    lines.append("=" * 70)
    lines.append("  ASSESSMENT")
    lines.append("=" * 70)
    lines.append("")
    all_slopes = list(slope_info.values()) if slope_info else []
    mean_slope = np.mean(all_slopes) if all_slopes else 0
    if mean_slope > 0.85:
        lines.append("STATUS: Signal strength looks GOOD (mean slope > 0.85).")
    elif mean_slope > 0.6:
        lines.append("STATUS: Mild under-prediction (mean slope {:.2f}). May improve with".format(mean_slope))
        lines.append("        more training or learning rate adjustment.")
    else:
        lines.append("STATUS: SIGNIFICANT under-prediction (mean slope {:.2f}).".format(mean_slope))
        lines.append("        Consider changes to loss function or coefficient scaling.")
    lines.append("")

    report = "\n".join(lines)

    path = os.path.join(save_dir, "06_summary.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"  ✓ 06_summary.txt")
    print()
    print(report)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def diagnose(model, generator, split="val_chopped", max_samples=30):
    """Run signal diagnostics with a pre-loaded model and generator."""
    os.makedirs(DIAG_DIR, exist_ok=True)

    dataset = generator.get_dataset()
    dataset = dataset.batch(1).map(adapt_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"\nRunning inference (max {max_samples} samples) ...")
    true_d, pred_d, masks, lcs = collect_predictions(model, dataset, max_samples)
    print(f"  Collected: {true_d.shape[0]} samples, shape={true_d.shape}")

    print("\nGenerating diagnostic plots ...")
    mae_per_step = plot_error_vs_time(true_d, pred_d, masks, lcs, DIAG_DIR)
    amp_stats = plot_amplitude_histogram(true_d, pred_d, masks, DIAG_DIR)
    coeff_stats = plot_coefficient_analysis(model, DIAG_DIR)
    slope_info = plot_per_band_scatter(true_d, pred_d, masks, DIAG_DIR)

    print("\nWriting summary report ...")
    write_summary(mae_per_step, amp_stats, coeff_stats, slope_info, DIAG_DIR)

    print("\nAll diagnostics saved to:", DIAG_DIR)


def main():
    parser = argparse.ArgumentParser(description="Diagnose weak delta signal strength.")
    parser.add_argument("--split", type=str, default="val_chopped",
                        choices=["train", "val_chopped", "val"])
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: final_model.keras)")
    args = parser.parse_args()

    os.makedirs(DIAG_DIR, exist_ok=True)

    # --- GPU config ---
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # --- Load model ---
    model_path = args.model or MODEL_PATH
    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, compile=False)
    print(f"  Parameters: {model.count_params():,}")

    # --- Load dataset ---
    data_path = VALIDATION_PATH if args.split in ("val_chopped", "val") else DATASET_PATH
    print(f"Loading dataset from {data_path} (split={args.split}) ...")
    generator = DatasetGenerator(data_path)

    diagnose(model, generator, split=args.split, max_samples=args.max_samples)

    print("\n✅ All diagnostics saved to:", DIAG_DIR)


if __name__ == "__main__":
    main()
