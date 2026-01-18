#!/usr/bin/env python3
"""
Export figures and tables for academic paper publication.

Generates publication-quality assets for:
- Materials and Data section (dataset statistics, distributions)
- Results and Discussion section (predictions, error metrics)

Usage:
    python export_paper_assets.py [--max_samples N]
"""

import os
# Force CPU to avoid XLA/CUDA JIT errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import tensorflow as tf

from dataset import DatasetGenerator
from build_model import load_model
from config import DATASET_PATH, VALIDATION_PATH

# =============================================================================
# Configuration
# =============================================================================

CHECKPOINTS_DIR = "/home/me/workspace/bspline_ndvi/checkpoints"
PROJECT_ROOT = "/home/me/workspace/bspline_ndvi"
MATERIAL_DIR = os.path.join(PROJECT_ROOT, "images/material")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "images/results")

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Band names for labeling
BAND_NAMES = ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B8A (NIR)']
BAND_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

# ESA WorldCover classes (indices 1-10 after removing class 0)
LANDCOVER_CLASSES = [
    'Tree cover', 'Shrubland', 'Grassland', 'Cropland', 
    'Built-up', 'Bare/sparse', 'Snow/ice', 'Water',
    'Wetland', 'Mangroves'
]
LANDCOVER_COLORS = [
    '#006400', '#8B4513', '#90EE90', '#FFD700',
    '#FF0000', '#D2B48C', '#FFFFFF', '#0000FF',
    '#00CED1', '#228B22'
]

# E-OBS weather variable names
WEATHER_VARS = ['Temp Mean', 'Humidity', 'Pressure', 'Radiation', 
                'Precipitation', 'Temp Min', 'Temp Max']


# =============================================================================
# Helper Functions
# =============================================================================

def adapt_inputs(x, y):
    """Transform dataset inputs to model-expected format.
    
    Returns tuple of (x_new, y, landcover) where landcover is kept
    separate for analysis.
    """
    t_meta = x['time'][:, -1, :]
    landcover = x['landcover']
    x_new = {
        'sentinel2_sequence': x['sentinel2'],
        'cloudmask_sequence': x['cloudmask'],
        'landcover_map': landcover,
        'weather_sequence': x['weather'],
        'temporal_metadata': t_meta
    }
    return x_new, y, landcover


def make_rgb(data, clip_min=0.0, clip_max=0.3):
    """Create RGB image from Sentinel-2 bands. 
    
    Args:
        data: (H, W, 4) array with bands [B02, B03, B04, B8A]
    
    Returns:
        (H, W, 3) RGB array normalized to [0, 1]
    """
    rgb = np.stack([
        data[:, :, 2],  # R = B04
        data[:, :, 1],  # G = B03
        data[:, :, 0],  # B = B02
    ], axis=-1)
    rgb = np.clip((rgb - clip_min) / (clip_max - clip_min), 0, 1)
    return rgb


def make_delta_rgb(data, vmin=-0.1, vmax=0.1):
    """Create RGB visualization for delta values.
    
    Args:
        data: (H, W, 4) delta array
        
    Returns:
        (H, W, 3) RGB array with deltas mapped to color
    """
    rgb = np.stack([
        data[:, :, 2],  # R = B04 delta
        data[:, :, 1],  # G = B03 delta
        data[:, :, 0],  # B = B02 delta
    ], axis=-1)
    rgb = np.clip((rgb - vmin) / (vmax - vmin), 0, 1)
    return rgb


def compute_kndvi(nir, red, sigma=0.5):
    """Compute kernel NDVI using RBF kernel.
    
    kNDVI = (1 - k(nir, red)) / (1 + k(nir, red))
    where k(a, b) = exp(-(a-b)^2 / (2*sigma^2))
    """
    diff_sq = (nir - red) ** 2
    k = np.exp(-diff_sq / (2 * sigma ** 2))
    kndvi = (1 - k) / (1 + k + 1e-8)
    return kndvi


# =============================================================================
# Materials and Data Exports
# =============================================================================

def collect_dataset_statistics(generator, max_samples=50):
    """Collect statistics from dataset samples.
    
    Returns dict with arrays for histograms and statistics.
    """
    stats = {
        'spectral_values': [],      # All reflectance values per band
        'cloud_fractions': [],       # Cloud coverage per sample
        'landcover_counts': [],      # One-hot landcover per sample
        'weather_values': [],        # Weather features
        'delta_values': [],          # Target delta values per band
        'sample_count': 0,
    }
    
    dataset = generator.get_dataset()
    
    for i, (x, y) in enumerate(dataset):
        if i >= max_samples:
            break
            
        # Sentinel-2 reflectance (T, H, W, 4)
        sentinel2 = x['sentinel2'].numpy()
        cloudmask = x['cloudmask'].numpy()
        landcover = x['landcover'].numpy()
        weather = x['weather'].numpy()
        
        # Y: (T, H, W, 9) = [mask(1), deltas(4), BAP(4)]
        y_data = y.numpy()
        deltas = y_data[:, :, :, 1:5]
        
        # Collect spectral values (subsample to reduce memory)
        stats['spectral_values'].append(sentinel2[:, ::4, ::4, :].reshape(-1, 4))
        
        # Cloud fraction per timestep
        cloud_frac = np.mean(cloudmask)
        stats['cloud_fractions'].append(cloud_frac)
        
        # Landcover (sum one-hot to get class counts)
        stats['landcover_counts'].append(landcover.sum(axis=(0, 1)))
        
        # Weather (flatten all timesteps)
        stats['weather_values'].append(weather.reshape(-1, 21))
        
        # Delta values (subsample)
        stats['delta_values'].append(deltas[:, ::4, ::4, :].reshape(-1, 4))
        
        stats['sample_count'] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{max_samples} samples...")
    
    # Concatenate all arrays
    stats['spectral_values'] = np.concatenate(stats['spectral_values'], axis=0)
    stats['landcover_counts'] = np.stack(stats['landcover_counts'], axis=0).sum(axis=0)
    stats['weather_values'] = np.concatenate(stats['weather_values'], axis=0)
    stats['delta_values'] = np.concatenate(stats['delta_values'], axis=0)
    
    return stats


def export_dataset_statistics_table(train_stats, val_stats, output_path):
    """Export dataset statistics to CSV."""
    data = {
        'Metric': [
            'Number of Samples',
            'Input Sequence Length',
            'Output Sequence Length',
            'Image Size',
            'Spectral Bands',
            'Mean Cloud Coverage (%)',
            'Reflectance Range (Mean)',
        ],
        'Training': [
            train_stats['sample_count'],
            10,
            20,
            '128×128',
            4,
            f"{np.mean(train_stats['cloud_fractions']) * 100:.1f}",
            f"[{train_stats['spectral_values'].mean():.4f}]",
        ],
        'Validation': [
            val_stats['sample_count'],
            10,
            20,
            '128×128',
            4,
            f"{np.mean(val_stats['cloud_fractions']) * 100:.1f}",
            f"[{val_stats['spectral_values'].mean():.4f}]",
        ],
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return df


def export_spectral_distribution(stats, output_path):
    """Create histogram of spectral band distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, (ax, name, color) in enumerate(zip(axes, BAND_NAMES, BAND_COLORS)):
        values = stats['spectral_values'][:, i]
        # Filter to reasonable range and subsample
        values = values[(values >= 0) & (values <= 0.5)]
        values = values[::10]  # Subsample for faster plotting
        
        ax.hist(values, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_xlabel('Reflectance')
        ax.set_ylabel('Frequency')
        ax.set_title(name)
        ax.set_xlim(0, 0.5)
        
        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'μ={mean_val:.3f}')
        ax.legend(loc='upper right')
    
    plt.suptitle('Spectral Band Reflectance Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def export_cloud_coverage(train_stats, val_stats, output_path):
    """Create cloud coverage distribution chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bin cloud fractions
    bins = np.linspace(0, 1, 11)
    train_hist, _ = np.histogram(train_stats['cloud_fractions'], bins=bins)
    val_hist, _ = np.histogram(val_stats['cloud_fractions'], bins=bins)
    
    x = np.arange(len(bins) - 1)
    width = 0.35
    
    ax.bar(x - width/2, train_hist, width, label='Training', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, val_hist, width, label='Validation', color='#FF9800', alpha=0.8)
    
    ax.set_xlabel('Cloud Coverage Fraction')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Cloud Coverage Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(b*100)}%' for b in bins[:-1]])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def export_landcover_distribution(stats, output_path):
    """Create landcover class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts = stats['landcover_counts']
    # Normalize to percentages
    percentages = counts / counts.sum() * 100
    
    # Filter classes with minimal presence (<0.5%)
    valid_indices = []
    valid_names = []
    valid_percentages = []
    valid_colors = []
    
    for i in range(len(LANDCOVER_CLASSES)):
        if percentages[i] >= 0.5:
            valid_indices.append(i)
            valid_names.append(LANDCOVER_CLASSES[i])
            valid_percentages.append(percentages[i])
            valid_colors.append(LANDCOVER_COLORS[i])
    
    # Sort by percentage descending
    sorted_indices = np.argsort(valid_percentages)[::-1]
    valid_names = [valid_names[i] for i in sorted_indices]
    valid_percentages = [valid_percentages[i] for i in sorted_indices]
    valid_colors = [valid_colors[i] for i in sorted_indices]
    
    x = np.arange(len(valid_names))
    bars = ax.bar(x, valid_percentages, color=valid_colors, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, valid_percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Land Cover Class')
    ax.set_ylabel('Percentage of Total Area (%)')
    ax.set_title('Land Cover Class Distribution (ESA WorldCover)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=45, ha='right')
    ax.set_ylim(0, max(valid_percentages) * 1.15)  # Add space for labels
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def export_weather_features(stats, output_path):
    """Create weather features box plot."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    # Weather array shape: (N, 21) = 7 vars × 3 aggregations (min, max, mean)
    weather = stats['weather_values']
    
    for i, (ax, var_name) in enumerate(zip(axes[:7], WEATHER_VARS)):
        # Extract min, max, mean for this variable
        base_idx = i * 3
        min_vals = weather[:, base_idx]
        max_vals = weather[:, base_idx + 1]
        mean_vals = weather[:, base_idx + 2]
        
        data = [min_vals[::100], max_vals[::100], mean_vals[::100]]  # Subsample
        bp = ax.boxplot(data, tick_labels=['Min Anom', 'Max Anom', 'Mean Clima'], patch_artist=True)
        
        colors = ['#E3F2FD', '#FFECB3', '#E8F5E9']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(var_name)
        ax.set_ylabel('Normalized Value')
    
    # Hide unused subplot
    axes[7].axis('off')
    
    plt.suptitle('E-OBS Weather Features Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def export_sample_input(generator, output_path):
    """Export example input RGB composite."""
    dataset = generator.get_dataset()
    
    for x, y in dataset.take(1):
        sentinel2 = x['sentinel2'].numpy()  # (T, H, W, 4)
        cloudmask = x['cloudmask'].numpy()
        
        # Select 5 timesteps
        n_cols = 5
        timesteps = np.linspace(0, sentinel2.shape[0] - 1, n_cols, dtype=int)
        
        fig, axes = plt.subplots(2, n_cols, figsize=(12, 5))
        
        for col, t in enumerate(timesteps):
            # RGB image
            rgb = make_rgb(sentinel2[t])
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f't={t}')
            axes[0, col].axis('off')
            
            # Cloud mask
            axes[1, col].imshow(cloudmask[t, :, :, 0], cmap='gray_r', vmin=0, vmax=1)
            axes[1, col].set_title(f'Cloud: {cloudmask[t].mean()*100:.0f}%')
            axes[1, col].axis('off')
        
        axes[0, 0].set_ylabel('RGB Composite', fontsize=12)
        axes[1, 0].set_ylabel('Cloud Mask', fontsize=12)
        
        plt.suptitle('Sample Input Sequence (Sentinel-2)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
        break


def export_temporal_sampling_diagram(output_path):
    """Create diagram showing temporal sampling strategy."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Timeline
    days = np.arange(0, 151, 5)
    
    # Input period (days 4-49, every 5 days)
    input_days = np.arange(4, 50, 5)
    # Target period (days 54-149, every 5 days)
    target_days = np.arange(54, 150, 5)
    
    # Draw timeline
    ax.axhline(y=0.5, color='gray', linewidth=2)
    
    # Input markers
    for d in input_days:
        ax.plot(d, 0.5, 'o', color='#2196F3', markersize=10)
    ax.fill_between([4, 49], 0.3, 0.7, alpha=0.2, color='#2196F3', label='Input (10 frames)')
    
    # Target markers
    for d in target_days:
        ax.plot(d, 0.5, 's', color='#4CAF50', markersize=8)
    ax.fill_between([54, 149], 0.3, 0.7, alpha=0.2, color='#4CAF50', label='Target (20 frames)')
    
    # Labels
    ax.annotate('Input\n(50 days)', xy=(27, 0.8), ha='center', fontsize=11, fontweight='bold')
    ax.annotate('Target\n(100 days)', xy=(100, 0.8), ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-5, 155)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Temporal Sampling Strategy (5-day intervals)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add tick marks for key days
    ax.set_xticks([0, 4, 49, 54, 149])
    ax.set_xticklabels(['0', '4', '49', '54', '149'])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Results and Discussion Exports
# =============================================================================

def collect_prediction_metrics(model, generator, max_samples=50):
    """Collect prediction metrics from model evaluation."""
    dataset = generator.get_dataset().batch(1).map(adapt_inputs)
    
    n_landcover_classes = len(LANDCOVER_CLASSES)
    n_timesteps = 20  # Expected number of forecast timesteps
    n_bands = 4  # Number of spectral bands
    
    metrics = {
        'mae_per_band': [],
        'rmse_per_band': [],
        'mae_per_timestep': [],
        'predictions': [],
        'ground_truth': [],
        'cloudmasks': [],
        'bap_composites': [],
        'landcovers': [],
        'sample_errors': [],
        # Landcover-specific: accumulate errors and counts per class
        'lc_error_sum': np.zeros(n_landcover_classes),
        'lc_pixel_count': np.zeros(n_landcover_classes),
        # Per-timestep, per-band error tracking for stacked chart
        'band_timestep_error_sum': np.zeros((n_timesteps, n_bands)),
        'band_timestep_pixel_count': np.zeros((n_timesteps, n_bands)),
    }
    
    for i, (x, y, landcover) in enumerate(dataset):
        if i >= max_samples:
            break
        
        # Predict
        pred = model.predict(x, verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        
        pred = pred[0]  # Remove batch dim
        y_data = y[0].numpy()
        landcover_map = landcover[0].numpy()  # (H, W, 10) one-hot
        
        cloudmask = y_data[:, :, :, 0]
        true_deltas = y_data[:, :, :, 1:5]
        bap = y_data[:, :, :, 5:9]
        
        # Compute errors (masked)
        valid_mask = (1 - cloudmask)[:, :, :, np.newaxis]
        diff = (pred - true_deltas) * valid_mask
        
        # Per-band MAE
        mae_band = np.abs(diff).sum(axis=(0, 1, 2)) / (valid_mask.sum() + 1e-8)
        metrics['mae_per_band'].append(mae_band)
        
        # Per-band RMSE
        rmse_band = np.sqrt((diff ** 2).sum(axis=(0, 1, 2)) / (valid_mask.sum() + 1e-8))
        metrics['rmse_per_band'].append(rmse_band)
        
        # Per-timestep MAE
        mae_timestep = np.abs(diff).mean(axis=(1, 2, 3))
        metrics['mae_per_timestep'].append(mae_timestep)
        
        # Store sample error
        sample_error = np.abs(diff).mean()
        metrics['sample_errors'].append(sample_error)
        
        # Per-landcover class MAE
        # Average error across time and bands: (H, W)
        pixel_error = np.abs(diff).mean(axis=(0, 3))  # Average over time and bands
        valid_pixel_mask = (1 - cloudmask).mean(axis=0)  # (H, W) - fraction of valid timesteps
        
        for lc_idx in range(n_landcover_classes):
            # Get pixels belonging to this landcover class
            lc_mask = landcover_map[:, :, lc_idx]  # (H, W)
            # Weight by both landcover membership and cloud validity
            weight = lc_mask * valid_pixel_mask
            metrics['lc_error_sum'][lc_idx] += (pixel_error * weight).sum()
            metrics['lc_pixel_count'][lc_idx] += weight.sum()
        
        # Per-timestep, per-band error for stacked chart
        actual_timesteps = min(pred.shape[0], n_timesteps)
        for t in range(actual_timesteps):
            timestep_valid = 1 - cloudmask[t]  # (H, W)
            valid_pixels = timestep_valid.sum()
            
            for band_idx in range(4):
                # Error at this timestep for this band: (H, W)
                band_error = np.abs(diff[t, :, :, band_idx])
                metrics['band_timestep_error_sum'][t, band_idx] += (band_error * timestep_valid).sum()
                metrics['band_timestep_pixel_count'][t, band_idx] += valid_pixels
        
        # Store a few samples for visualization
        if len(metrics['predictions']) < 5:
            metrics['predictions'].append(pred)
            metrics['ground_truth'].append(true_deltas)
            metrics['cloudmasks'].append(cloudmask)
            metrics['bap_composites'].append(bap)
            metrics['landcovers'].append(landcover_map)
        
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{max_samples} samples...")
    
    # Aggregate metrics
    metrics['mae_per_band'] = np.mean(metrics['mae_per_band'], axis=0)
    metrics['rmse_per_band'] = np.mean(metrics['rmse_per_band'], axis=0)
    metrics['mae_per_timestep'] = np.mean(metrics['mae_per_timestep'], axis=0)
    
    # Compute MAE per landcover class
    metrics['mae_per_landcover'] = np.divide(
        metrics['lc_error_sum'], 
        metrics['lc_pixel_count'],
        out=np.zeros_like(metrics['lc_error_sum']),
        where=metrics['lc_pixel_count'] > 0
    )
    
    # Compute MAE per timestep per band
    metrics['mae_per_timestep_band'] = np.divide(
        metrics['band_timestep_error_sum'],
        metrics['band_timestep_pixel_count'],
        out=np.zeros_like(metrics['band_timestep_error_sum']),
        where=metrics['band_timestep_pixel_count'] > 0
    )
    
    return metrics


def export_error_metrics_table(metrics, output_path):
    """Export error metrics to CSV."""
    data = {
        'Band': BAND_NAMES,
        'MAE': [f'{v:.5f}' for v in metrics['mae_per_band']],
        'RMSE': [f'{v:.5f}' for v in metrics['rmse_per_band']],
    }
    
    df = pd.DataFrame(data)
    
    # Add overall row
    overall = pd.DataFrame({
        'Band': ['Overall'],
        'MAE': [f"{np.mean(metrics['mae_per_band']):.5f}"],
        'RMSE': [f"{np.mean(metrics['rmse_per_band']):.5f}"],
    })
    df = pd.concat([df, overall], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return df


def export_prediction_comparison(metrics, output_path):
    """Create prediction vs ground truth comparison grid (RGB composites).
    
    Generates separate images for each sample: output_path_1.png, output_path_2.png, etc.
    Randomly selects samples from available predictions.
    """
    n_available = len(metrics['predictions'])
    n_samples = min(2, n_available)
    n_timesteps = 4
    
    # Randomly select sample indices
    sample_indices = np.random.choice(n_available, size=n_samples, replace=False)
    print(f"  Selected random samples: {sample_indices}")
    
    # Extract base path without extension
    base_path = output_path.rsplit('.', 1)[0]
    ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
    
    for i, sample_idx in enumerate(sample_indices):
        pred = metrics['predictions'][sample_idx]
        true = metrics['ground_truth'][sample_idx]
        
        # Create figure for this sample (2 rows: true and predicted)
        fig, axes = plt.subplots(
            2, n_timesteps, 
            figsize=(10, 3.5),
            gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
        )
        
        # Select timesteps to show
        timesteps = np.linspace(0, pred.shape[0] - 1, n_timesteps, dtype=int)
        
        for col, t in enumerate(timesteps):
            # Ground truth
            true_rgb = make_delta_rgb(true[t])
            axes[0, col].imshow(true_rgb)
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            axes[0, col].set_title(f't = {t}', fontsize=11, fontweight='bold', pad=8)
            
            # Row label (first column only)
            if col == 0:
                axes[0, col].set_ylabel('True', fontsize=10)
            
            # Prediction
            pred_rgb = make_delta_rgb(pred[t])
            axes[1, col].imshow(pred_rgb)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])
            
            # Row labels for prediction rows
            if col == 0:
                axes[1, col].set_ylabel('Predicted', fontsize=10)
        
        fig.suptitle(f'Sample {i + 1}: Prediction vs Ground Truth (Reflectance Deltas)', 
                     fontsize=14, fontweight='bold')
        # Use tight_layout with rect to leave space for suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        
        sample_path = f"{base_path}_{i + 1}.{ext}"
        plt.savefig(sample_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {sample_path}")


def export_temporal_error(metrics, output_path):
    """Create stacked bar chart of temporal error by spectral band."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get per-timestep, per-band MAE
    mae_ts_band = metrics['mae_per_timestep_band']  # (n_timesteps, n_bands)
    
    n_timesteps = mae_ts_band.shape[0]
    timesteps = np.arange(1, n_timesteps + 1)
    
    # Create stacked bar chart
    bottom = np.zeros(n_timesteps)
    for i, (band_name, band_color) in enumerate(zip(BAND_NAMES, BAND_COLORS)):
        ax.bar(timesteps, mae_ts_band[:, i], bottom=bottom, label=band_name, 
               color=band_color, edgecolor='white', linewidth=0.3)
        bottom += mae_ts_band[:, i]
    
    ax.set_xlabel('Forecast Timestep (5-day intervals)')
    ax.set_ylabel('Mean Absolute Error (Stacked)')
    ax.set_title('Prediction Error Over Forecast Horizon by Band', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, n_timesteps + 0.5)
    ax.set_ylim(0, None)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title='Spectral Band')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def export_spatial_error_heatmap(metrics, output_path):
    """Create spatial error heatmap."""
    # Average absolute error across samples and timesteps
    all_errors = []
    for i in range(len(metrics['predictions'])):
        pred = metrics['predictions'][i]
        true = metrics['ground_truth'][i]
        mask = metrics['cloudmasks'][i]
        
        error = np.abs(pred - true).mean(axis=-1)  # Average across bands
        error = error * (1 - mask)  # Apply cloud mask
        all_errors.append(error.mean(axis=0))  # Average across time
    
    avg_error = np.mean(all_errors, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(avg_error, cmap='YlOrRd', vmin=0, vmax=np.percentile(avg_error, 95))
    ax.set_title('Spatial Distribution of Prediction Error', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Absolute Error')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def compute_ndvi(nir, red):
    """Compute simple NDVI.
    
    NDVI = (NIR - Red) / (NIR + Red)
    """
    return (nir - red) / (nir + red + 1e-8)


def export_ndvi_comparison(metrics, output_path):
    """Create NDVI comparison visualization with proper grid layout."""
    n_samples = min(2, len(metrics['predictions']))
    n_timesteps = 4
    
    # Create figure with constrained layout for tight spacing
    fig, axes = plt.subplots(
        n_samples * 2, n_timesteps, 
        figsize=(10, 3.5 * n_samples),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
    )
    
    im = None  # Store for colorbar
    
    for sample_idx in range(n_samples):
        pred = metrics['predictions'][sample_idx]
        true = metrics['ground_truth'][sample_idx]
        bap = metrics['bap_composites'][sample_idx]
        
        # Reconstruct full reflectance
        pred_refl = bap + pred
        true_refl = bap + true
        
        # Compute NDVI (NIR = band 3, Red = band 2)
        pred_ndvi = compute_ndvi(pred_refl[:, :, :, 3], pred_refl[:, :, :, 2])
        true_ndvi = compute_ndvi(true_refl[:, :, :, 3], true_refl[:, :, :, 2])
        
        timesteps = np.linspace(0, pred.shape[0] - 1, n_timesteps, dtype=int)
        
        for col, t in enumerate(timesteps):
            row_true = sample_idx * 2
            row_pred = sample_idx * 2 + 1
            
            # Ground truth NDVI
            im = axes[row_true, col].imshow(true_ndvi[t], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            axes[row_true, col].set_xticks([])
            axes[row_true, col].set_yticks([])
            
            # Column labels (top row only)
            if sample_idx == 0:
                axes[row_true, col].set_title(f't = {t}', fontsize=11, fontweight='bold')
            
            # Row labels (first column only)
            if col == 0:
                axes[row_true, col].set_ylabel(f'Sample {sample_idx+1}\nTrue', fontsize=10)
            
            # Predicted NDVI
            axes[row_pred, col].imshow(pred_ndvi[t], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            axes[row_pred, col].set_xticks([])
            axes[row_pred, col].set_yticks([])
            
            # Row labels for prediction rows
            if col == 0:
                axes[row_pred, col].set_ylabel('Predicted', fontsize=10)
    
    # Add single colorbar on the right side
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.8, pad=0.02, aspect=30)
    cbar.set_label('NDVI', fontsize=11)
    
    plt.suptitle('NDVI Comparison (True vs Predicted)', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def export_mae_by_landcover(metrics, output_path):
    """Create bar chart of MAE by landcover class."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mae_values = metrics['mae_per_landcover']
    pixel_counts = metrics['lc_pixel_count']
    
    # Filter classes with enough pixels (at least 1% of total)
    total_pixels = pixel_counts.sum()
    min_pixels = total_pixels * 0.001  # 0.1% threshold
    
    # Prepare data
    valid_indices = []
    valid_names = []
    valid_mae = []
    valid_colors = []
    valid_percentages = []
    
    for i in range(len(LANDCOVER_CLASSES)):
        if pixel_counts[i] >= min_pixels and mae_values[i] > 0:
            valid_indices.append(i)
            valid_names.append(LANDCOVER_CLASSES[i])
            valid_mae.append(mae_values[i])
            valid_colors.append(LANDCOVER_COLORS[i])
            valid_percentages.append(pixel_counts[i] / total_pixels * 100)
    
    if not valid_names:
        print(f"Warning: No landcover classes with sufficient data for {output_path}")
        plt.close()
        return
    
    # Sort by MAE descending
    sorted_indices = np.argsort(valid_mae)[::-1]
    valid_names = [valid_names[i] for i in sorted_indices]
    valid_mae = [valid_mae[i] for i in sorted_indices]
    valid_colors = [valid_colors[i] for i in sorted_indices]
    valid_percentages = [valid_percentages[i] for i in sorted_indices]
    
    x = np.arange(len(valid_names))
    bars = ax.bar(x, valid_mae, color=valid_colors, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, valid_percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Land Cover Class')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error by Land Cover Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=45, ha='right')
    ax.set_ylim(0, max(valid_mae) * 1.15)  # Add space for labels
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add legend for bar meaning
    ax.annotate('(% of pixels shown above bars)', 
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main Export Function
# =============================================================================

def main(max_samples=200):
    """Run all exports."""
    
    # Create output directories
    os.makedirs(MATERIAL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PAPER ASSET EXPORT")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Materials and Data Section
    # -------------------------------------------------------------------------
    print("\n[1/2] MATERIALS AND DATA")
    print("-" * 40)
    
    print("\nCollecting training set statistics...")
    train_generator = DatasetGenerator(DATASET_PATH)
    train_stats = collect_dataset_statistics(train_generator, max_samples)
    
    print("\nCollecting validation set statistics...")
    val_generator = DatasetGenerator(VALIDATION_PATH)
    val_stats = collect_dataset_statistics(val_generator, max_samples)
    
    print("\nExporting materials figures...")
    
    # Dataset statistics table
    export_dataset_statistics_table(
        train_stats, val_stats,
        os.path.join(MATERIAL_DIR, "dataset_statistics.csv")
    )
    
    # Spectral distribution
    export_spectral_distribution(
        train_stats,
        os.path.join(MATERIAL_DIR, "spectral_distribution.png")
    )
    
    # Cloud coverage
    export_cloud_coverage(
        train_stats, val_stats,
        os.path.join(MATERIAL_DIR, "cloud_coverage.png")
    )
    
    # Landcover distribution
    export_landcover_distribution(
        train_stats,
        os.path.join(MATERIAL_DIR, "landcover_distribution.png")
    )
    
    # Weather features
    export_weather_features(
        train_stats,
        os.path.join(MATERIAL_DIR, "weather_features.png")
    )
    
    # Sample input
    export_sample_input(
        train_generator,
        os.path.join(MATERIAL_DIR, "sample_input.png")
    )
    
    # Temporal sampling diagram
    export_temporal_sampling_diagram(
        os.path.join(MATERIAL_DIR, "temporal_sampling.png")
    )
    
    # -------------------------------------------------------------------------
    # Results and Discussion Section
    # -------------------------------------------------------------------------
    print("\n[2/2] RESULTS AND DISCUSSION")
    print("-" * 40)
    
    # Load model
    model_path = os.path.join(CHECKPOINTS_DIR, "final_model.keras")
    print(f"\nLoading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Skipping results exports.")
        return
    
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Skipping results exports.")
        return
    
    print("\nCollecting prediction metrics on validation set...")
    val_metrics = collect_prediction_metrics(model, val_generator, max_samples)
    
    print("\nExporting results figures...")
    
    # Error metrics table
    export_error_metrics_table(
        val_metrics,
        os.path.join(RESULTS_DIR, "error_metrics.csv")
    )
    
    # Prediction comparison
    export_prediction_comparison(
        val_metrics,
        os.path.join(RESULTS_DIR, "prediction_comparison.png")
    )
    
    # Temporal error
    export_temporal_error(
        val_metrics,
        os.path.join(RESULTS_DIR, "temporal_error.png")
    )
    
    # Spatial error heatmap
    export_spatial_error_heatmap(
        val_metrics,
        os.path.join(RESULTS_DIR, "spatial_error_heatmap.png")
    )
    
    # NDVI comparison
    export_ndvi_comparison(
        val_metrics,
        os.path.join(RESULTS_DIR, "ndvi_comparison.png")
    )
    
    # MAE by landcover class
    export_mae_by_landcover(
        val_metrics,
        os.path.join(RESULTS_DIR, "mae_by_landcover.png")
    )
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nMaterials saved to: {MATERIAL_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export paper figures and tables.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum samples to process per dataset (default: 50)"
    )
    args = parser.parse_args()
    
    main(max_samples=args.max_samples)
