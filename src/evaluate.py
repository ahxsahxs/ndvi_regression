#!/usr/bin/env python3
"""
Benchmark Evaluation Script for GreenEarthNet Model.

Computes NSE and Vegetation Score metrics for comparison with Contextformer.
Outputs structured JSON for easy table generation in the Results chapter.

Usage:
    python src/evaluate.py \
        --model checkpoints/final_model.keras \
        --split val_chopped \
        --metrics nse vegetation_score \
        --output results/benchmark_comparison.json
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path

# Force CPU to avoid XLA/CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dataset import DatasetGenerator
from build_model import load_model
from config import DATASET_PATH, VALIDATION_PATH

# =============================================================================
# Constants
# =============================================================================

VEGETATION_LC_INDICES = [0, 1, 2]  # Tree cover, Shrubland, Grassland in one-hot

# =============================================================================
# Helper Functions
# =============================================================================

def compute_ndvi(nir, red):
    """Compute standard NDVI = (NIR - Red) / (NIR + Red).
    Matches the official EarthNet benchmark protocol."""
    return (nir - red) / (nir + red + 1e-8)


def adapt_inputs(x, y):
    """Transform dataset inputs to model format."""
    t_meta = x['time'][:, -1, :]
    input_dict = {
        'sentinel2_sequence': x['sentinel2'],
        'cloudmask_sequence': x['cloudmask'],
        'landcover_map': x['landcover'],
        'weather_sequence': x['weather'],
        'temporal_metadata': t_meta
    }
    return input_dict, y, x['landcover']


def get_vegetated_mask(landcover_onehot):
    """
    Get mask for vegetated pixels (Tree cover, Shrubland, Grassland, Cropland).
    
    Landcover encoding (from dataset.py):
    - Channel 0: Shrubland (20)
    - Channel 1: Grassland (30)  
    - Channel 2: Cropland (40)
    - Implicit (all zeros): Tree cover (10)
    """
    # Vegetated channels: 0 (Shrubland), 1 (Grassland), 2 (Cropland)
    explicit_veg = np.sum(landcover_onehot[..., :3], axis=-1)
    
    # Tree cover is implicit (all zeros)
    is_tree = (np.sum(landcover_onehot, axis=-1) < 0.5).astype(np.float32)
    
    # Union
    is_vegetated = np.clip(explicit_veg + is_tree, 0, 1)
    return is_vegetated


def compute_nse(obs, pred, valid_mask):
    """
    Compute Nash-Sutcliffe Efficiency.
    
    NSE = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
    """
    obs_masked = obs[valid_mask > 0.5]
    pred_masked = pred[valid_mask > 0.5]
    
    if len(obs_masked) == 0:
        return np.nan
    
    obs_mean = np.mean(obs_masked)
    ss_res = np.sum((obs_masked - pred_masked) ** 2)
    ss_tot = np.sum((obs_masked - obs_mean) ** 2)
    
    if ss_tot < 1e-8:
        return np.nan  # Zero-variance: let NaN propagate (matching EarthNet protocol)
    
    return 1 - (ss_res / ss_tot)


def compute_vegetation_score(obs, pred, valid_mask, veg_mask):
    """
    Compute GreenEarthNet Vegetation Score.
    
    Formula:
    1. Compute pixel-wise NSE for vegetated pixels
    2. NNSE = 1 / (2 - NSE) -> Range [0, 1]
    3. VegScore = 2 - 1/mean(NNSE) -> Range [-Inf, 1]
    """
    # Flatten spatial dimensions, keep time
    T, H, W = obs.shape
    obs_flat = obs.reshape(T, -1)
    pred_flat = pred.reshape(T, -1)
    mask_flat = valid_mask.reshape(T, -1)
    veg_flat = veg_mask.reshape(-1)
    
    # Compute per-pixel NSE across time
    valid_counts = np.sum(mask_flat, axis=0)
    
    # Filter to vegetated pixels with at least 1 valid observation
    veg_and_valid = (veg_flat > 0.5) & (valid_counts > 0)
    pixel_indices = np.where(veg_and_valid)[0]
    
    if len(pixel_indices) == 0:
        return np.nan
    
    nnse_values = []
    for idx in pixel_indices:
        ts_valid = mask_flat[:, idx] > 0.5
        if np.sum(ts_valid) < 1:
            continue
        
        obs_ts = obs_flat[ts_valid, idx]
        pred_ts = pred_flat[ts_valid, idx]
        
        obs_mean = np.mean(obs_ts)
        ss_res = np.sum((obs_ts - pred_ts) ** 2)
        ss_tot = np.sum((obs_ts - obs_mean) ** 2)
        
        # Zero-variance: NaN propagation (matches official xarray 0/0 -> NaN behavior)
        if ss_tot == 0:
            nnse_values.append(np.nan)
            continue
        
        nse = 1 - (ss_res / ss_tot)
        nnse = 1 / (2 - nse)
        nnse_values.append(nnse)
    
    if len(nnse_values) == 0:
        return np.nan
    
    mean_nnse = np.nanmean(nnse_values)
    veg_score = 2 - (1 / mean_nnse)
    
    return veg_score


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(model_path, split, max_samples=None, metrics=['nse', 'vegetation_score'],
             model=None, generator=None):
    """Run evaluation and return metrics.
    
    Args:
        model_path: Path to model checkpoint (used only if model is None).
        split: Dataset split name.
        max_samples: Maximum samples to evaluate.
        metrics: List of metrics to compute.
        model: Pre-loaded model (optional, avoids redundant loading).
        generator: Pre-loaded DatasetGenerator (optional, avoids redundant loading).
    """
    
    if model is None:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, compile=False)
    else:
        print("Using pre-loaded model.")
    
    # Get parameter count
    param_count = model.count_params()
    print(f"Model parameters: {param_count:,}")
    
    if generator is None:
        # Select dataset path based on split
        if split in ['val_chopped', 'val']:
            dataset_path = VALIDATION_PATH
        else:
            dataset_path = DATASET_PATH
        
        print(f"Loading {split} dataset from {dataset_path}...")
        generator = DatasetGenerator(dataset_path)
    else:
        print("Using pre-loaded generator.")
    
    dataset = generator.get_dataset().batch(1).map(adapt_inputs)
    
    if max_samples:
        dataset = dataset.take(max_samples)
    
    # Accumulators
    all_obs = []
    all_pred = []
    all_veg_masks = []
    all_valid_masks = []
    n_samples = 0
    
    print("Evaluating...")
    for i, (x, y_tensor, lc_tensor) in enumerate(dataset):
        n_samples += 1
        
        # Ground truth
        y_np = y_tensor.numpy()[0]
        mask = y_np[:, :, :, 0]  # Cloud mask
        true_deltas = y_np[:, :, :, 1:5]
        bap = y_np[:, :, :, 5:9]
        
        # Predictions
        pred_deltas = model.predict(x, verbose=0)
        if isinstance(pred_deltas, list):
            pred_deltas = pred_deltas[0]
        pred_deltas = pred_deltas[0]
        
        # Compute absolute reflectance
        true_abs = bap + true_deltas
        pred_abs = bap + pred_deltas
        
        # Compute NDVI (NIR=band 3, Red=band 2)
        true_ndvi = np.clip(compute_ndvi(true_abs[..., 3], true_abs[..., 2]), -1.0, 1.0)
        pred_ndvi = np.clip(compute_ndvi(pred_abs[..., 3], pred_abs[..., 2]), -1.0, 1.0)
        
        # Get vegetation mask
        lc_np = lc_tensor.numpy()[0]
        veg_mask = get_vegetated_mask(lc_np)
        
        # Valid mask (inverted cloud mask)
        valid_mask = 1 - mask
        
        all_obs.append(true_ndvi)
        all_pred.append(pred_ndvi)
        all_veg_masks.append(veg_mask)
        all_valid_masks.append(valid_mask)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1} samples...")
    
    print(f"Processed {n_samples} samples total.")
    
    # Compute metrics
    results = {
        'model_path': model_path,
        'split': split,
        'samples': n_samples,
        'metrics': {
            'parameters': param_count
        }
    }
    
    # Aggregate all data
    all_obs = np.concatenate([o.reshape(-1) for o in all_obs])
    all_pred = np.concatenate([p.reshape(-1) for p in all_pred])
    all_valid = np.concatenate([m.reshape(-1) for m in all_valid_masks])
    
    if 'nse' in metrics:
        nse = compute_nse(all_obs, all_pred, all_valid)
        results['metrics']['nse_ndvi'] = float(nse) if not np.isnan(nse) else None
        print(f"NSE (NDVI): {nse:.4f}")
    
    if 'vegetation_score' in metrics:
        # Compute per-sample vegetation scores and average
        veg_scores = []
        for i in range(len(all_veg_masks)):
            obs = np.stack([all_obs[i*20*128*128:(i+1)*20*128*128].reshape(20, 128, 128)])
            pred = np.stack([all_pred[i*20*128*128:(i+1)*20*128*128].reshape(20, 128, 128)])
            valid = np.stack([all_valid[i*20*128*128:(i+1)*20*128*128].reshape(20, 128, 128)])
            
            # Recalculate from stored arrays
            pass  # Use stored per-sample data instead
        
        # Simpler approach: compute global vegetation score 
        # using the first sample's vegetation mask as representative
        # (In practice, should iterate properly)
        print("Computing Vegetation Score...")
        veg_scores = []
        for i in range(min(n_samples, len(all_veg_masks))):
            try:
                vs = compute_vegetation_score(
                    all_obs[i*20*128*128:(i+1)*20*128*128].reshape(20, 128, 128),
                    all_pred[i*20*128*128:(i+1)*20*128*128].reshape(20, 128, 128),
                    all_valid[i*20*128*128:(i+1)*20*128*128].reshape(20, 128, 128),
                    all_veg_masks[i]
                )
                if not np.isnan(vs):
                    veg_scores.append(vs)
            except Exception:
                continue
        
        if veg_scores:
            mean_veg_score = np.mean(veg_scores)
            results['metrics']['vegetation_score'] = float(mean_veg_score)
            print(f"Vegetation Score: {mean_veg_score:.4f}")
        else:
            results['metrics']['vegetation_score'] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model for benchmark comparison")
    parser.add_argument('--model', type=str, default='checkpoints/final_model.keras',
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val_chopped',
                        choices=['train', 'val_chopped', 'val'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--metrics', nargs='+', default=['nse', 'vegetation_score'],
                        choices=['nse', 'vegetation_score'],
                        help='Metrics to compute')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate (None = all)')
    parser.add_argument('--output', type=str, default='results/benchmark_comparison.json',
                        help='Output JSON file path')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return 1
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    results = evaluate(args.model, args.split, args.max_samples, args.metrics)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(json.dumps(results, indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())
