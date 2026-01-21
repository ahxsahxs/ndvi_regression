#!/usr/bin/env python3
"""
Error Analysis Script for GreenEarthNet Model (Updated).

Assess model quality using standard metrics and the specific GreenEarthNet
'Vegetation Score'. Compares performance against a Persistence Baseline.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dataset import DatasetGenerator
from build_model import load_model, build_eo_convlstm_model
from config import VALIDATION_PATH

# =============================================================================
# Constants
# =============================================================================

LANDCOVER_CLASSES = [
    'Tree cover', 'Shrubland', 'Grassland', 'Cropland', 
    'Built-up', 'Bare/sparse', 'Snow/ice', 'Water',
    'Wetland', 'Mangroves', 'Moss/lichen'
]
VEGETATION_CLASSES = ['Tree cover', 'Shrubland', 'Grassland']

OUTPUT_DIR = "images/results/error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# =============================================================================
# Helper Functions
# =============================================================================

def compute_kndvi(nir, red, sigma=0.5):
    """Compute kernel NDVI using RBF kernel."""
    diff_sq = (nir - red) ** 2
    k = np.exp(-diff_sq / (2 * sigma ** 2))
    kndvi = (1 - k) / (1 + k + 1e-8)
    return kndvi

def adapt_inputs(x, y):
    """Transform dataset inputs."""
    t_meta = x['time'][:, -1, :] 
    input_dict = {
        'sentinel2_sequence': x['sentinel2'],
        'cloudmask_sequence': x['cloudmask'],
        'landcover_map': x['landcover'],
        'weather_sequence': x['weather'],
        'temporal_metadata': t_meta
    }
    return input_dict, y

class PersistenceModel:
    """Baseline that predicts no change (Deltas = 0)."""
    def predict(self, x, verbose=0):
        # x is dict. We need output shape (Batch, T_out, H, W, 4)
        # We can look at sentinel2_sequence to get shape
        s2 = x['sentinel2_sequence']
        batch_size = tf.shape(s2)[0]
        
        # Assumption: 20 output steps, 128x128, 4 bands
        # We return ZEROS for delta
        return np.zeros((batch_size, 20, 128, 128, 4), dtype=np.float32)

# =============================================================================
# Analysis Logic
# =============================================================================

def calculate_vegetation_score_stats(true_kndvi, pred_kndvi, mask, landcover_flat):
    """
    Compute pixel-wise NSE and aggregate for Vegetation Score.
    
    Formula:
    1. nse = NSE(true, pred) at cloud-free pixels
    2. nnse = 1 / (2 - nse) -> Range [0, 1]
    3. veg_score = 2 - 1/mean(nnse) -> Range [-Inf, 1]
       Computed over Vegetation Pixels (Trees, Shrubland, Grassland).
    
    Returns:
        dict of stats per landcover
    """
    T, N = true_kndvi.shape
    
    # Store NNSE values per landcover to limit memory usage
    lc_nnse = {lc: [] for lc in LANDCOVER_CLASSES}
    
    # Vectorized NSE per pixel
    # Numerator: sum((obs - pred)^2)
    diff_sq = (true_kndvi - pred_kndvi) ** 2
    # Mask invalid (cloudy)
    valid = (mask == 0)
    
    # If a pixel is never valid, we skip it
    valid_counts = np.sum(valid, axis=0) # (N,)
    valid_pixels_mask = valid_counts > 2 # Need variance
    
    # Denominator: sum((obs - mean_obs)^2)
    # Mean of obs per pixel
    # Handle masked mean
    obs_sum = np.sum(true_kndvi * valid, axis=0)
    obs_mean = np.divide(obs_sum, valid_counts, out=np.zeros_like(obs_sum), where=valid_counts>0)
    
    term2 = (true_kndvi - obs_mean[np.newaxis, :]) ** 2
    
    numerator = np.sum(diff_sq * valid, axis=0)
    denominator = np.sum(term2 * valid, axis=0)
    
    # NSE = 1 - num/den
    # Avoid division by zero
    nse = 1 - np.divide(numerator, denominator + 1e-8, out=np.ones_like(numerator)*(-9999), where=denominator>1e-6)
    
    # NNSE = 1 / (2 - NSE)
    # Clip NSE to reasonable range [-10, 1] to avoid exploding NNSE for very bad predictions
    nse_clipped = np.clip(nse, -10, 1)
    nnse = 1 / (2 - nse_clipped)
    
    # Aggregate by Landcover
    for i, lc_name in enumerate(LANDCOVER_CLASSES):
        # Pixels belonging to this class AND having valid NSE
        indices = np.where((landcover_flat == i) & valid_pixels_mask)[0]
        if len(indices) > 0:
            lc_nnse[lc_name].extend(nnse[indices])
            
    return lc_nnse

def run_analysis_refined(model_path, max_samples=None):
    print(f"Loading Main Model from {model_path}...")
    model_main = load_model(model_path, compile=False)
    model_baseline = PersistenceModel()
    
    print(f"Loading validation dataset...")
    generator = DatasetGenerator(VALIDATION_PATH)
    # Batch(1) is critical for adapt_inputs
    dataset = generator.get_dataset().batch(1).map(adapt_inputs)
    
    if max_samples:
        dataset = dataset.take(max_samples)

    # Accumulators for NNSE values [Main, Baseline]
    # We store list of NNSEs for each landcover to compute mean later
    results_nnse = {
        'Main': {lc: [] for lc in LANDCOVER_CLASSES},
        'Baseline': {lc: [] for lc in LANDCOVER_CLASSES}
    }
    
    # Temporal Accumulators (T=20)
    # We need fixed size for this. Assuming T=20
    # Will discover T dynamically
    temporal_stats = {
        'sse': None,
        'sst': None,
        'count': None,
        'sum_obs': None
    }
    
    n_processed = 0
    print("Starting evaluation loop...")
    
    for i, (x, y_true_tensor) in enumerate(dataset):
        n_processed += 1
        
        # Ground Truth
        y_true_np = y_true_tensor.numpy()
        mask = y_true_np[0, :, :, :, 0] # (T, H, W)
        true_deltas = y_true_np[0, :, :, :, 1:5]
        bap = y_true_np[0, :, :, :, 5:9]
        
        true_abs = bap + true_deltas
        true_kndvi = compute_kndvi(true_abs[..., 3], true_abs[..., 2]) # (T, H, W)
        
        # Landcover
        lc_map = x['landcover_map'].numpy()[0]
        lc_indices = np.argmax(lc_map, axis=-1)
        lc_flat = lc_indices.reshape(-1) # (N,)
        
        T, H, W = true_kndvi.shape
        true_kndvi_flat = true_kndvi.reshape(T, -1)
        mask_flat = mask.reshape(T, -1)
        
        # --- Evaluate Main Model ---
        pred_main_deltas = model_main.predict(x, verbose=0)
        if isinstance(pred_main_deltas, list): pred_main_deltas = pred_main_deltas[0]
        
        pred_main_abs = bap + pred_main_deltas[0]
        pred_main_kndvi = compute_kndvi(pred_main_abs[..., 3], pred_main_abs[..., 2])
        pred_main_flat = pred_main_kndvi.reshape(T, -1)
        
        stats_main = calculate_vegetation_score_stats(true_kndvi_flat, pred_main_flat, mask_flat, lc_flat)
        for lc, vals in stats_main.items():
            results_nnse['Main'][lc].extend(vals)
            
        # --- Temporal Error Accumulation ---
        # T, H, W = true_kndvi.shape
        # We want to aggregate over all valid pixels for each timestep t in 0..T-1
        
        if temporal_stats['sse'] is None:
            temporal_stats['sse'] = np.zeros(T)
            temporal_stats['sst'] = np.zeros(T)
            temporal_stats['count'] = np.zeros(T)
            temporal_stats['sum_obs'] = np.zeros(T)
            
        # Mask is (T, H, W). 0 is valid.
        valid_mask = (mask == 0) # (T, H, W)
        
        # We need "Global Mean Obs per Timestep" for NSE, but strict NSE uses
        # sum((obs - pred)^2) / sum((obs - mean_obs_global)^2).
        # We usually define mean_obs_global as mean over ALL samples at step t.
        # But here we process online. 
        # So we can accumulate Sum(Obs), Sum(Obs^2) to compute variance later?
        # SST = Sum((Obs - Mean)^2) = Sum(Obs^2) - N*Mean^2
        # Let's accumulate Sum(Obs) and Sum(Obs^2) separately?
        # Actually, let's just accumulate Sum((Obs - Pred)^2) and Sum(Obs) and Sum(Obs^2).
        
        # Valid counts per step
        step_counts = np.sum(valid_mask, axis=(1, 2)) # (T,)
        
        if np.any(step_counts > 0):
            # SSE = Sum((True - Pred)^2) masked
            sq_diff = (true_kndvi - pred_main_kndvi) ** 2
            step_sse = np.sum(sq_diff * valid_mask, axis=(1, 2)) # (T,)
            
            # Sum Obs and Sum Obs^2 for Variance
            step_sum_obs = np.sum(true_kndvi * valid_mask, axis=(1, 2))
            step_sum_obs_sq = np.sum((true_kndvi ** 2) * valid_mask, axis=(1, 2))
            
            temporal_stats['sse'] += step_sse
            temporal_stats['count'] += step_counts
            temporal_stats['sum_obs'] += step_sum_obs
            temporal_stats['sst'] += step_sum_obs_sq # Storing Sum(Obs^2) temporarily in sst
            
        # --- Evaluate Baseline Model (Persistence) ---
        # Persistence means pred_deltas = 0, so pred_abs = BAP
        pred_base_abs = bap # + 0
        pred_base_kndvi = compute_kndvi(pred_base_abs[..., 3], pred_base_abs[..., 2])
        pred_base_flat = pred_base_kndvi.reshape(T, -1)
        
        stats_base = calculate_vegetation_score_stats(true_kndvi_flat, pred_base_flat, mask_flat, lc_flat)
        for lc, vals in stats_base.items():
            results_nnse['Baseline'][lc].extend(vals)
            
        print(f"Processed sample {n_processed}...", end='\r')
        
    print("\nAnalysis complete. generating reports...")
    
    # --- Generate Report ---
    report_data = []
    
    for lc in LANDCOVER_CLASSES:
        # Main
        nnse_main = np.array(results_nnse['Main'][lc])
        if len(nnse_main) > 0:
            mean_nnse_main = np.mean(nnse_main)
            veg_score_main = 2 - (1 / mean_nnse_main)
        else:
            veg_score_main = np.nan
            
        # Baseline
        nnse_base = np.array(results_nnse['Baseline'][lc])
        if len(nnse_base) > 0:
            mean_nnse_base = np.mean(nnse_base)
            veg_score_base = 2 - (1 / mean_nnse_base)
        else:
            veg_score_base = np.nan
            
        report_data.append({
            'Landcover': lc,
            'VegetationScore_Main': veg_score_main,
            'VegetationScore_Baseline': veg_score_base,
            'PixelCount': len(nnse_main),
            'IsVegetation': lc in VEGETATION_CLASSES
        })
        
    df = pd.DataFrame(report_data)
    csv_path = os.path.join(OUTPUT_DIR, "refined_error_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved report to {csv_path}")
    
    # Compare Global Vegetation Score (Aggregated over veg classes)
    veg_df = df[df['IsVegetation']]
    # We need weighted average of NNSE, but here we just averaged locally.
    # Ideally should aggregate ALL vegetation pixels then compute mean.
    # But approximate average of scores is OK for viz.
    
    print("\n--- Summary ---")
    print(df[['Landcover', 'VegetationScore_Main', 'VegetationScore_Baseline']].round(4))
    
    plot_comparison(df)

    # --- Temporal Error Analysis Output ---
    if temporal_stats['count'] is not None:
        T_steps = len(temporal_stats['sse'])
        temp_data = []
        
        for t in range(T_steps):
            N = temporal_stats['count'][t]
            if N > 0:
                rmse = np.sqrt(temporal_stats['sse'][t] / N)
                sum_sq = temporal_stats['sst'][t]
                sum_obs = temporal_stats['sum_obs'][t]
                sst_real = sum_sq - (sum_obs ** 2 / N)
                
                if sst_real > 1e-6:
                    nse = 1 - (temporal_stats['sse'][t] / sst_real)
                else:
                    nse = np.nan
            else:
                rmse = np.nan
                nse = np.nan
                
            temp_data.append({
                'Step': t + 1,
                'RMSE': rmse,
                'NSE': nse,
                'Count': N
            })
            
        df_temp = pd.DataFrame(temp_data)
        temp_csv_path = os.path.join(OUTPUT_DIR, "temporal_error_report.csv")
        df_temp.to_csv(temp_csv_path, index=False)
        print(f"Saved temporal report to {temp_csv_path}")
        
        plot_temporal_error(df_temp)

def plot_temporal_error(df_temporal):
    """Plot Error Evolution over Forecast Horizon."""
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.gca()
    sns.lineplot(data=df_temporal, x='Step', y='RMSE', marker='o', ax=ax1, color='blue', label='Root Mean Square Error (RMSE)')
    ax1.set_ylabel('RMSE (kNDVI)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=df_temporal, x='Step', y='NSE', marker='s', ax=ax2, color='orange', label='Nash-Sutcliffe Efficiency (NSE)')
    ax2.set_ylabel('NSE', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(-0.2, 1.0)
    
    ax1.grid(True, alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    out_path = os.path.join(OUTPUT_DIR, "temporal_error.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def plot_comparison(df):
    """Plot Comparison of Vegetation Scores."""
    # Melt for seaborn
    melted = df.melt(id_vars=['Landcover', 'IsVegetation'], 
                     value_vars=['VegetationScore_Main', 'VegetationScore_Baseline'],
                     var_name='Model', value_name='VegetationScore')
    
    plt.figure(figsize=(10, 6))
    
    # Filter for interesting classes (Vegetation + maybe Cropland)
    subset = melted[melted['IsVegetation'] | (melted['Landcover'] == 'Cropland')]
    
    sns.barplot(data=subset, x='Landcover', y='VegetationScore', hue='Model', palette='muted')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Target Mean Performance')
    plt.axhline(1, color='green', linewidth=0.8, linestyle=':', label='Perfect Prediction')
    
    plt.title('Vegetation Score: Model vs Persistence Baseline')
    plt.ylabel('Vegetation Score (1 is perfect)')
    plt.ylim(-0.5, 1.1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "vegetation_score_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/final_model.keras")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: {args.model} not found")
        exit(1)
        
    run_analysis_refined(args.model, args.max_samples)
