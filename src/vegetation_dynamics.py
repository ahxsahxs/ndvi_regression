#!/usr/bin/env python3
"""
Vegetation Dynamics Analysis Script (Refined).

Analyzes the relationship between predictors (Weather, Time) and target variables
(kNDVI dynamics) using Explainable AI (XAI) techniques:
- Permutation Feature Importance
- Surrogate Decision Trees (Rule Extraction)
- Extreme Event Attribution (Drought/Heatwave)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# E-OBS variables
WEATHER_VARS = ['Temp Mean', 'Humidity', 'Pressure', 'Radiation', 
                'Precipitation', 'Temp Min', 'Temp Max']

OUTPUT_DIR = "images/results/vegetation_dynamics"
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

def get_all_feature_names():
    """Generate list of all feature names including Weather, Landcover, and Time."""
    names = []
    # Weather
    aggs = ['MinAnom', 'MaxAnom', 'MeanClim']
    for var in WEATHER_VARS:
        for agg in aggs:
            names.append(f"{var}_{agg}")
    
    # Static / Semi-static
    names.extend(['Landcover_ID', 'Year_Norm', 'Sin_DOY', 'Cos_DOY', 'Step_Index'])
    return names

# =============================================================================
# Analysis Logic
# =============================================================================

def run_dynamics_analysis_refined(model_path, max_samples=None):
    print(f"Loading Main Model from {model_path}...")
    model = load_model(model_path, compile=False)
    
    print(f"Loading validation dataset...")
    generator = DatasetGenerator(VALIDATION_PATH)
    dataset = generator.get_dataset().batch(1).map(adapt_inputs)
    
    if max_samples:
        dataset = dataset.take(max_samples)
        
    feature_names = get_all_feature_names()
    
    # Collect data for XAI
    X_data = []
    y_data = []
    
    print("Collecting dynamics data...")
    n_processed = 0
    
    for x, y_true_tensor in dataset:
        n_processed += 1
        
        # Predict
        y_pred_deltas = model.predict(x, verbose=0)
        if isinstance(y_pred_deltas, list): y_pred_deltas = y_pred_deltas[0]
        
        # Weather Sequence (1, T, 21)
        weather_seq = x['weather_sequence'].numpy()[0]
        
        # Temporal Meta (1, 3) -> [Year, Sin, Cos] at Start of Input?
        # Dataset says t_meta = x['time'][:, -1, :] which is Last Input Step.
        t_meta = x['temporal_metadata'].numpy()[0]
        year_norm = t_meta[0]
        sin_doy_start = t_meta[1]
        cos_doy_start = t_meta[2]
        
        # Reconstruct approximate Day of Year (radians) from sin/cos
        # atan2(sin, cos) -> radians in [-pi, pi]
        doy_rad_start = np.arctan2(sin_doy_start, cos_doy_start)
        if doy_rad_start < 0: doy_rad_start += 2 * np.pi
        
        # 5 days per step. Total year ~ 365 days -> 2pi radians
        # rad_per_step = (5 / 365) * 2pi approx 0.086
        rad_per_step = (5 / 365.25) * 2 * np.pi
        
        # Landcover (Tile Mode)
        lc_map = x['landcover_map'].numpy()[0]
        # Dominant class index (0-10)
        lc_counts = np.sum(lc_map, axis=(0,1))
        lc_mode_idx = np.argmax(lc_counts)
        # We use the categorical ID (0..10) as a feature. 
        # Ideally One-Hot, but Trees handle Ordinal/Integer fine if cardinality low.
        
        # kNDVI Calculation
        y_true_np = y_true_tensor.numpy()
        mask = y_true_np[0, :, :, :, 0]
        bap = y_true_np[0, :, :, :, 5:9]
        
        # Predicted Absolute -> kNDVI
        pred_abs = bap + y_pred_deltas[0]
        pred_kndvi = compute_kndvi(pred_abs[..., 3], pred_abs[..., 2])
        
        bap_kndvi = compute_kndvi(bap[..., 3], bap[..., 2])
        
        T_steps = pred_kndvi.shape[0]
        
        # Spatial Mean of VALID pixels
        valid_mask_any = np.any(mask == 0, axis=0) # (H, W)
        
        if np.sum(valid_mask_any) == 0:
            continue
            
        # BAP kNDVI
        bap_val = np.mean(bap_kndvi[0][valid_mask_any])
        
        # Predicted kNDVI Series
        pred_series = []
        for t in range(T_steps):
            val = np.mean(pred_kndvi[t][valid_mask_any])
            pred_series.append(val)
            
        full_series = np.concatenate([[bap_val], pred_series])
        deltas = np.diff(full_series) # (T_steps,)
        
        # Accumulate Data
        for t in range(T_steps):
            # Construct feature vector
            weather_feats = weather_seq[t] # (21,)
            
            # Temporal update
            doy_rad_t = doy_rad_start + (t + 1) * rad_per_step # Shift by steps
            sin_t = np.sin(doy_rad_t)
            cos_t = np.cos(doy_rad_t)
            
            # Combined features: [Weather (21), LC (1), Year (1), Sin (1), Cos (1), Step (1)]
            feats = np.concatenate([
                weather_feats,
                [lc_mode_idx, year_norm, sin_t, cos_t, float(t)]
            ])
            
            X_data.append(feats)
            y_data.append(deltas[t])
            
        print(f"Processed sample {n_processed}...", end='\r')
        
    print("\nData collected. Running XAI analysis...")
    
    X_df = pd.DataFrame(X_data, columns=feature_names)
    y_target = np.array(y_data)
    
    # 1. Permutation Feature Importance
    # Train a lightweight surrogate model (Random Forest) to mimic the deep model's behavior
    # mapping Weather -> Delta kNDVI.
    # Note: This explains "How much does Weather feature X affect Predicted Delta".
    
    surrogate = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    surrogate.fit(X_df, y_target)
    
    # Compute Importance
    perm_importance = permutation_importance(surrogate, X_df, y_target, n_repeats=10, random_state=42)
    
    sorted_idx = perm_importance.importances_mean.argsort()
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    plt.barh(X_df.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Importance of Weather Features for Predicted kNDVI Change")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    
    # 2. Surrogate Rule Extraction (Decision Tree)
    # Train a simple Decision Tree (depth=3) for interpretability
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X_df, y_target)
    
    # Export Rules of Text
    rules = export_text(tree, feature_names=list(feature_names))
    with open(os.path.join(OUTPUT_DIR, "decision_rules.txt"), "w") as f:
        f.write("Decision Tree Rules (Weather -> Delta kNDVI):\n")
        f.write(rules)
    print("Saved decision rules.")
    
    # 3. Extreme Event Analysis
    # Define "Extreme Event" as High Temp (MeanClim > 0.8 normalized?) or Low Precip?
    # Or define based on percentiles of the collected data.
    
    # Let's find "Heatwave" conditions: High Temp Max Anom
    temp_max_col = [c for c in feature_names if 'Temp Max_MaxAnom' in c][0]
    thresh_hw = X_df[temp_max_col].quantile(0.95)
    
    heatwave_mask = X_df[temp_max_col] > thresh_hw
    hw_deltas = y_target[heatwave_mask]
    normal_deltas = y_target[~heatwave_mask]
    
    print(f"Heatwave Threshold ({temp_max_col}): {thresh_hw:.3f}")
    print(f"Avg Delta during Heatwave: {np.mean(hw_deltas):.4f}")
    print(f"Avg Delta during Normal: {np.mean(normal_deltas):.4f}")
    
    # Boxplot Comparison
    plt.figure(figsize=(6, 5))
    data_hw = pd.DataFrame({'Condition': ['Heatwave']*len(hw_deltas) + ['Normal']*len(normal_deltas), 
                            'Delta kNDVI': np.concatenate([hw_deltas, normal_deltas])})
    
    sns.boxplot(data=data_hw, x='Condition', y='Delta kNDVI', palette='Set2')
    plt.title('Vegetation Response to Extreme Heat (Top 5% Max Temp)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "extreme_event_heatwave.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/final_model.keras")
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: {args.model} not found")
        exit(1)
        
    run_dynamics_analysis_refined(args.model, args.max_samples)
