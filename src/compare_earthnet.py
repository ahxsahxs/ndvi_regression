#!/usr/bin/env python3
"""
Script to generate predictions for EarthNet comparison and score them.
Generates predictions in the format: {pred_dir}/{region}/{cubename}.nc
Then runs EarthNet scoring.
"""

import os
import argparse
import numpy as np
import xarray as xr
import tensorflow as tf
from pathlib import Path
import shutil
import earthnet as entk
try:
    import earthnet.score_v2 as score_v2
except ImportError:
    # Fallback/find where score_v2 is if not directly exposed
    from earthnet import score_v2

# Monkeypatch normalized_NSE to fully reimplement it (fixing s2_dlmask and np.NaN)
# _original_normalized_NSE = score_v2.normalized_NSE # Not used anymore



def _patched_normalized_NSE(targ, pred, name_ndvi_pred="ndvi_pred", additional_metrics=False):
    """
    Patched version of earthnet.score_v2.normalized_NSE.
    Fixes:
    1. Handles s2_dlmask -> s2_mask renaming.
    2. Uses np.nan instead of np.NaN (numpy 2.0 compatibility).
    """
    if 's2_mask' not in targ and 's2_dlmask' in targ:
        targ = targ.rename({'s2_dlmask': 's2_mask'})

    pred_start_idx = len(targ.time.isel(time=slice(4, None, 5))) - len(pred.time)

    nir = targ.s2_B8A.isel(time=slice(4, None, 5)).isel(
        time=slice(pred_start_idx, None)
    )
    red = targ.s2_B04.isel(time=slice(4, None, 5)).isel(
        time=slice(pred_start_idx, None)
    )
    mask = targ.s2_mask.isel(time=slice(4, None, 5)).isel(
        time=slice(pred_start_idx, None)
    )

    # FIX: Use np.nan instead of np.NaN (numpy 2.0 compat)
    # Use standard NDVI to match the official benchmark protocol
    targ_ndvi = ((nir - red) / (nir + red + 1e-8)).where(mask == 0, np.nan)

    pred_ndvi = pred[name_ndvi_pred]

    nnse = 1 / (
        2
        - (
            1
            - (
                ((targ_ndvi - pred_ndvi) ** 2).sum("time")
                / ((targ_ndvi - targ_ndvi.mean("time")) ** 2).sum("time")
            )
        )
    )

    n_obs = (mask == 0).sum("time")

    if additional_metrics:
        # We don't implement additional metrics here as not requested and complex to copy-paste.
        # Fallback to original if we really had to, but it would fail on np.NaN.
        # Given usage, we raise error to be safe.
        raise NotImplementedError("Patched normalized_NSE does not support additional_metrics=True")
    else:
        df = xr.Dataset(
            {"NNSE": nnse, "landcover": targ.esawc_lc, "n_obs": n_obs}
        ).to_dataframe()

    return df.drop(columns="sentinel:product_id", errors="ignore")

score_v2.normalized_NSE = _patched_normalized_NSE
# Also patch entk.normalized_NSE if it points to the same function
if hasattr(entk, 'normalized_NSE'):
    entk.normalized_NSE = _patched_normalized_NSE

# Force CPU to avoid XLA/CUDA issues if needed, or let user decide.
# For consistency with evaluate.py, we might want to be careful, 
# but usually inference is fine on GPU. evaluate.py forced CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dataset import DatasetGenerator
from build_model import load_model
from config import DATASET_PATH, VALIDATION_PATH



def prepare_input_for_model(x_dict):
    """
    Adapts the output of DatasetGenerator.prepare_x for the model.
    Adds batch dimension and extracts temporal metadata.
    """
    # Add batch dimension to all arrays
    x_batch = {k: np.expand_dims(v, axis=0) for k, v in x_dict.items()}
    
    # Extract temporal metadata (last time step of input)
    # x_dict['time'] has shape (T_input, 3)
    # We want (1, 3) for the model
    t_meta = x_batch['time'][:, -1, :]
    
    input_dict = {
        'sentinel2_sequence': x_batch['sentinel2'],
        'cloudmask_sequence': x_batch['cloudmask'],
        'landcover_map': x_batch['landcover'],
        'weather_sequence': x_batch['weather'],
        'temporal_metadata': t_meta
    }
    return input_dict

def run_comparison(model_path='checkpoints/final_model.keras', split='val_chopped',
                   output_dir='predictions', max_samples=None,
                   model=None, generator=None):
    """Generate predictions and compare with EarthNet.
    
    Args:
        model_path: Path to model checkpoint (used only if model is None).
        split: Dataset split to use.
        output_dir: Directory to save predictions.
        max_samples: Maximum samples to process.
        model: Pre-loaded model (optional, avoids redundant loading).
        generator: Pre-loaded DatasetGenerator (optional, avoids redundant loading).
    """
    # 1. Load Model
    if model is None:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, compile=False)
    else:
        print("Using pre-loaded model.")

    # 2. Determine Dataset Path
    if generator is None:
        if split in ['val_chopped', 'val']:
            dataset_path = VALIDATION_PATH
        else:
            dataset_path = DATASET_PATH
        
        print(f"Processing dataset from {dataset_path}...")
        generator = DatasetGenerator(dataset_path)
    else:
        if split in ['val_chopped', 'val']:
            dataset_path = VALIDATION_PATH
        else:
            dataset_path = DATASET_PATH
        print("Using pre-loaded generator.")
    
    # 3. Find files
    files = generator.files
    if max_samples:
        files = files[:max_samples]
        print(f"Processing first {max_samples} samples.")
    
    print(f"Found {len(files)} files.")

    # 4. Generate Predictions
    for i, file_path in enumerate(files):
        try:
            # Load nc file
            with xr.open_dataset(file_path) as ds:
                # Check for minimum length
                if len(ds.time) < generator.input_days + generator.target_days:
                    print(f"Skipping {file_path}: time series too short")
                    continue
                
                # Prepare Inputs
                x_raw = generator.prepare_x(ds)
                
                # Get basic info for reconstruction
                last_input_bap = x_raw['sentinel2'][-1] # (H, W, 4)
                
                # Prepare dict for model (add batch dim, etc)
                model_input = prepare_input_for_model(x_raw)
                
                # Predict
                pred_deltas = model.predict(model_input, verbose=0)
                if isinstance(pred_deltas, list):
                    pred_deltas = pred_deltas[0]
                pred_deltas = pred_deltas[0] # remove batch dim -> (T_out, H, W, 4)
                
                # Calculate Absolute Values
                pred_abs = last_input_bap + pred_deltas
                
                # Compute NDVI
                pred_ndvi = (pred_abs[..., 3] - pred_abs[..., 2]) / (pred_abs[..., 3] + pred_abs[..., 2] + 1e-8)
                
                # 5. Save Prediction
                path_obj = Path(file_path)
                region = path_obj.parent.name
                cubename = path_obj.stem
                
                save_dir = Path(output_dir) / region
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{cubename}.nc"
                
                target_slice = slice(generator.input_days + 4, generator.input_days + generator.target_days, 5)
                target_times = ds.time.isel(time=target_slice).values
                
                # Create xarray dataset
                da = xr.DataArray(
                    pred_ndvi,
                    coords={
                        'time': target_times[:pred_ndvi.shape[0]],
                        'lat': ds.lat if 'lat' in ds.coords else np.arange(128),
                        'lon': ds.lon if 'lon' in ds.coords else np.arange(128)
                    },
                    dims=['time', 'lat', 'lon'],
                    name='ndvi_pred'
                )
                
                # Save to nc
                da.to_netcdf(save_path)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} samples...")

    print("Prediction generation complete.")
    
    # 6. Score
    print("\nStarting EarthNet Scoring...")
    
    try:
        scores = entk.score_over_dataset(str(dataset_path), str(output_dir))
        print("\n" + "="*30)
        print("SCORING RESULTS")
        print("="*30)
        print(f"Vegetation Score (veg_score): {scores.get('veg_score', 'N/A')}")
        print("-" * 30)
        print("All Scores:", scores)
        print("="*30)
        
    except Exception as e:
        print(f"Error during scoring: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Generate predictions and compare with EarthNet")
    parser.add_argument('--model', type=str, default='checkpoints/final_model.keras',
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val_chopped',
                        choices=['train', 'val_chopped', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (for testing)')
    
    args = parser.parse_args()
    
    run_comparison(
        model_path=args.model,
        split=args.split,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
