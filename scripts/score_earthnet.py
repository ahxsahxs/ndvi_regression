import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
# Patch Dense layer to ignore quantization_config from older/newer keras versions
original_dense_from_config = keras.layers.Dense.from_config
@classmethod
def patched_dense_from_config(cls, config):
    if "quantization_config" in config:
        del config["quantization_config"]
    return original_dense_from_config(config)
keras.layers.Dense.from_config = patched_dense_from_config

import xarray as xr
from tqdm import tqdm
import earthnet as entk
import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import DatasetGenerator
from build_model import load_model
from config import DATASET_PATH, VALIDATION_PATH
from gpu_config import configure_gpu

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'preds')

def score_predictions(model_epoch=None, split='val_chopped', limit=None):
    configure_gpu()
    if model_epoch is None:
        model_path = os.path.join(CHECKPOINTS_DIR, "final_model.keras")
    else:
        model_path = os.path.join(CHECKPOINTS_DIR, f"model_at_{model_epoch}.keras")
        
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, compile=False)

    dataset_path = VALIDATION_PATH if split in ['val', 'val_chopped'] else DATASET_PATH
    
    print(f"Loading {split} dataset from {dataset_path}...")
    generator = DatasetGenerator(dataset_path)

    pred_dir_full = os.path.join(PRED_DIR, f"model_{model_epoch if model_epoch else 'final'}_{split}")
    os.makedirs(pred_dir_full, exist_ok=True)

    print(f"Predictions will be saved to: {pred_dir_full}")

    valid_files = 0
    
    file_list = generator.files
    if limit is not None:
        file_list = file_list[:limit]

    target_wrapper_dir = os.path.join(PRED_DIR, f"targets_wrapper_{split}")
    os.makedirs(target_wrapper_dir, exist_ok=True)

    for file_path in tqdm(file_list, desc="Generating predictions and wrappers"):
        try:
            with xr.open_dataset(file_path) as ds:
                if len(ds.time) < generator.input_days + generator.target_days:
                    continue
                
                fname = os.path.basename(file_path)
                
                # EarthNet's `score_over_dataset` reads files from the targets directory.
                # It takes the parent folder of the target as the `region`, and looks for 
                # predictions in `pred_dir / region / cubename`.
                # Since we are placing target wrappers directly in `target_wrapper_dir`,
                # the region EarthNet perceives will be `os.path.basename(target_wrapper_dir)`.
                region = os.path.basename(target_wrapper_dir)
                
                region_dir = os.path.join(pred_dir_full, region)
                os.makedirs(region_dir, exist_ok=True)
                out_path = os.path.join(region_dir, fname)
                
                if os.path.exists(out_path):
                    valid_files += 1
                    continue
                
                x = generator.prepare_x(dataset=ds)
                last_img = generator.compute_bap(x["sentinel2"], x["cloudmask"])
                
                x_batch = {k: np.expand_dims(v, axis=0) for k, v in x.items()}
                
                t_meta = x_batch['time'][:, -1, :]
                x_new = {
                    'sentinel2_sequence': x_batch['sentinel2'],
                    'landcover_map': x_batch['landcover'],
                    'weather_sequence': x_batch['weather'],
                    'temporal_metadata': t_meta,
                    'target_start_doy': x_batch['target_start_doy']
                }
                
                y_pred = model.predict(x_new, verbose=0)
                if isinstance(y_pred, list):
                    pred_deltas = y_pred[0]
                else:
                    pred_deltas = y_pred
                
                pred_deltas = pred_deltas[0] 
                
                T_out = pred_deltas.shape[0]
                bap_tiled = np.tile(np.expand_dims(last_img, 0), (T_out, 1, 1, 1))
                pred_reconstructed = pred_deltas + bap_tiled  
                
                red = pred_reconstructed[:, :, :, 2]
                nir = pred_reconstructed[:, :, :, 3]
                ndvi_pred = (nir - red) / (nir + red + 1e-8)
                
                target_slice = slice(generator.input_days + 4, generator.input_days + generator.target_days, 5)
                target_times = ds.time.isel(time=target_slice).values

                # Extract the s2_dlmask (or s2_mask) from the original target file
                if 's2_dlmask' in ds:
                    actual_mask = ds.s2_dlmask.isel(time=target_slice).values
                elif 's2_mask' in ds:
                    actual_mask = ds.s2_mask.isel(time=target_slice).values
                else:
                    actual_mask = np.zeros_like(ndvi_pred)

                
                lon_dim = 'lon' if 'lon' in ds.dims else 'x'
                lat_dim = 'lat' if 'lat' in ds.dims else 'y'
                
                pred_ds = xr.Dataset(
                    data_vars=dict(
                        ndvi_pred=(["time", lat_dim, lon_dim], ndvi_pred),
                        s2_mask=(["time", lat_dim, lon_dim], actual_mask),
                    ),
                    coords=dict(
                        time=target_times,
                    )
                )
                
                if lat_dim in ds.coords:
                    pred_ds.coords[lat_dim] = ds.coords[lat_dim].values
                if lon_dim in ds.coords:
                    pred_ds.coords[lon_dim] = ds.coords[lon_dim].values
                
                pred_ds.to_netcdf(out_path)
                valid_files += 1

                target_wrapper_path = os.path.join(target_wrapper_dir, fname)
                if not os.path.exists(target_wrapper_path):
                    # EarthNet needs s2_mask in the target dataset
                    ds_copy = ds.copy(deep=True)
                    if 's2_dlmask' in ds_copy:
                        ds_copy['s2_mask'] = ds_copy['s2_dlmask']
                    ds_copy.to_netcdf(target_wrapper_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Generated predictions for {valid_files} cubes.")
    
    print(f"Running EarthNet scoring on {target_wrapper_dir} vs {pred_dir_full}...")
    try:
        scores = entk.score_over_dataset(target_wrapper_dir, pred_dir_full, additional_metrics=True)
        print("Scores:")
        print(scores)

        for key in scores:
            if isinstance(scores[key], pd.DataFrame):
                scores[key].to_csv(os.path.join(pred_dir_full, f"{key}.csv"), index=False)
                del scores[key]
        
        import json
        with open(os.path.join(pred_dir_full, "scores.json"), "w") as f:
            json.dump(scores, f, indent=4)
        print(f"Saved scores to {os.path.join(pred_dir_full, 'scores.json')}")
            
    except Exception as e:
        print(f"Scoring error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=None, help="Epoch to load")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files for testing")
    args = parser.parse_args()
    score_predictions(model_epoch=args.model, split=args.split, limit=args.limit)
