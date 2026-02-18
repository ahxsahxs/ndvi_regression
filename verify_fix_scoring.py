
import xarray as xr
import numpy as np
import os

def compute_kndvi(nir, red, sigma=1):
    """Compute kernel NDVI using RBF kernel."""
    diff_sq = (nir - red) ** 2
    k = np.exp(-diff_sq / (2 * sigma ** 2))
    kndvi = (1 - k) / (1 + k + 1e-8)
    return kndvi

def calculate_score(targ_path, pred_path):
    print(f"Loading Target: {targ_path}")
    targ = xr.open_dataset(targ_path)
    print(f"Loading Prediction: {pred_path}")
    pred = xr.open_dataset(pred_path)

    # Logic from patched normalized_NSE
    if 's2_mask' not in targ and 's2_dlmask' in targ:
        targ = targ.rename({'s2_dlmask': 's2_mask'})

    # Align time based on prediction length
    pred_start_idx = len(targ.time.isel(time=slice(4, None, 5))) - len(pred.time)
    
    # Extract bands for Target
    nir = targ.s2_B8A.isel(time=slice(4, None, 5)).isel(
        time=slice(pred_start_idx, None)
    )
    red = targ.s2_B04.isel(time=slice(4, None, 5)).isel(
        time=slice(pred_start_idx, None)
    )
    mask = targ.s2_mask.isel(time=slice(4, None, 5)).isel(
        time=slice(pred_start_idx, None)
    )

    # FIX: Use kNDVI for targets
    print("Computing kNDVI for Target...")
    targ_ndvi = compute_kndvi(nir, red).where(mask == 0, np.nan)
    
    pred_ndvi = pred['ndvi_pred']

    # Shapes check
    print(f"Target NDVI Shape: {targ_ndvi.shape}")
    print(f"Pred NDVI Shape: {pred_ndvi.shape}")

    # NNSE Calculation
    numerator = ((targ_ndvi - pred_ndvi) ** 2).sum("time")
    denominator = ((targ_ndvi - targ_ndvi.mean("time")) ** 2).sum("time")
    
    nse = 1 - (numerator / denominator)
    
    # NNSE formula: 1 / (2 - NSE)
    nnse = 1 / (2 - nse)
    
    # Filter valid pixels (where mask is 0 at least once? No, n_obs logic)
    # The sum("time") handles NaNs if skipna=True (default in xarray)
    # But we need to be careful about pixels with no valid observations
    
    # Mean NNSE
    mean_nnse = nnse.mean().item()
    print(f"Mean NNSE: {mean_nnse}")
    
    veg_score = 2 - (1 / mean_nnse)
    print(f"Vegetation Score (Approx): {veg_score}")

    return veg_score

if __name__ == "__main__":
    # Paths based on my previous exploration
    targ_path = "/home/me/workspace/probformer/data/greenearthnet/val_chopped/JAS20/minicube_0_29SND_39.29_-8.56.nc"
    pred_path = "/home/me/workspace/bspline_ndvi/predictions_test/JAS20/minicube_0_29SND_39.29_-8.56.nc"
    
    if os.path.exists(targ_path) and os.path.exists(pred_path):
        calculate_score(targ_path, pred_path)
    else:
        print("Files not found.")
