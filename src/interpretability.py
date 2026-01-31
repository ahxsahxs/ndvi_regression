import os
# Force CPU to avoid XLA/CUDA JIT errors with Sign op if needed, but user said tf-gpu, so let's try default first.
# If OOM or errors occur, we might fall back, but let's stick to env config.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

"""
Interpretability Analysis Script for Growth Curve Parameters.

This script extracts and analyzes the latent growth curve parameters (A, lambda, B) 
predicted by the model for vegetated regions of interest.

Interpretation Logic:
---------------------
The script analyzes the differences between Near-Infrared (NIR) and Red bands to infer vegetation dynamics:

1. Growth Rate (lambda):
   - If lambda_NIR >> lambda_RED: Indicates rapid green-up (canopy structure develops faster than chlorophyll absorption).
   - If lambda_NIR approx lambda_RED: Balanced growth.

2. Amplitude (A):
   - If A_NIR >> A_RED: Indicates high potential biomass/productivity (strong NIR scattering vs Red absorption).
   - If A_NIR approx A_RED: Sparse or stressed vegetation.

3. Baseline (B):
   - High B_NIR: Persistent vegetation signal at the start of the cycle.

Example Output:
---------------
Vegetated ROI: 15770 pixels (96.3%)

Average Parameters in Vegetated ROI:
Band         | A (Ampl)   | λ (Rate)   | B (Offset)
--------------------------------------------------
Blue (B02)   |  -0.0018   |   0.1615   |   0.0018
Green (B03)  |  -0.0047   |   0.1716   |  -0.0007
Red (B04)    |  -0.0034   |   0.1716   |  -0.0091
NIR (B8A)    |  -0.0369   |   0.1720   |  -0.0016

--- Vegetation Dynamics Interpretation ---
Comparison NIR vs RED:
  Amplitude Delta (NIR - RED): -0.0335
  Rate Delta (NIR - RED):      0.0004
  Baseline Delta (NIR - RED):  0.0075

Summary:
- NIR and RED signals grow at comparable rates.
- The vegetation shows low peak biomass (weak NIR dominance), possibly stressed or sparse cover.
""" 

import argparse
import numpy as np
import tensorflow as tf
import keras
from dataset import DatasetGenerator
from build_model import load_model, RegressionParameterHead
from config import DATASET_PATH

# ESA WorldCover 10m v100 classes
# 10: Tree cover, 20: Shrubland, 30: Grassland, 40: Cropland, 
# 50: Built-up, 60: Bare / sparse vegetation, 70: Snow and Ice, 
# 80: Permanent water bodies, 90: Herbaceous wetland, 95: Mangroves, 100: Moss and lichen
VEGETATED_CLASSES = [10, 20, 30, 40, 90, 95, 100]
ESA_CLASSES = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])

CHECKPOINTS_DIR = "/home/me/workspace/bspline_ndvi/checkpoints"

def get_parameter_model(full_model):
    """
    Extracts the part of the model that predicts A, lambda, B.
    
    The full model structure (from build_model.py) is:
    inputs -> ... -> latent_fused -> RegressionParameterHead -> (A, lambda, B) -> ...
    
    We need to find the output of RegressionParameterHead.
    """
    # Find the RegressionParameterHead layer
    reg_layer = None
    for layer in full_model.layers:
        if isinstance(layer, RegressionParameterHead):
            reg_layer = layer
            break
            
    if reg_layer is None:
        raise ValueError("Could not find RegressionParameterHead in the model.")

    # We want the output of this layer.
    # Keras functional API allows us to create a new model from inputs to this layer's output.
    # However, RegressionParameterHead returns a tuple (A, lambda, B).
    
    # Let's inspect the layer output in the full model
    # reg_layer.output should be a list/tuple of tensors if it's connected.
    # Actually, in the build_model code:
    # A, lambda_param, B = RegressionParameterHead(...)(latent_fused)
    
    # We can reconstruct a model that outputs these intermediate tensors.
    # The full model inputs are standard.
    # The outputs we want are the tensors produced by 'regression_head'.
    
    # We can use the layer name if we know it, 'regression_head'.
    try:
        layer_output = full_model.get_layer('regression_head').output
        # layer_output is likely a list [A, lambda, B]
    except ValueError:
        print("Could not get layer 'regression_head' by name. Searching by type...")
        layer_output = reg_layer.output

    return keras.Model(inputs=full_model.inputs, outputs=layer_output)

def get_vegetated_mask(landcover_onehot):
    """
    Creates a binary mask for vegetated pixels.
    landcover_onehot: (H, W, 10) - matching the dataset encoding (excluding class 0 padding if any? dataset.py drops the first col maybe?)
    
    dataset.py says:
    lc_one_hot = np.eye(len(ESA_CLASSES))[lc_indices]
    esawc_lc = lc_one_hot[..., 1:].astype(np.float32) 
    
    Wait, dataset.py line 139: esawc_lc = lc_one_hot[..., 1:]
    This implies it drops the 0th index? 
    ESA_CLASSES has 11 elements. 
    lc_indices are 0..10.
    np.eye(11) -> (11, 11).
    lc_one_hot -> (..., 11).
    [..., 1:] -> (..., 10).
    So it drops the index 0 which corresponds to ESA class 10 (Tree cover)?
    
    Let's re-read dataset.py carefully.
    ESA_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100] (11 items)
    lc_indices = np.searchsorted(ESA_CLASSES, val) -> 0..10
    lc_one_hot has shape (..., 11).
    esawc_lc = lc_one_hot[..., 1:] -> This drops index 0 (Class 10 - Trees)?? 
    
    If it drops index 0, then we lose Trees? That seems like a bug in dataset.py or I am misinterpreting.
    
    Let's check dataset.py again.
    Line 139: `esawc_lc = lc_one_hot[..., 1:].astype(np.float32)`
    If `lc_indices` is 0 (Class 10), one-hot is [1, 0, ...].
    Slicing [1:] gives [0, 0, ...]. So Trees become all-zeros?
    
    If this is true, the model might not be seeing Trees explicitly? Or maybe "all zeros" means Trees?
    Arguments for "all zeros = Trees": Softmax usually requires N classes. If we use 10 channels for 11 classes, the missing one is implicit if they sum to 1?
    But here it's just raw features into ConvLSTM.
    
    Anyway, effectively:
    Input channel 0 -> Index 1 (Class 20)
    Input channel 1 -> Index 2 (Class 30)
    ...
    Input channel 9 -> Index 10 (Class 100)
    
    So "Trees" (Index 0) is indeed the all-zeros vector in the input `landcover` tensor.
    
    WE NEED TO RECONSTRUCT THIS CAREFULLY.
    
    If `landcover` vector sum is 0, it is Class 10 (Trees).
    If `landcover` vector has 1 at pos k, it is Class ESA_CLASSES[k+1].
    
    Let's verify this logic.
    """
    # landcover_onehot: (H, W, 10)
    
    # Check if a pixel is one of the "explicit" classes (indices 1..10 in original array)
    # Original indices:
    # 0: 10 (Trees) -> DROPPED in dataset.py?
    # 1: 20 (Shrubland) -> Channel 0
    # 2: 30 (Grassland) -> Channel 1
    # 3: 40 (Cropland) -> Channel 2
    # 4: 50 (Built-up) -> Channel 3
    # 5: 60 (Bare) -> Channel 4
    # 6: 70 (Snow) -> Channel 5
    # 7: 80 (Water) -> Channel 6
    # 8: 90 (Wetland) -> Channel 7
    # 9: 95 (Mangroves) -> Channel 8
    # 10: 100 (Moss) -> Channel 9
    
    # Vegetated Classes in ESA codes: 10, 20, 30, 40, 90, 95, 100
    # Corresponding to:
    # 10 (Trees): Implicit (sum=0)
    # 20 (Shrubland): Channel 0
    # 30 (Grassland): Channel 1
    # 40 (Cropland): Channel 2
    # 90 (Wetland): Channel 7
    # 95 (Mangroves): Channel 8
    # 100 (Moss): Channel 9
    
    # So we sum channels [0, 1, 2, 7, 8, 9].
    # PLUS pixels where sum(all channels) < 0.5 (which implies Trees).
    
    # Channel indices for vegetated explicit classes
    # 0 (20), 1 (30), 2 (40), 7 (90), 8 (95), 9 (100)
    veg_channels = [0, 1, 2, 7, 8, 9]
    
    is_explicit_veg = tf.reduce_sum(tf.gather(landcover_onehot, veg_channels, axis=-1), axis=-1)
    
    # Is tree? (All channels zero)
    is_tree = 1.0 - tf.reduce_sum(landcover_onehot, axis=-1)
    # Clip to be safe against float errors, though they are 0.0 or 1.0 from dataset
    is_tree = tf.cast(is_tree > 0.5, tf.float32)
    
    # Union
    is_vegetated = tf.clip_by_value(is_explicit_veg + is_tree, 0.0, 1.0)
    
    return is_vegetated

def analyze_sample(model, dataset_iterator, sample_idx=0):
    print(f"\n--- Analyzing Sample {sample_idx} ---")
    
    # Fetch sample
    x_batch, y_batch = next(dataset_iterator)
    # This loop logic in main is better, just fetching one here is sloppy if used in loop.
    # But let's assume valid iterator.
    
    # Predict parameters
    # Model info:
    # A: Amplitude
    # lambda: Rate
    # B: Offset
    # Shape: (Batch, H, W, Bands)
    # Bands: B02(Blue), B03(Green), B04(Red), B8A(NIR)
    
    preds = model.predict(x_batch, verbose=0)
    # properties of functional model outputs: will be list [A, lambda, B]
    pred_A = preds[0]
    pred_lambda = preds[1]
    pred_B = preds[2]
    
    # Get Landcover from input to define ROI
    # x_batch['landcover'] has shape (Batch, 128, 128, 10) based on model input shape, 
    # but dataset.py output signature says (128, 128, 10). 
    # Batch dimension is added by .batch(1) in main.
    lc_batch = x_batch['landcover_map'] # (1, 128, 128, 10)
    
    # Create mask (1, 128, 128)
    veg_mask = get_vegetated_mask(lc_batch)
    veg_mask_np = veg_mask.numpy()
    
    pixel_count = np.sum(veg_mask_np)
    total_pixels = 128 * 128
    print(f"Vegetated ROI: {int(pixel_count)} pixels ({pixel_count/total_pixels:.1%})")
    
    if pixel_count == 0:
        print("No vegetated pixels found in this sample.")
        return

    # Calculate statistics per band
    bands = ["Blue (B02)", "Green (B03)", "Red (B04)", "NIR (B8A)"]
    
    # Storage for band stats
    stats = {
        'A': [],
        'lambda': [],
        'B': []
    }
    
    print("\nAverage Parameters in Vegetated ROI:")
    print(f"{'Band':<12} | {'A (Ampl)':<10} | {'λ (Rate)':<10} | {'B (Offset)':<10}")
    print("-" * 50)
    
    for b_idx, b_name in enumerate(bands):
        # Extract band data: (1, H, W)
        A_band = pred_A[..., b_idx]
        L_band = pred_lambda[..., b_idx]
        B_band = pred_B[..., b_idx]
        
        # Apply mask
        # mask is (1, H, W), data is (1, H, W)
        # We perform boolean indexing
        mask_bool = veg_mask_np > 0.5
        
        mean_A = np.mean(A_band[mask_bool])
        mean_L = np.mean(L_band[mask_bool])
        mean_B = np.mean(B_band[mask_bool])
        
        stats['A'].append(mean_A)
        stats['lambda'].append(mean_L)
        stats['B'].append(mean_B)
        
        print(f"{b_name:<12} | {mean_A:8.4f}   | {mean_L:8.4f}   | {mean_B:8.4f}")
        
    interpret_dynamics(stats)

def interpret_dynamics(stats):
    print("\n--- Vegetation Dynamics Interpretation ---")
    
    # Indices: 2=Red, 3=NIR
    idx_red = 2
    idx_nir = 3
    
    # Amplitude
    A_red = stats['A'][idx_red]
    A_nir = stats['A'][idx_nir]
    
    # Rate
    L_red = stats['lambda'][idx_red]
    L_nir = stats['lambda'][idx_nir]
    
    # Offset
    B_red = stats['B'][idx_red]
    B_nir = stats['B'][idx_nir]
    
    # 1. kNDVI / Greening Logic
    # kNDVI relates to the distance/difference between NIR and Red.
    # If NIR grows faster (Higher lambda) or has higher amplitude, the area becomes "greener".
    
    print(f"Comparison NIR vs RED:")
    print(f"  Amplitude Delta (NIR - RED): {A_nir - A_red:.4f}")
    print(f"  Rate Delta (NIR - RED):      {L_nir - L_red:.4f}")
    print(f"  Baseline Delta (NIR - RED):  {B_nir - B_red:.4f}")
    
    conclusions = []
    
    # Analyze Growth Rate
    if L_nir > L_red * 1.1:
        conclusions.append("The NIR signal grows significantly faster than the RED signal, indicating rapid green-up.")
    elif L_nir < L_red * 0.9:
        conclusions.append("The NIR signal grows slower than the RED signal, suggesting delayed or weak vegetation development.")
    else:
        conclusions.append("NIR and RED signals grow at comparable rates.")
        
    # Analyze Amplitude (Capacity/Biomass)
    if A_nir > A_red + 0.1:
        conclusions.append("The vegetation reaches a high peak biomass (strong NIR dominance vs Red at peak).")
    elif A_nir < A_red:
        conclusions.append("The vegetation shows low peak biomass (weak NIR dominance), possibly stressed or sparse cover.")
        
    # Analyze Baseline
    if B_nir > B_red + 0.1:
        conclusions.append("There is a strong persistent vegetation signal (high baseline NIR) even at the start of the cycle.")
        
    print("\nSummary:")
    for c in conclusions:
        print(f"- {c}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=None, help="Epoch to load (default: final_model)")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to analyze")
    args = parser.parse_args()
    
    if args.model is None:
        model_path = os.path.join(CHECKPOINTS_DIR, "final_model.keras")
    else:
        model_path = os.path.join(CHECKPOINTS_DIR, f"model_at_{args.model}.keras")
        
    print(f"Loading full model from {model_path}...")
    try:
        full_model = load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Extracting parameter sub-model...")
    try:
        param_model = get_parameter_model(full_model)
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return
        
    print("Loading dataset...")
    generator = DatasetGenerator(DATASET_PATH)
    dataset = generator.get_dataset().batch(1)
    
    # Need to adapt inputs just like in visualize.py/train.py
    def adapt_inputs(x, y):
        # Same logic as train.py
        t_meta = x['time'][:, -1, :]
        x_new = {
            'sentinel2_sequence': x['sentinel2'],
            'cloudmask_sequence': x['cloudmask'],
            'landcover_map': x['landcover'],
            'weather_sequence': x['weather'],
            'temporal_metadata': t_meta
        }
        return x_new, y
    
    dataset = dataset.map(adapt_inputs)
    iterator = iter(dataset)
    
    for i in range(args.samples):
        analyze_sample(param_model, iterator, sample_idx=i)

if __name__ == "__main__":
    main()
