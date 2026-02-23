import os
import argparse
import tensorflow as tf
import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import DatasetGenerator
from build_model import load_model
from config import DATASET_PATH, VALIDATION_PATH

CHECKPOINTS_DIR = "/home/me/workspace/bspline_ndvi/checkpoints"

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

def visualize(model_epoch=None, split='train', model=None, generator=None):
    """Visualize model predictions as target vs predicted reflectance images.
    
    Args:
        model_epoch: Epoch to load (default: final_model).
        split: Dataset split (train/val_chopped).
        model: Pre-loaded model (optional, avoids redundant loading).
        generator: Pre-loaded DatasetGenerator (optional, avoids redundant loading).
    """
    if model is None:
        if model_epoch is None:
            model_path = os.path.join(CHECKPOINTS_DIR, "final_model.keras")
        else:
            model_path = os.path.join(CHECKPOINTS_DIR, f"model_at_{model_epoch}.keras")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return

        print(f"Loading model from {model_path}...")
        try:
            model = load_model(model_path, compile=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("Using pre-loaded model.")

    # Create output directory
    output_dir = "images/predictions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if generator is None:
        # Select dataset path
        if split in ['val_chopped', 'val']:
            dataset_path = VALIDATION_PATH
        else:
            dataset_path = DATASET_PATH

        print(f"Loading {split} dataset from {dataset_path}...")
        generator = DatasetGenerator(dataset_path)
    else:
        print("Using pre-loaded generator.")
    
    dataset = generator.get_dataset()
    dataset = dataset.batch(1)
    
    # We need to map inputs to match model expectation
    dataset = dataset.map(adapt_inputs)

    # Iterate through all samples
    for sample_idx, (x_batch, y_batch) in enumerate(dataset):
        print(f"Processing sample {sample_idx}...")
        
        try:
            intermediate_model = keras.models.Model(
                inputs=model.inputs,
                outputs=[model.output, model.get_layer('fourier_head').output]
            )
            y_pred, coeffs_pred = intermediate_model.predict(x_batch, verbose=0)
        except Exception as e:
            print(f"Could not extract fourier_head: {e}")
            y_pred = model.predict(x_batch, verbose=0)
            coeffs_pred = None
            
        # Handle multi-output models
        if isinstance(y_pred, list):
            print(f"  Model returned list of outputs. Using first one.")
            pred_deltas = y_pred[0]
        else:
            pred_deltas = y_pred
        
        # Extract data from batch
        # Y output: shape (1, T, 128, 128, 9) 
        # Channels: 0=mask, 1-4=deltas (B02,B03,B04,B8A), 5-8=BAP (B02,B03,B04,B8A)
        y_data = y_batch[0].numpy()  # (T, 128, 128, 9)
        true_deltas = y_data[:, :, :, 1:5]  # Channels 1-4 are deltas
        y_bap = y_data[:, :, :, 5:9]        # Channels 5-8 are BAP composite
        
        pred_deltas = pred_deltas[0]  # (T, 128, 128, 4)
        if coeffs_pred is not None:
            coeffs_pred = coeffs_pred[0] # (H, W, bands, 2K)
        
        # Helper to create RGB from bands (B02=Blue, B03=Green, B04=Red)
        # Band order in data: 0=B02, 1=B03, 2=B04, 3=B8A
        def make_rgb(data, clip_min=0.0, clip_max=0.3):
            """Create RGB image from bands. data shape: (H, W, 4)"""
            rgb = np.stack([
                data[:, :, 2],  # R = B04
                data[:, :, 1],  # G = B03
                data[:, :, 0],  # B = B02
            ], axis=-1)
            # Normalize to 0-1 range for display
            rgb = np.clip((rgb - clip_min) / (clip_max - clip_min), 0, 1)
            return rgb
        
        # Select 5 timesteps to display
        n_cols = 5
        T_in = 10  # input frames (from dataset config)
        T_out = y_data.shape[0]
        timesteps_y = np.linspace(0, T_out - 1, n_cols, dtype=int)
        
        # Map indices to human-readable day labels: label = (T_in + index + 1) * 5
        day_labels_y = np.array([(T_in + i + 1) * 5 for i in range(T_out)], dtype=int)
        
        # --- Image 1: Target vs Predicted Reflectance ---
        fig, axes = plt.subplots(
            2, n_cols, figsize=(2.2 * n_cols, 3.5),
            gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
        )
        fig.suptitle(f"Reflectance Comparison — Sample {sample_idx}", fontsize=16, fontweight='bold')
        
        for col_idx in range(n_cols):
            t_y = timesteps_y[col_idx]
            
            # Row 0: True reflectance (BAP + true delta)
            true_image = y_bap[t_y] + true_deltas[t_y]
            rgb_true = make_rgb(true_image)
            axes[0, col_idx].imshow(rgb_true)
            axes[0, col_idx].set_title(f"t = {day_labels_y[t_y]}", fontsize=11, fontweight='bold')
            axes[0, col_idx].set_xticks([])
            axes[0, col_idx].set_yticks([])
            
            if col_idx == 0:
                axes[0, col_idx].set_ylabel('Target', fontsize=10)
            
            # Row 1: Predicted reflectance (BAP + pred delta)
            pred_image = y_bap[t_y] + pred_deltas[t_y]
            rgb_pred = make_rgb(pred_image)
            axes[1, col_idx].imshow(rgb_pred)
            axes[1, col_idx].set_xticks([])
            axes[1, col_idx].set_yticks([])
            
            if col_idx == 0:
                axes[1, col_idx].set_ylabel('Predicted', fontsize=10)
        
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_reflectance.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        
        # --- Image 2: Target vs Predicted Deltas ---
        fig, axes = plt.subplots(
            2, n_cols, figsize=(2.2 * n_cols, 3.5),
            gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
        )
        fig.suptitle(f"Deltas Comparison — Sample {sample_idx}", fontsize=16, fontweight='bold')
        
        for col_idx in range(n_cols):
            t_y = timesteps_y[col_idx]
            
            # Shift by 0.15 so 0 becomes visible midpoint for Deltas
            rgb_true_delta = make_rgb(true_deltas[t_y] + 0.15, clip_min=0.0, clip_max=0.3)
            axes[0, col_idx].imshow(rgb_true_delta)
            axes[0, col_idx].set_title(f"t = {day_labels_y[t_y]}", fontsize=11, fontweight='bold')
            axes[0, col_idx].set_xticks([])
            axes[0, col_idx].set_yticks([])
            if col_idx == 0:
                axes[0, col_idx].set_ylabel('Target Deltas', fontsize=10)
            
            rgb_pred_delta = make_rgb(pred_deltas[t_y] + 0.15, clip_min=0.0, clip_max=0.3)
            axes[1, col_idx].imshow(rgb_pred_delta)
            axes[1, col_idx].set_xticks([])
            axes[1, col_idx].set_yticks([])
            if col_idx == 0:
                axes[1, col_idx].set_ylabel('Pred Deltas', fontsize=10)
        
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(output_dir, f"sample_{sample_idx:04d}_deltas.png"), dpi=150)
        plt.close(fig)

        # --- Image 3: Average Harmonic parameters for Vegetated Areas ---
        if coeffs_pred is not None:
            # We locate vegetated pixels using the landcover map
            # x_batch['landcover_map'] has shape (1, 128, 128, 10)
            lc = x_batch['landcover_map'][0].numpy()
            # Non-vegetated classes are: 50 (Built-up), 60 (Bare), 70 (Snow), 80 (Water)
            # which correspond to indices 3, 4, 5, 6 in the 10-channel array.
            non_veg = (lc[:, :, 3] + lc[:, :, 4] + lc[:, :, 5] + lc[:, :, 6]) > 0.5
            veg_mask = ~non_veg
            
            if np.sum(veg_mask) > 0:
                coeffs_veg = coeffs_pred[veg_mask] # (N_veg, bands, 2K)
                avg_coeffs = np.mean(coeffs_veg, axis=0) # (bands, 2K)
                
                T_plot = 73
                K = avg_coeffs.shape[-1] // 2
                omega = 2.0 * np.pi / T_plot
                t_arr = np.arange(1, T_plot + 1)
                
                band_names = ["B02", "B03", "B04", "B8A"]
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle(f"Fourier Harmonics for Vegetated Areas (NDVI > 0.4)\nSample {sample_idx}", fontsize=14)
                
                for b_idx in range(4):
                    ax = axes[b_idx // 2, b_idx % 2]
                    coeffs_b = avg_coeffs[b_idx] # (2K,)
                    
                    harmonic_curve = np.zeros_like(t_arr, dtype=np.float32)
                    for k in range(K):
                        freq = (k + 1) * omega
                        harmonic_curve += coeffs_b[2*k] * np.cos(freq * t_arr) + coeffs_b[2*k+1] * np.sin(freq * t_arr)
                    
                    ax.plot(t_arr * 5, harmonic_curve, label=f"{band_names[b_idx]} Harmonic", color='green', linewidth=2)
                    ax.set_title(band_names[b_idx])
                    ax.set_xlabel("Day of Year")
                    ax.set_ylabel("Delta Synthesis")
                    
                    param_text = []
                    for k in range(K):
                        param_text.append(f"k={k+1}: a={coeffs_b[2*k]:.3f}, b={coeffs_b[2*k+1]:.3f}")
                    ax.annotate("\n".join(param_text), xy=(0.02, 0.98), xycoords='axes fraction',
                                verticalalignment='top', fontsize=9,
                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                    
                    ax.grid(True, linestyle='--', alpha=0.6)
                
                fig.tight_layout(rect=[0, 0, 1, 0.92])
                fig.savefig(os.path.join(output_dir, f"sample_{sample_idx:04d}_harmonics.png"), dpi=150)
                plt.close(fig)
            else:
                print(f"Skipping harmonic plot for sample {sample_idx}, no vegetated pixels found.")
        
        if sample_idx >= 10:  # Limit to 10 samples
            break
    
    print(f"Generated images in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions.")
    parser.add_argument(
        "--model",
        type=int,
        default=None,
        help="Epoch number to load model_at_{epoch}.keras. If not specified, loads final_model.keras."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_chopped",
        help="Dataset split (train/val_chopped)."
    )
    args = parser.parse_args()
    
    visualize(model_epoch=args.model, split=args.split)
