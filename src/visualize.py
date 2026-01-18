import os
# Force CPU to avoid XLA/CUDA JIT errors with Sign op
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from dataset import DatasetGenerator
from losses import MaskedHuberLoss, ImprovedkNDVILoss, kNDVILoss
from build_model import load_model
from config import DATASET_PATH

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

def visualize(model_epoch=None):
    if model_epoch is None:
        model_path = os.path.join(CHECKPOINTS_DIR, "final_model.keras")
    else:
        model_path = os.path.join(CHECKPOINTS_DIR, f"model_at_{model_epoch}.keras")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Create output directory
    output_dir = "images/predictions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading data...")
    generator = DatasetGenerator(DATASET_PATH)
    dataset = generator.get_dataset()
    dataset = dataset.batch(1)
    
    # We need to map inputs to match model expectation
    dataset = dataset.map(adapt_inputs)

    # Iterate through all samples
    for sample_idx, (x_batch, y_batch) in enumerate(dataset):
        print(f"Processing sample {sample_idx}...")
        y_pred = model.predict(x_batch, verbose=0)
        
        # Handle multi-output models (e.g. if model has sign branch)
        if isinstance(y_pred, list):
            print(f"  Model returned list of outputs. Using first one.")
            pred_deltas = y_pred[0]
        else:
            pred_deltas = y_pred
        
        # Extract data from batch
        # X input: sentinel2 sequence - shape (1, T_in, 128, 128, 4) -> bands B02, B03, B04, B8A
        x_sentinel = x_batch['sentinel2_sequence'][0].numpy()  # (T_in, 128, 128, 4)
        
        # Y output: shape (1, 12, 128, 128, 9) 
        # Channels: 0=mask, 1-4=deltas (B02,B03,B04,B8A), 5-8=BAP (B02,B03,B04,B8A)
        y_data = y_batch[0].numpy()  # (12, 128, 128, 9)
        cloudmask = y_data[:, :, :, 0]      # Channel 0 is mask
        true_deltas = y_data[:, :, :, 1:5]  # Channels 1-4 are deltas
        y_bap = y_data[:, :, :, 5:9]        # Channels 5-8 are BAP composite (for RGB)
        
        pred_deltas = pred_deltas[0]  # (12, 128, 128, 4)
        
        # Calculate and print variance of predictions
        delta_std = np.std(pred_deltas, axis=(1, 2)) 
        print(f"  Predicted Delta Spatial Std (Time 0, Band 0): {delta_std[0, 0]:.6f}")
        
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
        
        def make_delta_rgb(data, mask=None):
            """Create RGB visualization from delta bands. data shape: (H, W, 4)
            Masked pixels (mask=1) are shown in magenta."""
            rgb = np.stack([
                data[:, :, 2],  # R = B04 delta
                data[:, :, 1],  # G = B03 delta
                data[:, :, 0],  # B = B02 delta
            ], axis=-1)
            # Normalize deltas: map [-0.1, 0.1] to [0, 1] for visualization
            rgb = np.clip((rgb + 0.1) / 0.2, 0, 1)
            # Apply cloud mask: show masked pixels in magenta
            if mask is not None:
                mask_3d = np.expand_dims(mask, axis=-1)
                magenta = np.array([1.0, 0.0, 1.0])
                rgb = np.where(mask_3d > 0.5, magenta, rgb)
            return rgb
        
        # Select 5 timesteps to display
        n_cols = 5
        timesteps_x = np.linspace(0, x_sentinel.shape[0] - 1, n_cols, dtype=int)
        timesteps_y = np.linspace(0, y_data.shape[0] - 1, n_cols, dtype=int)
        
        # Create 4x5 grid
        fig, axes = plt.subplots(4, n_cols, figsize=(3 * n_cols, 3 * 4))
        fig.suptitle(f"Sample {sample_idx}", fontsize=16)
        
        row_labels = ['X Input (RGB)', 'Y Output (RGB)', 'True Delta (RGB)', 'Pred Delta (RGB)']
        
        for col_idx in range(n_cols):
            # Row 0: X input RGB
            t_x = timesteps_x[col_idx]
            rgb_x = make_rgb(x_sentinel[t_x])
            axes[0, col_idx].imshow(rgb_x)
            axes[0, col_idx].set_title(f"t={t_x}")
            axes[0, col_idx].axis('off')
            
            # Row 1: Y output RGB (BAP composite)
            t_y = timesteps_y[col_idx]
            rgb_y = make_rgb(y_bap[t_y])
            axes[1, col_idx].imshow(rgb_y)
            axes[1, col_idx].set_title(f"t={t_y}")
            axes[1, col_idx].axis('off')
            
            # Row 2: True delta RGB (with cloud mask)
            delta_true_rgb = make_delta_rgb(true_deltas[t_y], cloudmask[t_y])
            axes[2, col_idx].imshow(delta_true_rgb)
            axes[2, col_idx].set_title(f"t={t_y}")
            axes[2, col_idx].axis('off')
            
            # Row 3: Predicted delta RGB (no cloud mask - show raw predictions)
            delta_pred_rgb = make_delta_rgb(pred_deltas[t_y])
            axes[3, col_idx].imshow(delta_pred_rgb)
            axes[3, col_idx].set_title(f"t={t_y}")
            axes[3, col_idx].axis('off')
        
        # Add row labels on the left
        for row_idx, label in enumerate(row_labels):
            axes[row_idx, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=10)
            axes[row_idx, 0].yaxis.set_label_position('left')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{sample_idx:04d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        if sample_idx >= 5:  # Limit to 5 samples
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
    args = parser.parse_args()
    visualize(model_epoch=args.model)
