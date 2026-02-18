import os
# Force CPU to avoid XLA/CUDA JIT errors with Sign op
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from dataset import DatasetGenerator
from losses import MaskedHuberLoss, ImprovedkNDVILoss, kNDVILoss
from build_model import load_model, RegressionParameterHead
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
    """Visualize model predictions.
    
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
    else:
        print("Using pre-loaded model.")
        output_dir = "images/predictions"
        os.makedirs(output_dir, exist_ok=True)

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
        T_in = x_sentinel.shape[0]
        T_out = y_data.shape[0]
        timesteps_x = np.linspace(0, T_in - 1, n_cols, dtype=int)
        timesteps_y = np.linspace(0, T_out - 1, n_cols, dtype=int)
        
        # Map indices to human-readable day labels: label = (index + 1) * 5
        day_labels_x = np.array([(i + 1) * 5 for i in range(T_in)], dtype=int)
        day_labels_y = np.array([(T_in + i + 1) * 5 for i in range(T_out)], dtype=int)
        
        # --- Figure 1: Reflectance Comparison ---
        fig1, axes1 = plt.subplots(
            2, n_cols, figsize=(2.2 * n_cols, 3.5),
            gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
        )
        fig1.suptitle(f"Reflectance Comparison — Sample {sample_idx}", fontsize=16, fontweight='bold')
        
        for col_idx in range(n_cols):
            t_y = timesteps_y[col_idx]
            
            # Row 0: True reflectance (BAP + true delta)
            true_image = y_bap[t_y] + true_deltas[t_y]
            rgb_true = make_rgb(true_image)
            axes1[0, col_idx].imshow(rgb_true)
            axes1[0, col_idx].set_title(f"t = {(t_y + 1) * 5}", fontsize=11, fontweight='bold')
            axes1[0, col_idx].set_xticks([])
            axes1[0, col_idx].set_yticks([])
            
            if col_idx == 0:
                axes1[0, col_idx].set_ylabel('Target', fontsize=10)
            
            # Row 1: Predicted reflectance (BAP + pred delta)
            pred_image = y_bap[t_y] + pred_deltas[t_y]
            rgb_pred = make_rgb(pred_image)
            axes1[1, col_idx].imshow(rgb_pred)
            axes1[1, col_idx].set_xticks([])
            axes1[1, col_idx].set_yticks([])
            
            if col_idx == 0:
                axes1[1, col_idx].set_ylabel('Predicted', fontsize=10)
        
        fig1.tight_layout(rect=[0, 0, 1, 0.92])
        save_path1 = os.path.join(output_dir, f"sample_{sample_idx:04d}_reflectance.png")
        fig1.savefig(save_path1, dpi=150)
        plt.close(fig1)
        
        # --- Figure 2: Deltas ---
        fig2, axes2 = plt.subplots(2, n_cols, figsize=(3 * n_cols, 3 * 2))
        fig2.suptitle(f"Deltas — Sample {sample_idx}", fontsize=16, fontweight='bold')
        
        row_labels_delta = ['True Delta (RGB)', 'Pred Delta (RGB)']
        
        for col_idx in range(n_cols):
            t_y = timesteps_y[col_idx]
            
            # Row 0: True delta RGB (with cloud mask)
            delta_true_rgb = make_delta_rgb(true_deltas[t_y], cloudmask[t_y])
            axes2[0, col_idx].imshow(delta_true_rgb)
            axes2[0, col_idx].set_title(f"t={day_labels_y[t_y]}")
            axes2[0, col_idx].axis('off')
            
            # Row 1: Predicted delta RGB (no cloud mask - show raw predictions)
            delta_pred_rgb = make_delta_rgb(pred_deltas[t_y])
            axes2[1, col_idx].imshow(delta_pred_rgb)
            axes2[1, col_idx].set_title(f"t={day_labels_y[t_y]}")
            axes2[1, col_idx].axis('off')
        
        # Add row labels on the left
        for row_idx, label in enumerate(row_labels_delta):
            axes2[row_idx, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=10)
            axes2[row_idx, 0].yaxis.set_label_position('left')
        
        fig2.tight_layout()
        save_path2 = os.path.join(output_dir, f"sample_{sample_idx:04d}_deltas.png")
        fig2.savefig(save_path2, dpi=150)
        plt.close(fig2)
        
        if sample_idx >= 5:  # Limit to 5 samples
            break
    
    print(f"Generated images in {output_dir}")


def get_parameter_model(full_model):
    """Extract the parameter prediction sub-model."""
    for layer in full_model.layers:
        if isinstance(layer, RegressionParameterHead):
            try:
                layer_output = full_model.get_layer('regression_head').output
            except ValueError:
                layer_output = layer.output
            return keras.Model(inputs=full_model.inputs, outputs=layer_output)
    raise ValueError("Could not find RegressionParameterHead in the model.")


def visualize_params(model_epoch=None, output_path=None, sample_idx=0, split='train',
                     model=None, generator=None):
    """Visualize growth curve parameter maps (A, lambda, B) for Figure 4.4.
    
    Args:
        model_epoch: Epoch to load (default: final_model).
        output_path: Output file path.
        sample_idx: Sample index to visualize.
        split: Dataset split (train/val_chopped).
        model: Pre-loaded full model (optional, avoids redundant loading).
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
            full_model = load_model(model_path, compile=False)
            param_model = get_parameter_model(full_model)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("Using pre-loaded model.")
        param_model = get_parameter_model(model)

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
    
    dataset = generator.get_dataset().batch(1).map(adapt_inputs)
    
    # Get sample
    for i, (x_batch, y_batch) in enumerate(dataset):
        if i < sample_idx:
            continue
        
        print(f"Processing sample {i}...")
        preds = param_model.predict(x_batch, verbose=0)
        
        # preds is [A, lambda, B], each with shape (1, H, W, 4)
        pred_A = preds[0][0]     # (H, W, 4)
        pred_lambda = preds[1][0]
        pred_B = preds[2][0]
        
        # Create 2x3 grid: [NIR Amplitude, Red Amplitude] [NIR Rate, Red Rate] [NIR Offset, Red Offset]
        # Or 2x2 grid showing NIR band parameters
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        band_names = ['Blue (B02)', 'Green (B03)', 'Red (B04)', 'NIR (B8A)']
        
        # NIR band (index 3)
        nir_idx = 3
        red_idx = 2
        
        # Top row: Amplitude (A)
        norm_A = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
        im0 = axes[0, 0].imshow(pred_A[:, :, nir_idx], cmap='RdBu_r', norm=norm_A)
        axes[0, 0].set_title('Amplitude (A) - NIR', fontsize=12)
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
        
        im1 = axes[0, 1].imshow(pred_A[:, :, red_idx], cmap='RdBu_r', norm=norm_A)
        axes[0, 1].set_title('Amplitude (A) - Red', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        
        # Bottom row: Rate (lambda) and Offset (B)
        im2 = axes[1, 0].imshow(pred_lambda[:, :, nir_idx], cmap='viridis', vmin=0, vmax=0.5)
        axes[1, 0].set_title('Rate (λ) - NIR', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
        
        norm_B = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
        im3 = axes[1, 1].imshow(pred_B[:, :, nir_idx], cmap='RdBu_r', norm=norm_B)
        axes[1, 1].set_title('Offset (B) - NIR', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)
        
        plt.suptitle(f'Growth Curve Parameters (Sample {sample_idx})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"Saved: {output_path}")
        else:
            output_dir = "images/results/interpretability"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "parameter_maps.png")
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        
        plt.close()
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions.")
    parser.add_argument(
        "--model",
        type=int,
        default=None,
        help="Epoch number to load model_at_{epoch}.keras. If not specified, loads final_model.keras."
    )
    parser.add_argument(
        "--show-params",
        action='store_true',
        help="Show growth curve parameter maps (A, lambda, B) instead of predictions."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to visualize (for --show-params mode)."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (for --show-params mode)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_chopped",
        help="Dataset split (train/val_chopped)."
    )
    args = parser.parse_args()
    
    if args.show_params:
        visualize_params(model_epoch=args.model, output_path=args.output, sample_idx=args.sample, split=args.split)
    else:
        visualize(model_epoch=args.model, split=args.split)
