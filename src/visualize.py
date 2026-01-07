import os
# Force CPU to avoid XLA/CUDA JIT errors with Sign op
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from dataset import DatasetGenerator
from losses import MaskedHuberLoss, ImprovedkNDVILoss, kNDVILoss
from build_model import (
    build_eo_convlstm_model, 
    BSplineBasisLayer, 
    BroadcastFusionLayer
)
from config import DATASET_PATH

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

def visualize():
    model_path = "/home/me/workspace/bspline_ndvi/checkpoints/final_model.keras"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Create output directory
    output_dir = "images/predictions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"Loading full model from {model_path}...")
    try:
        # MONKEYPATCH: Handle 'quantization_config' argument mismatch
        # The checkpoint seems to use a version of Keras with quantization support,
        # but the current environment does not. We patch Dense.__init__ to ignore it.
        original_dense_init = keras.layers.Dense.__init__
        def patched_dense_init(self, *args, **kwargs):
            if 'quantization_config' in kwargs:
                kwargs.pop('quantization_config')
            original_dense_init(self, *args, **kwargs)
            
        # Apply patch
        keras.layers.Dense.__init__ = patched_dense_init

        # Define valid custom objects
        custom_objects = {
            'BSplineBasisLayer': BSplineBasisLayer,
            'BroadcastFusionLayer': BroadcastFusionLayer,
            'MaskedHuberLoss': MaskedHuberLoss,
            'ImprovedkNDVILoss': ImprovedkNDVILoss,
            'kNDVILoss': kNDVILoss
        }
        
        # Load the model with its own architecture
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully.")
        
        # Revert patch
        keras.layers.Dense.__init__ = original_dense_init
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
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
            # Assuming first output is the deltas (main output)
            # Adjust if your architecture is different.
            # Usually regression output is 1st or named.
            # Let's inspect shape to be sure?
            # For now, taking [0] is a safe bet for regression part if list.
            print(f"  Model returned list of outputs. Using first one.")
            pred_deltas = y_pred[0]
        else:
            pred_deltas = y_pred
            
        # Check shapes
        # y_batch shape: (1, 12, 128, 128, 9)
        # pred_deltas shape: (1, 12, 128, 128, 4)
        
        true_deltas = y_batch[0, :, :, :, 1:5].numpy()  # Channels 1-4 are deltas
        cloudmask = y_batch[0, :, :, :, 0].numpy()     # Channel 0 is mask
        
        pred_deltas = pred_deltas[0]                   # (12, 128, 128, 4)
        
        # Plotting
        # We'll plot 3 timesteps: Start, Middle, End
        timesteps = [0, 5, 11]
        # Let's plot B04 (Red) -> index 2
        band_idx = 2
        band_name = "B04"

        rows = len(timesteps)
        cols = 3  # True, Pred, Cloudmask
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        fig.suptitle(f"Delta Predictions ({band_name}) - Sample {sample_idx}", fontsize=16)
        
        for i, t in enumerate(timesteps):
            # True
            ax_true = axes[i, 0]
            val_min, val_max = -0.1, 0.1
            im_true = ax_true.imshow(true_deltas[t, :, :, band_idx], cmap='RdBu', vmin=val_min, vmax=val_max)
            ax_true.set_title(f"True Delta (t={t})")
            plt.colorbar(im_true, ax=ax_true)
            
            # Pred
            ax_pred = axes[i, 1]
            im_pred = ax_pred.imshow(pred_deltas[t, :, :, band_idx], cmap='RdBu', vmin=val_min, vmax=val_max)
            ax_pred.set_title(f"Pred Delta (t={t})")
            plt.colorbar(im_pred, ax=ax_pred)
            
            # Mask
            ax_mask = axes[i, 2]
            im_mask = ax_mask.imshow(cloudmask[t, :, :], cmap='gray', vmin=0, vmax=1)
            ax_mask.set_title(f"Cloudmask (t={t})")
            plt.colorbar(im_mask, ax=ax_mask)
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{sample_idx:04d}.png")
        plt.savefig(save_path)
        plt.close()
        
        if sample_idx >= 5: # Limit to 5 samples
            break
    
    print(f"Generated images in {output_dir}")

if __name__ == "__main__":
    visualize()
