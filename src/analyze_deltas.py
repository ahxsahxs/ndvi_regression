
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataset import DatasetGenerator
from config import DATASET_PATH

def analyze_deltas():
    print("Initializing dataset generator...")
    generator = DatasetGenerator(DATASET_PATH)
    dataset = generator.get_dataset()
    
    # Analyze 100 samples
    n_samples = 100
    dataset = dataset.take(n_samples)
    
    # Store aggregated deltas per band
    # Bands: B02, B03, B04, B8A
    band_names = ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B8A (NIR)']
    all_deltas = [[] for _ in range(4)]
    
    print(f"Analyzing {n_samples} samples...")
    for idx, (x_batch, y_batch) in tqdm(enumerate(dataset), total=n_samples):
        # y_batch structure: [Mask(1), Deltas(4), BAP(4)]
        # We need Deltas: indices 1:5
        # y_batch is a single sample tensor (T, H, W, 9)
        deltas = y_batch[..., 1:5].numpy() # (T, H, W, 4)
        mask = y_batch[..., 0:1].numpy() # (T, H, W, 1)
        
        # Only consider valid pixels (mask == 0 is clear/valid)
        valid_mask = (mask == 0).squeeze() # (T, H, W)
        
        for b in range(4):
            band_deltas = deltas[..., b] # (T, H, W)
            valid_deltas = band_deltas[valid_mask]
            
            # Subsample if too large to save memory
            if len(valid_deltas) > 10000:
                valid_deltas = np.random.choice(valid_deltas, 10000, replace=False)
                
            all_deltas[b].extend(valid_deltas)

    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    print("\n--- Delta Distribution Statistics ---")
    
    for b in range(4):
        data = np.array(all_deltas[b])
        data = data[~np.isnan(data)] # Remove NaNs
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        n_pos = np.sum(data > 0)
        n_neg = np.sum(data < 0)
        n_zero = np.sum(data == 0)
        total = len(data)
        
        pct_pos = (n_pos / total) * 100
        pct_neg = (n_neg / total) * 100
        pct_zero = (n_zero / total) * 100
        
        print(f"\nBand {band_names[b]}:")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  Pos: {pct_pos:.1f}% | Neg: {pct_neg:.1f}% | Zero: {pct_zero:.1f}%")
        
        # Plot
        ax = axes[b]
        ax.hist(data, bins=50, range=(-0.2, 0.2), color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"{band_names[b]}\nPos: {pct_pos:.1f}% vs Neg: {pct_neg:.1f}%")
        ax.set_xlabel('Delta Reflectance')
        ax.set_ylabel('Count')
        ax.axvline(0, color='red', linestyle='--')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'images/delta_analysis.png'
    os.makedirs('images', exist_ok=True)
    plt.savefig(output_path)
    print(f"\nHistogram saved to {output_path}")

if __name__ == "__main__":
    analyze_deltas()
