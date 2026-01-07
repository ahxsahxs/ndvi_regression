
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
import numpy as np
from losses import kNDVILoss

def test_kndvi_loss():
    print("Testing kNDVILoss...")
    
    # Dimensions
    B, T, H, W = 2, 5, 32, 32
    
    # 1. Mock inputs
    # y_pred: Deltas (4 channels)
    y_pred = tf.random.normal((B, T, H, W, 4))
    
    # y_true: [Mask(1), Deltas(4), BAP(4)] -> 9 channels
    mask = tf.zeros((B, T, H, W, 1)) # All valid
    deltas = tf.random.normal((B, T, H, W, 4))
    bap = tf.random.normal((B, T, H, W, 4))
    
    y_true = tf.concat([mask, deltas, bap], axis=-1)
    
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_true shape: {y_true.shape}")
    
    # 2. Instantiate Loss
    loss_fn = kNDVILoss(alpha=1.0, beta=1.0, sigma=0.5)
    
    # 3. Compute Loss
    loss_val = loss_fn(y_true, y_pred)
    print(f"Loss value: {loss_val.numpy()}")
    
    assert not np.isnan(loss_val.numpy()), "Loss is NaN"
    assert loss_val.numpy() > 0, "Loss should be positive"
    
    # 4. Gradient Check (Validity)
    with tf.GradientTape() as tape:
        tape.watch(y_pred)
        loss = loss_fn(y_true, y_pred)
    
    grads = tape.gradient(loss, y_pred)
    print("Gradients computed.")
    assert grads is not None, "Gradients are None"
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    test_kndvi_loss()
