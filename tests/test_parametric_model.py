
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
from build_model import build_eo_convlstm_model

def test_model():
    print("Building model...")
    model = build_eo_convlstm_model()
    # model.summary() # Commented out to avoid console width issues during automation
    
    print("\nCreating dummy input...")
    # input_shape=(18, 128, 128, 4)
    # cloudmask_shape=(18, 128, 128, 1)
    # landcover_shape=(128, 128, 10)
    # temporal_shape=(3,)
    # weather_shape=(12, 21)
    
    batch_size = 2
    img_input = tf.random.normal((batch_size, 18, 128, 128, 4))
    cloudmask_input = tf.zeros((batch_size, 18, 128, 128, 1))
    landcover_input = tf.zeros((batch_size, 128, 128, 10))
    temporal_input = tf.random.normal((batch_size, 3))
    weather_input = tf.random.normal((batch_size, 12, 21))
    
    print("Running forward pass...")
    outputs = model([img_input, cloudmask_input, landcover_input, temporal_input, weather_input])
    
    print(f"Output shape: {outputs.shape}")
    
    expected_shape = (batch_size, 12, 128, 128, 4)
    assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    print("Verification Successful!")

if __name__ == "__main__":
    test_model()
