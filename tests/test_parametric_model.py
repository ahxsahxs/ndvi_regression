
import os
import sys
import tempfile

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
from build_model import build_eo_convlstm_model, load_model

def test_model_build_and_forward():
    """Test that model builds and runs forward pass correctly."""
    print("Building model...")
    model = build_eo_convlstm_model()
    
    print("\nCreating dummy input...")
    # Match actual dataset shapes:
    # input_shape=(10, 128, 128, 4)
    # cloudmask_shape=(10, 128, 128, 1)
    # landcover_shape=(128, 128, 10)
    # temporal_shape=(3,)
    # weather_shape=(20, 21)
    
    batch_size = 2
    img_input = tf.random.normal((batch_size, 10, 128, 128, 4))
    cloudmask_input = tf.zeros((batch_size, 10, 128, 128, 1))
    landcover_input = tf.zeros((batch_size, 128, 128, 10))
    temporal_input = tf.random.normal((batch_size, 3))
    weather_input = tf.random.normal((batch_size, 20, 21))
    
    print("Running forward pass...")
    outputs = model([img_input, cloudmask_input, landcover_input, temporal_input, weather_input])
    
    print(f"Output shape: {outputs.shape}")
    
    expected_shape = (batch_size, 20, 128, 128, 4)
    assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    print("Forward pass test PASSED!")
    return model


def test_model_serialization(model):
    """Test that model can be saved and loaded."""
    print("\nTesting model serialization...")
    
    # Create test input
    batch_size = 1
    img_input = tf.random.normal((batch_size, 10, 128, 128, 4))
    cloudmask_input = tf.zeros((batch_size, 10, 128, 128, 1))
    landcover_input = tf.zeros((batch_size, 128, 128, 10))
    temporal_input = tf.random.normal((batch_size, 3))
    weather_input = tf.random.normal((batch_size, 20, 21))
    test_inputs = [img_input, cloudmask_input, landcover_input, temporal_input, weather_input]
    
    # Get original output
    original_output = model(test_inputs)
    
    # Save model to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.keras')
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        
        print("Loading model...")
        loaded_model = load_model(model_path, compile=False)
        
        print("Running forward pass on loaded model...")
        loaded_output = loaded_model(test_inputs)
        
        # Check outputs match
        diff = tf.reduce_max(tf.abs(original_output - loaded_output))
        print(f"Max output difference after load: {diff.numpy()}")
        
        assert diff < 1e-5, f"Loaded model output differs by {diff.numpy()}"
        print("Serialization test PASSED!")


def test_model():
    """Run all model tests."""
    model = test_model_build_and_forward()
    test_model_serialization(model)
    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_model()
