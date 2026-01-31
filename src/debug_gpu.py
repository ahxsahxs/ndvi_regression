
import tensorflow as tf
import os

def check_gpu():
    print("TensorFlow Version:", tf.__version__)
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nDetected {len(gpus)} Physical GPUs:")
    for gpu in gpus:
        print(f"  - {gpu}")
        
    if not gpus:
        print("\nERROR: No GPUs detected!")
        return

    # Try to configure them (emulating the fix)
    print("\nAttempting to configure GPUs with memory growth and 32GB limit...")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=32*1024)]
                )
        print("Success: Configuration applied (or already set if initialized).")
    except RuntimeError as e:
        print(f"Configuration failed (might be already initialized): {e}")

    # Run a simple computation
    print("\nRunning test computation on GPU...")
    try:
        with tf.device('/device:GPU:0'):
            # Create random tensors
            a = tf.random.normal((1000, 1000))
            b = tf.random.normal((1000, 1000))
            # Matrix multiplication
            c = tf.matmul(a, b)
            
            # Conv2D test (often triggers CuDNN)
            input_shape = (1, 128, 128, 4)
            x = tf.random.normal(input_shape)
            layer = tf.keras.layers.Conv2D(32, 3, padding='same')
            y = layer(x)
            
            print(f"MatMul Result Shape: {c.shape}")
            print(f"Conv2D Result Shape: {y.shape}")
            print("\nSUCCESS: Basic GPU operations executed without error.")
            
    except Exception as e:
        print(f"\nFAILURE: Execution failed with error:\n{e}")

if __name__ == "__main__":
    check_gpu()
