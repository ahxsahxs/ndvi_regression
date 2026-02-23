"""
GPU Configuration Utility for TensorFlow.

Shared module that configures GPU memory growth and limits.
Extracted from train.py's configure_gpu() so all scripts
(training, analysis, evaluation) use the same GPU setup.

Must be called BEFORE any TensorFlow operations (model loading, etc.).
Falls back gracefully to CPU if no GPU is detected.
"""

import os
import tensorflow as tf

# Suppress verbose TF logging (INFO and WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def configure_gpu(memory_limit_mb=32 * 1024):
    """
    Configure GPU memory growth and optional memory limit.

    - Enables memory growth so TF doesn't pre-allocate all VRAM.
    - Sets a per-GPU memory cap (default 32 GB) to allow sharing
      with other processes.
    - Safe to call multiple times; catches RuntimeError if GPUs
      are already initialized.

    :param memory_limit_mb: Maximum GPU memory in MB (default 32768).
    :type memory_limit_mb: int
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )]
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"[GPU] {len(gpus)} Physical GPU(s), "
                  f"{len(logical_gpus)} Logical GPU(s) configured "
                  f"(limit {memory_limit_mb} MB each)")
        except RuntimeError as e:
            # Memory growth / config must be set before GPU initialization
            print(f"[GPU] Configuration note: {e}")
    else:
        print("[GPU] No GPUs detected — running on CPU.")
