from models import ConvLSTMSplineModel
from config import MODEL_CONFIG
from losses import SplineLoss

import keras
from keras import backend as K
import numpy as np
import tensorflow as tf

# Build cubic spline model for comparison
MODEL = ConvLSTMSplineModel(**MODEL_CONFIG)

DUMMY_INPUT = {
    'time': tf.zeros((1, 10, 3)),
    'sentinel2': tf.zeros((1, 10, 128, 128, 4)),
    'cloudmask': tf.zeros((1, 10, 128, 128, 1)),
    'landcover': tf.zeros((1, 128, 128, 10)),
}
MODEL(DUMMY_INPUT)
MODEL.summary()


# %%

def fit_dummy_model():
    # Compile
    MODEL.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=SplineLoss(smoothness_weight=0.01, curvature_weight=0.005),
        metrics=['mae', 'mse']
    )
    
    # Test with dummy data
    print("\n" + "=" * 60)
    print("Testing Model")
    print("=" * 60)
    
    INPUT_FRAMES = MODEL_CONFIG["input_frames"]
    OUTPUT_FRAMES = MODEL_CONFIG["output_frames"]
    IMG_SIZE = MODEL_CONFIG["img_size"]
    CHANNELS = MODEL_CONFIG["out_channels"]
    BATCH_SIZE = 1
    
    dummy_input = {
        'time': np.random.randn(BATCH_SIZE, INPUT_FRAMES, 3).astype(np.float32),
        'sentinel2': np.random.randn(BATCH_SIZE, INPUT_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS).astype(np.float32),
        'cloudmask': np.random.randn(BATCH_SIZE, INPUT_FRAMES, IMG_SIZE, IMG_SIZE, 1).astype(np.float32),
        'landcover': np.random.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 10).astype(np.float32),
    }
    dummy_target = {
        'deltas': np.random.randn(BATCH_SIZE, OUTPUT_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS).astype(np.float32),
        'cloudmask': np.random.randn(BATCH_SIZE, OUTPUT_FRAMES, IMG_SIZE, IMG_SIZE, 1).astype(np.float32),
    }

    predictions = MODEL(dummy_input, training=False)
    print(f"Input shape: {dummy_input['sentinel2'].shape}")
    print(f"Output shape: {predictions['deltas'].shape}")
    print(f"Target shape: {dummy_target['deltas'].shape}")
    
    # Compute loss
    loss_fn = SplineLoss(smoothness_weight=0.01, curvature_weight=0.005)
    loss = loss_fn(dummy_target, predictions)
    print(f"Loss: {loss.numpy():.4f}")
    
    print("\n" + "=" * 60)
    print("Training Example")
    print("=" * 60)
    
    MODEL.fit(
        dummy_input,
        dummy_target,
        batch_size=BATCH_SIZE,
        epochs=5,
        verbose=1
    )
    
fit_dummy_model()

# %%
del MODEL
K.clear_session()