import os
from pathlib import Path
import keras
import keras.backend as K
import tensorflow as tf

from dataset import DatasetGenerator
from models import ConvLSTMSplineModel
from losses import SplineLoss
from config import MODEL_CONFIG, DATASET_PATH

def train_bspline_model(
    train_dir,
    val_dir=None,
    batch_size=2,
    epochs=200,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints',
    log_dir='logs/bspline',
    checkpoint_to_resume=None,
    initial_epoch=0
):
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading data from {train_dir}...")

    # 1. Dataset
    generator = DatasetGenerator(train_dir)
    train_dataset = generator.get_dataset()

    # Shuffle and Batch
    train_dataset = train_dataset.shuffle(100).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = val_generator = None
    if val_dir and os.path.exists(val_dir):
        print(f"Loading validation data from {val_dir}...")
        val_generator = DatasetGenerator(val_dir)
        val_dataset = val_generator.get_dataset()
        val_dataset = val_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    # 2. Model - Either load from checkpoint or create new
    if checkpoint_to_resume and os.path.exists(checkpoint_to_resume):
        print(f"Loading model from checkpoint: {checkpoint_to_resume}")
        # Load the full model including optimizer state
        model = keras.models.load_model(
            checkpoint_to_resume
        )
        print("Model loaded successfully. Resuming training...")
        model.summary()
    else:
        if checkpoint_to_resume:
            print(f"Warning: Checkpoint {checkpoint_to_resume} not found. Creating new model...")

        print("Creating ConvFormer model...")
        model = ConvLSTMSplineModel(**MODEL_CONFIG)

        # 3. Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss=SplineLoss(smoothness_weight=0.01, curvature_weight=0.005)

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])

    # 4. Callbacks
    # Save full model (not just weights) to preserve optimizer state
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.keras')

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,  # Save full model including optimizer
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=25,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    # 5. Train
    print(f"Starting training from epoch {initial_epoch}...")

    # Calculate steps per epoch
    steps_per_epoch = max(1, len(generator.files) // batch_size)
    validation_steps = None
    if val_dataset is not None and val_generator:
        validation_steps = max(1, len(val_generator.files) // batch_size)

    print(f"Steps per epoch: {steps_per_epoch}")
    if validation_steps:
        print(f"Validation steps: {validation_steps}")

    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
        return model

    # Save final model (full model with optimizer state)
    final_model_path = os.path.join(checkpoint_dir, 'final_model.keras')
    print(f"Saving final model to {final_model_path}...")
    model.save(final_model_path)

    del model
    K.clear_session()

    print("Training complete.")


if __name__ == "__main__":
    SPLINE_PATH = Path("/home/me/workspace/bspline_ndvi")

    LOGS_PATH = SPLINE_PATH / "logs"

    CHECKPOINTS_PATH = SPLINE_PATH / "checkpoints"
    BEST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "best_model.keras"

    train_bspline_model(
        train_dir=str(DATASET_PATH),
        val_dir=None,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_dir=str(CHECKPOINTS_PATH),
        log_dir=str(LOGS_PATH),
        checkpoint_to_resume=str(BEST_CHECKPOINT_PATH),
    )
