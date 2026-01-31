import os
from shutil import rmtree
from pathlib import Path
import keras
import keras.backend as K
import tensorflow as tf
import argparse

from dataset import DatasetGenerator
from build_model import build_eo_convlstm_model, load_model
from losses import MaskedHuberLoss, kNDVILoss, ImprovedkNDVILoss, EnablekNDVICallback
from config import DATASET_PATH, VALIDATION_PATH


class WarmupScheduler(keras.callbacks.Callback):
    """Learning rate warmup over the first few epochs."""
    def __init__(self, warmup_epochs=5, initial_lr=1e-5, target_lr=3e-4):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(lr)
            print(f'\nWarmup: Setting learning rate to {lr:.2e}')


def adapt_inputs(x, y):
    # Select the last time step for temporal metadata to get (batch, 3)
    # dataset time shape: (batch, input_frames, 3)
    t_meta = x['time'][:, -1, :]
    
    x_new = {
        'sentinel2_sequence': x['sentinel2'],
        'cloudmask_sequence': x['cloudmask'],
        'landcover_map': x['landcover'],
        'temporal_metadata': t_meta,
        'weather_sequence': x['weather']
    }
    return x_new, y


def train_bspline_model(
    train_dir,
    val_dir=None,
    batch_size=1,
    epochs=2,
    learning_rate=3e-4,
    checkpoint_dir='checkpoints',
    log_dir='logs/bspline',
    resume_from=None,
    initial_epoch=0
):
    # Delete old files ONLY if starting fresh
    if resume_from is None:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            rmtree(checkpoint_path)
        
        logs_path = Path(log_dir)
        if logs_path.exists():
            rmtree(logs_path)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading data from {train_dir}...")

    # 1. Dataset
    generator = DatasetGenerator(train_dir)
    train_dataset = generator.get_dataset()

    # Shuffle and Batch (buffer=15 for proper randomization)
    train_dataset = train_dataset.shuffle(15).batch(batch_size).repeat()
    # Apply mapping to match model inputs
    train_dataset = train_dataset.map(adapt_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_generator = None
    if val_dir and os.path.exists(val_dir):
        print(f"Loading validation data from {val_dir}...")
        val_generator = DatasetGenerator(val_dir)
        val_dataset = val_generator.get_dataset()
        val_dataset = val_dataset.batch(batch_size).repeat()
        val_dataset = val_dataset.map(adapt_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # 2. Model
    if resume_from:
        print(f"Resuming training from {resume_from}...")
        model = load_model(resume_from, compile=True)
    else:
        print("Creating EO ConvLSTM model...")
        model = build_eo_convlstm_model()
        
        # 3. Compile with gradient clipping for ConvLSTM stability
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1.0  # Gradient clipping to prevent exploding gradients
        )
        loss = ImprovedkNDVILoss(
            regression_weight=5.0,
            variance_weight=2.0,
            kndvi_clip=0.5,      # Gradient clipping threshold
            sigma=1.0            # RBF kernel sigma
        )
        model.compile(optimizer=optimizer, loss=loss)

    # Display Model Summary
    model.summary()

    # 4. Callbacks
    # Save full model (not just weights) to preserve optimizer state
    checkpoint_path = os.path.join(checkpoint_dir, 'model_at_{epoch}.keras')

    # Get loss function for epoch callback
    loss_fn = model.loss
    
    # Determine which metric to monitor (prefer val_loss when available)
    monitor_metric = 'val_loss' if val_dataset is not None else 'loss'
    
    callbacks = [
        # Warmup LR for first 5 epochs to stabilize training
        WarmupScheduler(
            warmup_epochs=5,
            initial_lr=1e-5,
            target_lr=learning_rate
        ),
        # Enable kNDVI loss after regression has stabilized
        EnablekNDVICallback(
            enable_epoch=20,
            kndvi_weight=1.0
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_metric,
            save_best_only=False,
            save_weights_only=False,  # Save full model including optimizer
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=15,
            min_lr=1e-9,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=35,
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
        # Attempt save on interrupt?
        # return model

    # Save final model (full model with optimizer state)
    final_model_path = os.path.join(checkpoint_dir, 'final_model.keras')
    print(f"Saving final model to {final_model_path}...")
    model.save(final_model_path)
    
    del model
    K.clear_session()
    print("Training complete.")



def configure_gpu():
    """
    Configures GPU memory growth and limits to prevent allocation failures.
    
    Sets memory growth to True for all physical devices and enforces a
    maximum memory limit of 32GB (32 * 1024 MB) per GPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set approximate memory limit (32GB)
                # This prevents TF from taking all memory if multiple processes are running
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=32*1024)]
                )
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            print("GPU Memory Growth Enabled & Limited to 32GB")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    # Configure GPUs before any other TensorFlow operations
    configure_gpu()

    parser = argparse.ArgumentParser(description="Train BSpline NDVI Model")
    
    # Path Arguments
    parser.add_argument('--train_dir', type=str, default=str(DATASET_PATH), help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, default=str(VALIDATION_PATH), help='Path to validation dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/bspline', help='Directory for TensorBoard logs')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to .keras model file to resume training from')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Epoch to start/resume from')

    args = parser.parse_args()

    # Convert paths to absolute if needed, or leave as is.
    
    train_bspline_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        initial_epoch=args.initial_epoch
    )

if __name__ == "__main__":
    main()