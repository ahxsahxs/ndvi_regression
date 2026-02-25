import os
from shutil import rmtree
from pathlib import Path
import keras
import keras.backend as K
import tensorflow as tf
import argparse

from dataset import DatasetGenerator
from build_model import build_eo_convlstm_model, load_model
from losses import DeltaRegressionLoss
from config import DATASET_PATH, VALIDATION_PATH


class WarmupScheduler(keras.callbacks.Callback):
    """Linear learning rate warmup over the first few epochs."""
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
    # Select the last time step for temporal metadata: (batch, input_frames, 3) -> (batch, 3)
    t_meta = x['time'][:, -1, :]

    x_new = {
        'sentinel2_sequence': x['sentinel2'],
        'landcover_map': x['landcover'],
        'temporal_metadata': t_meta,
        'weather_sequence': x['weather'],
        'target_start_doy': x['target_start_doy']
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
    # Delete old logs/checkpoints only when starting fresh
    if resume_from is None:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            rmtree(checkpoint_path)

        logs_path = Path(log_dir)
        if logs_path.exists():
            rmtree(logs_path)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading data from {train_dir}...")

    generator = DatasetGenerator(train_dir)
    train_dataset = generator.get_dataset()
    train_dataset = train_dataset.shuffle(15).batch(batch_size).repeat()
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

    if resume_from:
        print(f"Resuming training from {resume_from}...")
        model = load_model(resume_from, compile=True)
    else:
        print("Creating EO ConvLSTM model...")
        model = build_eo_convlstm_model()

        # clipnorm=5.0: previous value of 1.0 was too aggressive for a 3-layer
        # ConvLSTM — gradients reaching the first encoder layer were cut before
        # they could produce useful weight updates.
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=5.0
        )
        loss = DeltaRegressionLoss(
            regression_weight=5.0,
            edge_weight=0.1,
        )
        model.compile(optimizer=optimizer, loss=loss)

    model.summary()

    checkpoint_path = os.path.join(checkpoint_dir, 'model_at_{epoch}.keras')
    monitor_metric = 'val_loss' if val_dataset is not None else 'loss'

    callbacks = [
        WarmupScheduler(warmup_epochs=5, initial_lr=1e-5, target_lr=learning_rate),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_metric,
            save_best_only=False,
            save_weights_only=False,
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

    steps_per_epoch = max(1, len(generator.files) // batch_size)
    validation_steps = None
    if val_dataset is not None and val_generator:
        validation_steps = max(1, len(val_generator.files) // batch_size)

    print(f"Starting training from epoch {initial_epoch}...")
    print(f"Steps per epoch: {steps_per_epoch}")
    if validation_steps:
        print(f"Validation steps: {validation_steps}")

    try:
        model.fit(
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

    final_model_path = os.path.join(checkpoint_dir, 'final_model.keras')
    print(f"Saving final model to {final_model_path}...")
    model.save(final_model_path)

    del model
    K.clear_session()
    print("Training complete.")


def configure_gpu():
    """Configure GPU memory growth to prevent allocation failures."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=32 * 1024)]
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)


def main():
    configure_gpu()

    parser = argparse.ArgumentParser(description="Train EO ConvLSTM + Fourier model")

    parser.add_argument('--train_dir', type=str, default=str(DATASET_PATH))
    parser.add_argument('--val_dir', type=str, default=str(VALIDATION_PATH))
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/bspline')
    parser.add_argument('--resume_from', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--initial_epoch', type=int, default=0)

    args = parser.parse_args()

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
