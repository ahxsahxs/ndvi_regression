import keras
from keras import layers
import tensorflow as tf

@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class CubicSplineLayer(layers.Layer):
    """
    Layer for cubic polynomial splines.
    """

    def __init__(self, num_future_frames=20, name='cubic_spline_layer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_future_frames = num_future_frames

    def call(self, spline_params):
        # Create t_matrix inside call to avoid scope issues
        t = tf.range(1, self.num_future_frames + 1, dtype=tf.float32)
        t_norm = t / self.num_future_frames

        t_matrix = tf.stack([
            tf.ones_like(t_norm),
            t_norm,
            t_norm ** 2,
            t_norm ** 3
        ], axis=-1)

        batch = tf.shape(spline_params)[0]
        h = tf.shape(spline_params)[1]
        w = tf.shape(spline_params)[2]
        channels = tf.shape(spline_params)[3]

        params_flat = tf.reshape(spline_params, [-1, 4])
        preds_flat = tf.matmul(params_flat, t_matrix, transpose_b=True)
        preds = tf.reshape(preds_flat, [batch, h, w, channels, self.num_future_frames])
        preds = tf.transpose(preds, [0, 4, 1, 2, 3])

        return preds

    def get_config(self):
        config = super().get_config()
        config.update({"num_future_frames": self.num_future_frames})
        return config
    

@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class ConvLSTMSplineModel(keras.Model):
    def __init__(
        self,
        input_frames=10,
        output_frames=20,
        img_size=128,
        inp_channels=4,
        out_channels=4,
        conv_filters=[30, 30, 30],
        dropout_rate=0.3,
        **kwargs
    ):
        """
        ConvLSTM model with attention mechanisms and B-spline predictions.

        Args:
            input_frames: Number of input temporal frames
            output_frames: Number of output temporal frames to predict
            img_size: Spatial dimension of images
            inp_channels: Number of spectral bands in the input data
            out_channels: Number of spectral bands in the target data
            n_control_points: Number of B-spline control points
            conv_filters: List of filter sizes for ConvLSTM layers
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(**kwargs)

        # Store configuration
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.img_size = img_size
        self.out_channels = out_channels
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate

        # ConvLSTM layers
        self.convlstm_layers = []
        self.batch_norms_lstm = []

        for i, filters in enumerate(conv_filters):
            return_sequences = (i < len(conv_filters) - 1)

            self.convlstm_layers.append(
                layers.ConvLSTM2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    padding='same',
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'convlstm_{i+1}'
                )
            )
            self.batch_norms_lstm.append(layers.BatchNormalization(name=f"convlstm_{i+1}_batchnorm"))

        # Additional spatial processing
        self.conv2d_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2d_1")
        self.conv2d_1_batchnorm = layers.BatchNormalization(name="conv2d_1_batchnorm")
        self.conv2d_1_dropout = layers.Dropout(dropout_rate, name=f"conv2d_1_dropout")

        self.conv2d_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2d_2")
        self.conv2d_2_batchnorm = layers.BatchNormalization(name="conv2d_2_batchnorm")

        # Spline parameters
        self.n_params = 4
        self.spline_layer = CubicSplineLayer(
            num_future_frames=output_frames,
            name="spline_layer"
        )

        # Parameter prediction layer
        self.conv2d_spline_params = layers.Conv2D(
            out_channels * self.n_params,
            (1, 1),
            padding='same',
            name='conv2d_spline_params'
        )

        # Reshape layer
        self.reshape_params = layers.Reshape(
            (img_size, img_size, out_channels, self.n_params),
            name='reshape_params'
        )

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Tensors for
             - Time (batch, frames, 3)
             - Sentinel 2 (batch, frames, height, width, channels)
             - Cloudmask (batch, frames, height, width, 1)
             - Landcover (batch, height, width, 10)
            training: Boolean indicating training mode

        Returns:
            predictions: Tensor of shape (batch, output_frames, height, width, out_channels)
        """
        if isinstance(inputs, dict):
            time = inputs['time']
            sentinel2 = inputs['sentinel2']
            cloudmask = inputs['cloudmask']
            landcover = inputs['landcover']
        else:
            time, sentinel2, cloudmask, landcover = inputs

        # Get shapes
        batch_size = tf.shape(sentinel2)[0]
        frames = tf.shape(sentinel2)[1]
        height = tf.shape(sentinel2)[2]
        width = tf.shape(sentinel2)[3]

        # 1. Tile Time: (batch, frames, 3) -> (batch, frames, H, W, 3)
        time_tiled = tf.reshape(time, [batch_size, frames, 1, 1, 3])
        time_tiled = tf.tile(time_tiled, [1, 1, height, width, 1])

        # 2. Tile Landcover: (batch, H, W, 10) -> (batch, frames, H, W, 10)
        landcover_tiled = tf.expand_dims(landcover, axis=1) # (batch, 1, H, W, 10)
        landcover_tiled = tf.tile(landcover_tiled, [1, frames, 1, 1, 1])

        x = tf.concat([
            sentinel2,
            cloudmask,
            time_tiled,
            landcover_tiled
        ], axis=-1)

        print("X Shape: ", tf.shape(x)) 
        

        # ConvLSTM layers for spatiotemporal feature extraction
        for convlstm, bn in zip(self.convlstm_layers, self.batch_norms_lstm):
            x = convlstm(x, training=training)
            x = bn(x, training=training)

        # Additional spatial processing
        x = self.conv2d_1(x, training=training)
        x = self.conv2d_1_batchnorm(x, training=training)
        x = self.conv2d_1_dropout(x, training=training)

        x = self.conv2d_2(x, training=training)
        x = self.conv2d_2_batchnorm(x, training=training)

        # Predict spline parameters
        spline_params = self.conv2d_spline_params(x, training=training)
        spline_params = self.reshape_params(spline_params)

        # Generate predictions through spline interpolation
        predictions = self.spline_layer(spline_params, training=training)

        return {"deltas": predictions}

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            
            # Extract the specific target and prediction for the main loss
            # y contains: sentinel2, cloudmask, deltas
            # y_pred contains: deltas
            
            # Add dummy cloudmask to y_pred to satisfy SplineLoss requirements
            y_pred_loss = {"deltas": y_pred["deltas"], "cloudmask": y["cloudmask"]}
            
            # Compute loss manually to avoid Keras structure checks
            loss = self.loss(y, y_pred_loss)
            
            # Add regularization losses
            if self.losses:
                loss += tf.add_n(self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y["deltas"], y_pred["deltas"], sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Compute predictions
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        y_pred_loss = {"deltas": y_pred["deltas"], "cloudmask": y["cloudmask"]}
        loss = self.loss(y, y_pred_loss)
        
        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y["deltas"], y_pred["deltas"])
        
        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}


    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "img_size": self.img_size,
            "out_channels": self.out_channels,
            "conv_filters": self.conv_filters,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)