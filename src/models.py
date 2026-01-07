import keras
from keras import layers
import tensorflow as tf

@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class ConvLSTMPredictionModel(keras.Model):
    def __init__(
        self,
        input_frames=18,
        output_frames=12,
        img_size=128,
        out_channels=4,
        conv_filters=[64],
        dropout_rate=0.3,
        poly_degree=2,
        **kwargs
    ):
        """
        ConvLSTM model with attention mechanisms and pixel-wise MLP predictions.

        Args:
            input_frames: Number of input temporal frames
            output_frames: Number of output temporal frames to predict
            img_size: Spatial dimension of images
            out_channels: Number of spectral bands in the target data
            conv_filters: List of filter sizes for ConvLSTM layers
            dropout_rate: Dropout rate for regularization
            poly_degree: Degree of the polynomial curve used for future forecasting
        """
        super().__init__(**kwargs)

        # Store configuration
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.img_size = img_size
        self.out_channels = out_channels
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        self.poly_degree = poly_degree

        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # ConvLSTM layers
        self.convlstm_layers = []

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

        # Pixel-wise MLP
        self.mlp = keras.Sequential([
            layers.Conv2D(64, (3, 3), padding='same', activation='tanh', name="mlp_hidden"),
            layers.Conv2D(out_channels * (self.poly_degree + 1), (3, 3), activation='tanh', padding='same', name="mlp_output")
        ], name="pixel_wise_mlp")

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
        
        # ConvLSTM layers for spatiotemporal feature extraction
        for convlstm in self.convlstm_layers:
            x = convlstm(x, training=training)

        # Predict using pixel-wise MLP
        x = self.mlp(x, training=training) # (batch, h, w, output_frames * out_channels)

        # Reshape to (batch, output_frames, h, w, out_channels)
        # x is (batch, h, w, frames*channels)
        curve_params = tf.reshape(x, [batch_size, height, width, self.out_channels, self.poly_degree+1])
        
        # deltas[t] = a*t² + b*t + c
        deltas = self.horner_polynomial(curve_params)

        return {"deltas": deltas, "cloudmask": None, "sentinel2": None}


    def horner_polynomial(self, params:tf.Tensor):
        """
        Evaluate polynomial using vectorized Horner's method.
        
        Args:
            params: Tensor of shape (batch, height, width, n_bands, poly_degree+1)
                    Coefficients ordered from highest to lowest degree
                    [a_n, a_{n-1}, ..., a_1, a_0]
        Returns:
            deltas: Tensor of shape (batch, output_frames, height, width, n_bands)
        """
        # Create time values: [1, 2, 3, ..., output_frames]
        times = tf.range(1, self.output_frames + 1, dtype=tf.float32)
        times = tf.reshape(times, [1, self.output_frames, 1, 1, 1])
        
        # Expand params: (batch, 1, height, width, n_bands, poly_degree+1)
        params_expanded = tf.expand_dims(params, axis=1)
        
        # Split coefficients along last dimension
        # coeffs is a list of (poly_degree+1) tensors, each (batch, 1, H, W, n_bands)
        coeffs = tf.unstack(params_expanded, axis=-1)
        
        # Initialize with highest degree coefficient
        result = coeffs[0]
        
        # Horner's method: result = result * t + next_coeff
        for coeff in coeffs[1:]:
            result = result * times + coeff
        
        return result

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.loss(
                y,
                y_pred,
                sample_weight=sample_weight,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
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
        loss = self.loss(y, y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
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
            "poly_degree": self.poly_degree,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
    