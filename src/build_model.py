
import tensorflow as tf
from keras import layers, models, Input, regularizers
import keras


@keras.utils.register_keras_serializable(package="EOModel")
class CloudAwareGatingLayer(layers.Layer):
    """
    Applies cloud-aware gating to zero out contributions from cloudy pixels.
    
    Multiplies input features by (1 - cloudmask), effectively zeroing out
    features in cloudy regions. This prevents learning spurious patterns
    from corrupted cloud pixel values.
    
    Input: [features, cloudmask]
        - features: (Batch, Time, H, W, C) tensor
        - cloudmask: (Batch, Time, H, W, 1) tensor where 1=cloudy, 0=clear
    
    Output: features * (1 - cloudmask), broadcasted across channels
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        features, cloudmask = inputs
        gate = 1.0 - cloudmask
        return features * gate


@keras.utils.register_keras_serializable(package="EOModel")
class LastTimestepLayer(layers.Layer):
    """
    Extracts the last timestep from a sequence tensor.
    
    Serializable alternative to Lambda(lambda t: t[:, -1, :, :, :]).
    
    Input: (Batch, Time, H, W, C)
    Output: (Batch, H, W, C)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, -1, :, :, :]


@keras.utils.register_keras_serializable(package="EOModel")
class RegressionParameterHead(layers.Layer):
    """
    Generates triplet regression parameters (A, λ, B) from latent embedding.
    
    Takes a latent space embedding and outputs three parameter maps:
    - A (amplitude): Linear activation, can be negative for decreasing trends
    - λ (rate): Softplus activation to ensure λ > 0
    - B (offset): Linear activation for baseline offset
    
    :param n_bands: Number of output bands (default 4 for RGBNIR).
    :type n_bands: int
    :param hidden_units: Hidden units in intermediate conv layers.
    :type hidden_units: int
    """
    def __init__(self, n_bands=4, hidden_units=64, noise_stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.n_bands = n_bands
        self.hidden_units = hidden_units
        self.noise_stddev = noise_stddev  # Diversity noise during training
        
        # Shared feature extraction
        self.conv1 = layers.Conv2D(
            filters=hidden_units,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name='param_conv1'
        )
        self.conv2 = layers.Conv2D(
            filters=hidden_units,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name='param_conv2'
        )
        
        # Separate heads for each parameter
        # A: amplitude (linear, can be positive or negative)
        self.conv_A = layers.Conv2D(
            filters=n_bands,
            kernel_size=(1, 1),
            padding='same',
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name='param_A'
        )
        
        # λ: rate (softplus to ensure > 0, initialized near 1.0)
        self.conv_lambda = layers.Conv2D(
            filters=n_bands,
            kernel_size=(1, 1),
            padding='same',
            activation='softplus',
            kernel_regularizer=regularizers.l2(1e-5),
            name='param_lambda'
        )
        
        # B: offset (linear)
        self.conv_B = layers.Conv2D(
            filters=n_bands,
            kernel_size=(1, 1),
            padding='same',
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name='param_B'
        )
        
        # --- Mode Collapse Safeguards ---
        # Learnable scale/bias to prevent trivial zero outputs
        self.A_scale = self.add_weight(
            name='A_scale', shape=(1, 1, 1, n_bands),
            initializer=keras.initializers.Constant(0.2),
            trainable=True
        )
        self.lambda_min = self.add_weight(
            name='lambda_min', shape=(1, 1, 1, n_bands),
            initializer=keras.initializers.Constant(0.5),
            trainable=True
        )
        self.lambda_scale = self.add_weight(
            name='lambda_scale', shape=(1, 1, 1, n_bands),
            initializer=keras.initializers.Constant(2.0),
            trainable=True
        )
    
    def call(self, latent, training=None):
        """
        Args:
            latent: (Batch, H, W, C) latent embedding
            training: Whether in training mode (for noise injection)
        
        Returns:
            Tuple of (A, lambda_param, B), each with shape (Batch, H, W, n_bands)
        """
        x = self.conv1(latent)
        x = self.conv2(x)
        
        # --- Amplitude A ---
        # Use tanh to bound A, then scale by learnable factor
        # This prevents A from going to zero (trivial solution)
        A_raw = self.conv_A(x)
        A = tf.tanh(A_raw) * tf.abs(self.A_scale)  # Bounded, non-zero scale
        
        # Add diversity noise during training to prevent all pixels having same A
        if training:
            A = A + tf.random.normal(tf.shape(A), stddev=self.noise_stddev)
        
        # --- Growth Rate λ ---
        # Constrain to [lambda_min, lambda_min + lambda_scale]
        # This prevents λ→0 (no growth) or λ→∞ (instant saturation)
        lambda_raw = self.conv_lambda(x)  # Already softplus, so > 0
        lambda_min_abs = tf.abs(self.lambda_min) + 0.1  # Ensure min > 0
        lambda_scale_abs = tf.abs(self.lambda_scale) + 0.1
        lambda_param = lambda_min_abs + tf.sigmoid(lambda_raw) * lambda_scale_abs
        
        # --- Offset B ---
        # Small offset, bounded to prevent B from absorbing all variation
        B_raw = self.conv_B(x)
        B = tf.tanh(B_raw) * 0.1  # Bounded to [-0.1, 0.1]
        
        return A, lambda_param, B
    
    def build(self, input_shape):
        # Build sublayers with the correct input shape
        # input_shape can be list or tuple, normalize to tuple
        input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        self.conv1.build(input_shape)
        conv1_output_shape = input_shape[:-1] + (self.hidden_units,)
        self.conv2.build(conv1_output_shape)
        self.conv_A.build(conv1_output_shape)
        self.conv_lambda.build(conv1_output_shape)
        self.conv_B.build(conv1_output_shape)
        super().build(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_bands": self.n_bands,
            "hidden_units": self.hidden_units,
            "noise_stddev": self.noise_stddev,
        })
        return config


@keras.utils.register_keras_serializable(package="EOModel")
class WeatherTimeAdjustmentMLP(layers.Layer):
    """
    MLP that produces time-varying adjustment factors from weather and temporal inputs.
    
    Takes temporal metadata and weather sequence, processes through MLP, and
    outputs per-timestep adjustment multipliers for the growth rate.
    
    :param output_steps: Number of output timesteps (default 20).
    :type output_steps: int
    :param hidden_units: Hidden units in MLP layers.
    :type hidden_units: int
    """
    def __init__(self, output_steps=20, hidden_units=64, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = output_steps
        self.hidden_units = hidden_units
        
        # Temporal metadata encoder
        self.temporal_dense = layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name='temporal_encode'
        )
        
        # Weather sequence encoder (process each timestep)
        self.weather_dense1 = layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name='weather_encode1'
        )
        self.weather_dense2 = layers.Dense(
            hidden_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name='weather_encode2'
        )
        
        # Fusion and output
        self.fusion_dense = layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name='fusion'
        )
        
        # Output: adjustment per timestep (multiplicative factor around 1.0)
        self.output_dense = layers.Dense(
            output_steps,
            activation='sigmoid',  # Output in [0, 1], will be scaled to [0.5, 1.5]
            kernel_regularizer=regularizers.l2(1e-5),
            name='adjustment_output'
        )
    
    def call(self, inputs):
        """
        Args:
            inputs: List of [temporal_metadata, weather_sequence]
                - temporal_metadata: (Batch, 3) - Year, Cos_DOY, Sin_DOY
                - weather_sequence: (Batch, output_steps, 21) - Weather per step
        
        Returns:
            adjustment: (Batch, output_steps) - Multiplicative adjustment factors [0.5, 1.5]
        """
        temporal, weather = inputs
        
        # Encode temporal (Batch, hidden_units)
        temporal_enc = self.temporal_dense(temporal)
        
        # Encode weather: (Batch, output_steps, 21) -> (Batch, output_steps, hidden/2)
        weather_enc = self.weather_dense1(weather)
        weather_enc = self.weather_dense2(weather_enc)
        
        # Aggregate weather across time: (Batch, hidden/2)
        weather_agg = tf.reduce_mean(weather_enc, axis=1)
        
        # Fuse temporal and weather
        fused = tf.concat([temporal_enc, weather_agg], axis=-1)
        fused = self.fusion_dense(fused)
        
        # Output adjustment factors
        raw_adj = self.output_dense(fused)  # (Batch, output_steps) in [0, 1]
        
        # Scale to [0.5, 1.5] to allow both slowdown and speedup
        adjustment = raw_adj + 0.5
        
        return adjustment
    
    def build(self, input_shape):
        # input_shape is a list: [temporal_shape, weather_shape]
        temporal_shape = tuple(input_shape[0]) if isinstance(input_shape[0], list) else input_shape[0]
        weather_shape = tuple(input_shape[1]) if isinstance(input_shape[1], list) else input_shape[1]
        self.temporal_dense.build(temporal_shape)
        self.weather_dense1.build(weather_shape)
        weather_inter_shape = weather_shape[:-1] + (self.hidden_units,)
        self.weather_dense2.build(weather_inter_shape)
        # Fusion input: hidden_units + hidden_units//2
        fusion_input_shape = (temporal_shape[0], self.hidden_units + self.hidden_units // 2)
        self.fusion_dense.build(fusion_input_shape)
        fusion_output_shape = (temporal_shape[0], self.hidden_units)
        self.output_dense.build(fusion_output_shape)
        super().build(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_steps": self.output_steps,
            "hidden_units": self.hidden_units,
        })
        return config


@keras.utils.register_keras_serializable(package="EOModel")
class GrowthCurveLayer(layers.Layer):
    """
    Computes temporal delta sequence from growth curve parameters with adjustment.
    
    Given parameters A, λ, B per pixel per band, and time adjustment factors,
    computes: δ(t) = A · (1 - exp(-λ · t · adj(t))) + B
    
    :param output_steps: Number of timesteps to generate (default 20).
    :type output_steps: int
    """
    def __init__(self, output_steps=20, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = output_steps
    
    def call(self, inputs):
        """
        Args:
            inputs: Tuple of (A, lambda_param, B, adjustment)
                - A: (Batch, H, W, Bands) - Amplitude
                - lambda_param: (Batch, H, W, Bands) - Growth rate (> 0)
                - B: (Batch, H, W, Bands) - Offset
                - adjustment: (Batch, output_steps) - Time adjustment factors
        
        Returns:
            deltas: (Batch, output_steps, H, W, Bands)
        """
        A, lambda_param, B, adjustment = inputs
        
        # Create time vector normalized to [0, 1]
        t = tf.linspace(0.0, 1.0, self.output_steps)  # (output_steps,)
        
        # Reshape for broadcasting: (1, output_steps, 1, 1, 1)
        t = tf.reshape(t, [1, self.output_steps, 1, 1, 1])
        
        # Reshape adjustment: (Batch, output_steps) -> (Batch, output_steps, 1, 1, 1)
        adj = tf.reshape(adjustment, [-1, self.output_steps, 1, 1, 1])
        
        # Expand parameters: (Batch, H, W, Bands) -> (Batch, 1, H, W, Bands)
        A = tf.expand_dims(A, axis=1)
        lambda_param = tf.expand_dims(lambda_param, axis=1)
        B = tf.expand_dims(B, axis=1)
        
        # Growth curve formula with adjustment
        # δ(t) = A · (1 - exp(-λ · output_steps · t · adj)) + B
        effective_time = self.output_steps * t * adj
        deltas = A * (1.0 - tf.exp(-lambda_param * effective_time)) + B
        
        return deltas
    
    def get_config(self):
        config = super().get_config()
        config.update({"output_steps": self.output_steps})
        return config


@keras.utils.register_keras_serializable(package="EOModel")
class SpatialSmoothingLayer(layers.Layer):
    """
    Applies learnable spatial smoothing to prevent sharp discontinuities.
    
    Uses depthwise separable convolution with small kernel for efficiency.
    Smoothing is applied per-timestep independently.
    
    :param kernel_size: Size of smoothing kernel (default 3).
    :type kernel_size: int
    """
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        
        # Depthwise convolution for per-channel smoothing
        self.smooth_conv = layers.DepthwiseConv2D(
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            use_bias=False,
            depthwise_initializer='ones',  # Start as identity-like
            depthwise_regularizer=regularizers.l2(1e-5),
            name='smooth_depthwise'
        )
        
        # Residual weight: blend between smoothed and original
        self.alpha = self.add_weight(
            name='smooth_alpha',
            shape=(1,),
            initializer=keras.initializers.Constant(0.1),
            trainable=True
        )
    
    def call(self, inputs):
        """
        Args:
            inputs: (Batch, Time, H, W, Bands)
        
        Returns:
            smoothed: (Batch, Time, H, W, Bands)
        """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        h = tf.shape(inputs)[2]
        w = tf.shape(inputs)[3]
        bands = inputs.shape[-1] or tf.shape(inputs)[-1]
        
        # Reshape to (Batch * Time, H, W, Bands) for Conv2D
        x = tf.reshape(inputs, [-1, h, w, bands])
        
        # Apply smoothing
        smoothed = self.smooth_conv(x)
        
        # Normalize by kernel area (approximate averaging)
        kernel_area = float(self.kernel_size * self.kernel_size)
        smoothed = smoothed / kernel_area
        
        # Residual blend: output = (1-α) * original + α * smoothed
        alpha_clipped = tf.clip_by_value(self.alpha, 0.0, 1.0)
        output = (1.0 - alpha_clipped) * x + alpha_clipped * smoothed
        
        # Reshape back to (Batch, Time, H, W, Bands)
        output = tf.reshape(output, [batch_size, time_steps, h, w, bands])
        
        return output
    
    def build(self, input_shape):
        # input_shape: (Batch, Time, H, W, Bands)
        # Build depthwise conv for the spatial dimensions
        input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        bands = input_shape[-1]
        spatial_shape = (input_shape[0], input_shape[2], input_shape[3], bands)
        self.smooth_conv.build(spatial_shape)
        super().build(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def build_eo_convlstm_model(
        input_shape=(10, 128, 128, 4), 
        cloudmask_shape=(10, 128, 128, 1),
        landcover_shape=(128, 128, 10),
        temporal_shape=(3,),
        weather_shape=(20, 21),
        output_shape=(20, 128, 128, 4),
    ):
    """
    Constructs a Growth Curve Regression model for Satellite Imagery Forecasting.

    Architecture:
    1. Apply cloudmask once to gate inputs
    2. Encode reflectancy + landcover through stacked ConvLSTM to latent space
    3. Predict regression parameters (A, λ, B) from latent embedding
    4. Generate weather/time adjustment factors via MLP
    5. Compute growth curves with adjustment
    6. Apply spatial smoothing

    :param input_shape: Shape of input image sequence (Time, H, W, C).
    :type input_shape: tuple
    :param cloudmask_shape: Shape of cloud mask sequence (Time, H, W, 1).
    :type cloudmask_shape: tuple
    :param landcover_shape: Shape of landcover map (H, W, Classes).
    :type landcover_shape: tuple
    :param temporal_shape: Shape of temporal metadata (Features,).
    :type temporal_shape: tuple
    :param weather_shape: Shape of weather sequence (Output_Steps, Features).
    :type weather_shape: tuple
    :param output_shape: Shape of output sequence (Time, H, W, C).
    :type output_shape: tuple
    :return: A Keras Model instance.
    :rtype: keras.models.Model
    """
    
    # --- Inputs ---
    img_input = Input(shape=input_shape, name='sentinel2_sequence')
    cloudmask_input = Input(shape=cloudmask_shape, name='cloudmask_sequence')
    landcover_input = Input(shape=landcover_shape, name='landcover_map')
    temporal_input = Input(shape=temporal_shape, name='temporal_metadata')
    weather_input = Input(shape=weather_shape, name='weather_sequence')
    
    # --- Prepare Landcover: Tile to time dimension ---
    # (Batch, H, W, 10) -> (Batch, 1, H, W, 10) -> (Batch, Time, H, W, 10)
    target_shape = (1,) + landcover_shape
    landcover_reshaped = layers.Reshape(target_shape, name='lc_reshape')(landcover_input)
    time_steps = input_shape[0]
    landcover_tiled = layers.UpSampling3D(size=(time_steps, 1, 1), name='lc_tile')(landcover_reshaped)

    # --- Concatenate inputs and apply cloud masking ONCE ---
    x = layers.Concatenate(axis=-1, name='input_concat')([img_input, landcover_tiled])
    x = CloudAwareGatingLayer(name='cloud_gating')([x, cloudmask_input])

    # --- Latent Space Encoder (ConvLSTM stack) ---
    # Extract spatiotemporal features, output is latent embedding per pixel
    enc1 = layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding='same',
        dropout=0.15, recurrent_dropout=0.1,
        return_sequences=True, name='encoder_lstm1'
    )(x)
    enc1 = layers.GroupNormalization(groups=4, axis=-1, name='encoder_norm1')(enc1)
    
    enc2 = layers.ConvLSTM2D(
        filters=48, kernel_size=(3, 3), padding='same',
        dropout=0.15, recurrent_dropout=0.1,
        return_sequences=True, name='encoder_lstm2'
    )(enc1)
    enc2 = layers.GroupNormalization(groups=4, axis=-1, name='encoder_norm2')(enc2)
    
    # Final encoder: reduce to single hidden state (latent embedding)
    latent = layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3), padding='same',
        dropout=0.15, recurrent_dropout=0.1,
        return_sequences=False, name='encoder_lstm3'
    )(enc2)
    latent = layers.GroupNormalization(groups=4, axis=-1, name='latent_norm')(latent)
    # Shape: (Batch, H, W, 64)
    
    # --- Skip Connection: Extract last timestep from intermediate encoder ---
    skip = LastTimestepLayer(name='skip_last')(enc2)  # (Batch, H, W, 48)
    
    # --- Fuse latent with skip for richer features ---
    latent_fused = layers.Concatenate(axis=-1, name='latent_skip_concat')([latent, skip])
    # Shape: (Batch, H, W, 64 + 48) = (Batch, H, W, 112)
    
    # --- Regression Parameter Head ---
    n_bands = output_shape[-1]
    output_steps = output_shape[0]
    
    A, lambda_param, B = RegressionParameterHead(
        n_bands=n_bands,
        hidden_units=64,
        name='regression_head'
    )(latent_fused)
    # Each: (Batch, H, W, n_bands)
    
    # --- Weather/Time Adjustment MLP ---
    adjustment = WeatherTimeAdjustmentMLP(
        output_steps=output_steps,
        hidden_units=64,
        name='weather_time_mlp'
    )([temporal_input, weather_input])
    # Shape: (Batch, output_steps)
    
    # --- Growth Curve Layer ---
    deltas = GrowthCurveLayer(
        output_steps=output_steps,
        name='growth_curve'
    )([A, lambda_param, B, adjustment])
    # Shape: (Batch, output_steps, H, W, n_bands)
    
    # --- Spatial Smoothing ---
    outputs = SpatialSmoothingLayer(
        kernel_size=3,
        name='spatial_smoothing'
    )(deltas)
    # Shape: (Batch, output_steps, H, W, n_bands)

    model = models.Model(
        inputs=[
            img_input,
            cloudmask_input,
            landcover_input,
            temporal_input,
            weather_input,
        ], 
        outputs=outputs
    )
    return model


# Example instantiation
# model = build_eo_convlstm_model()


def load_model(model_path: str, compile: bool = True) -> keras.Model:
    """
    Load a saved Keras model with all custom layers registered.
    
    This function handles loading models saved with custom layers defined in
    this module, automatically providing the necessary custom_objects mapping.
    
    :param model_path: Path to the .keras model file.
    :type model_path: str
    :param compile: Whether to compile the model after loading (default True).
    :type compile: bool
    :return: Loaded Keras model.
    :rtype: keras.Model
    :raises FileNotFoundError: If model_path does not exist.
    :raises Exception: If loading fails for any reason.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # All custom layers are registered via @keras.utils.register_keras_serializable
    # so they should be found automatically. We provide them explicitly for safety.
    custom_objects = {
        'CloudAwareGatingLayer': CloudAwareGatingLayer,
        'LastTimestepLayer': LastTimestepLayer,
        'RegressionParameterHead': RegressionParameterHead,
        'WeatherTimeAdjustmentMLP': WeatherTimeAdjustmentMLP,
        'GrowthCurveLayer': GrowthCurveLayer,
        'SpatialSmoothingLayer': SpatialSmoothingLayer,
    }
    
    model = keras.models.load_model(
        model_path, 
        custom_objects=custom_objects, 
        compile=compile
    )
    
    return model