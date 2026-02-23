
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
class FourierCoefficientHead(layers.Layer):
    """
    Predicts Fourier harmonic coefficients (a_k, b_k) from latent embedding.
    
    Replaces the old RegressionParameterHead (A, λ, B) with a set of
    cosine/sine amplitude coefficients per pixel per band, enabling
    non-monotonic temporal dynamics.
    
    Output per pixel per band: [a_1, b_1, a_2, b_2, ..., a_K, b_K]
    
    :param n_bands: Number of output spectral bands (default 4).
    :type n_bands: int
    :param n_harmonics: Number of Fourier harmonics K (default 3).
    :type n_harmonics: int
    :param hidden_units: Hidden units in intermediate conv layers.
    :type hidden_units: int
    :param noise_stddev: Training noise for diversity (default 0.02).
    :type noise_stddev: float
    """
    def __init__(self, n_bands=4, n_harmonics=3, hidden_units=64,
                 noise_stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.n_bands = n_bands
        self.n_harmonics = n_harmonics
        self.n_coeffs = 2 * n_harmonics  # a_k and b_k per harmonic
        self.hidden_units = hidden_units
        self.noise_stddev = noise_stddev
        
        # Shared feature extraction
        self.conv1 = layers.Conv2D(
            filters=hidden_units,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            name='fourier_conv1'
        )
        self.conv2 = layers.Conv2D(
            filters=hidden_units,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            name='fourier_conv2'
        )
        
        # Coefficient head: outputs all coefficients for all bands at once
        # Shape: (B, H, W, n_bands * 2K)
        self.conv_coeffs = layers.Conv2D(
            filters=n_bands * self.n_coeffs,
            kernel_size=(1, 1),
            padding='same',
            activation=None,
            kernel_regularizer=regularizers.l2(1e-4),
            name='fourier_coeffs'
        )
        
        # Learnable amplitude scale per band per coefficient
        self.coeff_scale = self.add_weight(
            name='coeff_scale', shape=(1, 1, 1, n_bands, self.n_coeffs),
            initializer=keras.initializers.Constant(1.0),
            trainable=True
        )
    
    def call(self, latent, training=None):
        """
        Args:
            latent: (Batch, H, W, C) latent embedding from encoder
            training: Whether in training mode (for noise injection)
        
        Returns:
            coefficients: (Batch, H, W, n_bands, 2K) Fourier coefficients
        """
        x = self.conv1(latent)
        x = self.conv2(x)
        
        # Raw coefficient prediction: (B, H, W, n_bands * 2K)
        raw = self.conv_coeffs(x)
        
        # Reshape to (B, H, W, n_bands, 2K)
        shape = tf.shape(raw)
        coeffs = tf.reshape(raw, [shape[0], shape[1], shape[2],
                                  self.n_bands, self.n_coeffs])
        
        # Scale by learnable amplitude — no activation bounding.
        # Amplitude is controlled by L2 regularization on conv layers
        # instead of softsign/tanh, allowing the model to learn larger
        # coefficients for fast vegetation changes.
        coeffs = coeffs * self.coeff_scale
        
        # Add diversity noise during training
        if training:
            coeffs = coeffs + tf.random.normal(
                tf.shape(coeffs), stddev=self.noise_stddev)
        
        return coeffs
    
    def build(self, input_shape):
        input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        self.conv1.build(input_shape)
        conv1_out = input_shape[:-1] + (self.hidden_units,)
        self.conv2.build(conv1_out)
        self.conv_coeffs.build(conv1_out)
        super().build(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_bands": self.n_bands,
            "n_harmonics": self.n_harmonics,
            "hidden_units": self.hidden_units,
            "noise_stddev": self.noise_stddev,
        })
        return config


@keras.utils.register_keras_serializable(package="EOModel")
class FourierSynthesisLayer(layers.Layer):
    """
    Synthesises temporal delta sequence from Fourier coefficients via matrix multiply.
    
    Precomputes the Fourier design matrix Φ of shape (T, 2K):
        Φ[t, 2k]   = cos((k+1) · ω · t)
        Φ[t, 2k+1] = sin((k+1) · ω · t)
    where ω = 2π / cycle_length.
    
    :param output_steps: Number of forecast timesteps T (default 20).
    :type output_steps: int
    :param n_harmonics: Number of Fourier harmonics K (default 3).
    :type n_harmonics: int
    :param cycle_length: Length of annual cycle in time steps (default 73).
    :type cycle_length: int
    """
    def __init__(self, output_steps=20, n_harmonics=3, cycle_length=73, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = output_steps
        self.n_harmonics = n_harmonics
        self.n_coeffs = 2 * n_harmonics
        self.cycle_length = cycle_length
        
        # Build the constant Fourier design matrix Φ (T, 2K)
        self._design_matrix = self._build_design_matrix()
    
    def _build_design_matrix(self):
        """Build constant Fourier basis matrix Φ of shape (T, 2K)."""
        import numpy as np
        T = self.output_steps
        K = self.n_harmonics
        # Use annual cycle length for frequency base
        omega = 2.0 * np.pi / self.cycle_length
        
        # Time indices normalised: 1, 2, ..., T
        t = np.arange(1, T + 1, dtype=np.float32)  # (T,)
        
        phi = np.zeros((T, 2 * K), dtype=np.float32)
        for k in range(K):
            freq = (k + 1) * omega
            phi[:, 2 * k]     = np.cos(freq * t)
            phi[:, 2 * k + 1] = np.sin(freq * t)
        
        return tf.constant(phi, dtype=tf.float32)  # (T, 2K)
    
    def call(self, coeffs):
        """
        Args:
            coeffs: (B, H, W, n_bands, 2K) base Fourier coefficients
        
        Returns:
            deltas: (B, T, H, W, n_bands)
        """
        # coeffs: (B, H, W, bands, 2K) -> (B, 1, H, W, bands, 2K)
        coeffs = tf.expand_dims(coeffs, axis=1)
        
        # Fourier design matrix: (T, 2K) -> (1, T, 1, 1, 1, 2K)
        phi = tf.reshape(self._design_matrix,
                         [1, self.output_steps, 1, 1, 1, self.n_coeffs])
        
        # Synthesis via element-wise multiply + sum over coefficients
        # δ(b,t,h,w,band) = Σ_j coeffs(b,t,h,w,band,j) · Φ(t,j)
        deltas = tf.reduce_sum(coeffs * phi, axis=-1)
        # Shape: (B, T, H, W, n_bands)
        
        return deltas
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_steps": self.output_steps,
            "n_harmonics": self.n_harmonics,
            "cycle_length": self.cycle_length,
        })
        return config


def build_eo_convlstm_model(
        input_shape=(10, 128, 128, 4), 
        cloudmask_shape=(10, 128, 128, 1),
        landcover_shape=(128, 128, 10),
        temporal_shape=(3,),
        weather_shape=(20, 21),
        output_shape=(20, 128, 128, 4),
        n_harmonics=6,
        cycle_length=73,
    ):
    """
    Constructs a Fourier Harmonics Regression model for Satellite Imagery Forecasting.

    Architecture:
    1. Apply cloudmask once to gate inputs
    2. Encode reflectancy + landcover through stacked ConvLSTM to latent space
    3. Predict Fourier coefficients (a_k, b_k) from latent embedding
    4. Generate per-step FiLM modulation (γ, β) from weather + DOY
    5. Synthesise temporal deltas via Fourier basis matrix multiply

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
    :param n_harmonics: Number of Fourier harmonics K (default 3).
    :type n_harmonics: int
    :param cycle_length: Length of annual cycle in time steps (default 73).
    :type cycle_length: int
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
    enc1 = layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding='same',
        dropout=0.3, recurrent_dropout=0.2,
        return_sequences=True, name='encoder_lstm1'
    )(x)
    enc1 = layers.GroupNormalization(groups=4, axis=-1, name='encoder_norm1')(enc1)
    
    enc2 = layers.ConvLSTM2D(
        filters=48, kernel_size=(3, 3), padding='same',
        dropout=0.3, recurrent_dropout=0.2,
        return_sequences=True, name='encoder_lstm2'
    )(enc1)
    enc2 = layers.GroupNormalization(groups=4, axis=-1, name='encoder_norm2')(enc2)
    
    # Final encoder: reduce to single hidden state (latent embedding)
    latent = layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3), padding='same',
        dropout=0.3, recurrent_dropout=0.2,
        return_sequences=False, name='encoder_lstm3'
    )(enc2)
    latent = layers.GroupNormalization(groups=4, axis=-1, name='latent_norm')(latent)
    # Shape: (Batch, H, W, 64)
    
    # --- Skip Connection: Extract last timestep from intermediate encoder ---
    skip = LastTimestepLayer(name='skip_last')(enc2)  # (Batch, H, W, 48)
    
    # --- Context encoding (Weather + Temporal) ---
    weather_flat = layers.Flatten(name='weather_flatten')(weather_input)
    weather_enc = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='weather_enc')(weather_flat)
    temporal_enc = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='temporal_enc')(temporal_input)
    context = layers.Concatenate(name='context_concat')([weather_enc, temporal_enc])
    
    # Broadcast context to spatial dimensions (Batch, 1, 1, 48) -> (Batch, H, W, 48)
    context_reshaped = layers.Reshape((1, 1, 48), name='context_reshape')(context)
    context_tiled = layers.UpSampling2D(size=(input_shape[1], input_shape[2]), name='context_tile')(context_reshaped)
    
    # --- Fuse latent with skip and context for richer features ---
    latent_fused = layers.Concatenate(axis=-1, name='latent_skip_context_concat')([latent, skip, context_tiled])
    
    # --- Fourier Coefficient Head ---
    n_bands = output_shape[-1]
    output_steps = output_shape[0]
    
    coefficients = FourierCoefficientHead(
        n_bands=n_bands,
        n_harmonics=n_harmonics,
        hidden_units=64,
        name='fourier_head'
    )(latent_fused)
    # Shape: (Batch, H, W, n_bands, 2K)
    
    # --- Fourier Synthesis ---
    outputs_raw = FourierSynthesisLayer(
        output_steps=output_steps,
        n_harmonics=n_harmonics,
        cycle_length=cycle_length,
        name='fourier_synthesis'
    )(coefficients)
    # Shape: (Batch, output_steps, H, W, n_bands)
    
    # No final activation — coefficients are scaled by a learnable factor
    # inside FourierCoefficientHead, with amplitude controlled by L2
    # regularization on conv layers. Previous softsign bounding caused
    # systematic under-prediction of large deltas.
    outputs = outputs_raw

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
        'FourierCoefficientHead': FourierCoefficientHead,
        'FourierSynthesisLayer': FourierSynthesisLayer,
    }
    
    model = keras.models.load_model(
        model_path, 
        custom_objects=custom_objects, 
        compile=compile
    )
    
    return model