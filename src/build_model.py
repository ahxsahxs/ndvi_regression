import tensorflow as tf
from keras import layers, models, Input
import keras


@keras.utils.register_keras_serializable(package="EOModel")
class CloudAwareGatingLayer(layers.Layer):
    """
    Kept for backward compatibility when loading old checkpoints via load_model().
    Not used in the current model architecture — the encoder now receives the
    BAP-filled sequence directly without gating.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        features, cloudmask = inputs
        return features * (1.0 - cloudmask)


@keras.utils.register_keras_serializable(package="EOModel")
class LearnableScale(layers.Layer):
    """
    Multiplies input by a single learnable scalar, initialised to `init_value`.

    Kept for backward compatibility when loading old checkpoints via load_model().
    Not used in the current model architecture.
    """
    def __init__(self, init_value=0.05, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=(),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.utils.register_keras_serializable(package="EOModel")
class LastTimestepLayer(layers.Layer):
    """
    Extracts the last timestep from a sequence tensor.

    Serializable alternative to Lambda(lambda t: t[:, -1, :, :, :]).

    Input:  (Batch, Time, H, W, C)
    Output: (Batch, H, W, C)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, -1, :, :, :]


@keras.utils.register_keras_serializable(package="EOModel")
class FourierCoefficientHead(layers.Layer):
    """
    Predicts Fourier harmonic coefficients (a_k, b_k) from the latent embedding.

    Output per pixel per band: [a_1, b_1, …, a_K, b_K]
    Shape: (B, H, W, n_bands, 2K)

    No L2 regularisation — AdamW weight_decay handles that.
    No learnable coeff_scale — convolutions output at full scale so harmonics
    are not suppressed relative to the DC offset path.

    DC offset is a separate Conv2D in build_eo_convlstm_model so this
    layer stays single-output (Keras 3 functional API cannot trace multi-tensor
    returns from custom layers).
    """
    def __init__(self, n_bands=4, n_harmonics=3, hidden_units=64,
                 noise_stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.n_bands = n_bands
        self.n_harmonics = n_harmonics
        self.n_coeffs = 2 * n_harmonics
        self.hidden_units = hidden_units
        self.noise_stddev = noise_stddev

        self.conv1 = layers.Conv2D(
            filters=hidden_units, kernel_size=(3, 3), padding='same',
            activation='relu', name='fourier_conv1'
        )
        self.conv2 = layers.Conv2D(
            filters=hidden_units, kernel_size=(3, 3), padding='same',
            activation='relu', name='fourier_conv2'
        )
        # Harmonic coefficients head: (B, H, W, n_bands * 2K)
        self.conv_coeffs = layers.Conv2D(
            filters=n_bands * self.n_coeffs, kernel_size=(1, 1),
            padding='same', activation=None, name='fourier_coeffs'
        )

    def call(self, latent, training=None):
        x = self.conv1(latent)
        x = self.conv2(x)

        raw = self.conv_coeffs(x)  # (B, H, W, n_bands * 2K)

        # Reshape using keras.ops — tf.reshape with tf.shape() breaks Keras 3
        # symbolic graph tracing; keras.ops.reshape preserves tensor connectivity.
        s = keras.ops.shape(raw)
        coeffs = keras.ops.reshape(raw, (s[0], s[1], s[2], self.n_bands, self.n_coeffs))

        if training:
            coeffs = coeffs + tf.random.normal(
                tf.shape(coeffs), stddev=self.noise_stddev, dtype=coeffs.dtype)

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
    Synthesises temporal delta sequence from Fourier coefficients.

    Computes the Fourier design matrix Φ dynamically from actual target DOY:
        Φ[t, 2k]   = cos((k+1) · ω · doy_t)
        Φ[t, 2k+1] = sin((k+1) · ω · doy_t)
    where ω = 2π / cycle_length_days.

    Uses keras.ops throughout _build_design_matrix so that tf.cos/tf.stack on
    KerasTensors don't break Keras 3 symbolic graph tracing.
    """
    def __init__(self, output_steps=20, n_harmonics=3, cycle_length_days=365,
                 step_interval=5, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = output_steps
        self.n_harmonics = n_harmonics
        self.n_coeffs = 2 * n_harmonics
        self.cycle_length_days = cycle_length_days
        self.step_interval = step_interval

    def _build_design_matrix(self, start_doy):
        """
        Build Fourier basis matrix Φ from normalised start DOY.

        Args:
            start_doy: (B, 1) normalised DOY in [0, 1]
        Returns:
            phi: (B, T, 2K)
        """
        T = self.output_steps
        K = self.n_harmonics
        # Compute omega as a plain Python float — avoids tf.constant on KerasTensors
        omega = 2.0 * 3.14159265358979 / float(self.cycle_length_days)

        start_day = start_doy * float(self.cycle_length_days)  # (B, 1)

        # Build step offsets as a Python list and convert once — never apply
        # tf.range / tf.cast directly on start_doy-derived tensors before this.
        step_values = [float((t + 1) * self.step_interval) for t in range(T)]
        step_offsets = keras.ops.convert_to_tensor(step_values, dtype=start_doy.dtype)  # (T,)
        step_offsets = keras.ops.reshape(step_offsets, (1, T))  # (1, T)

        doy_t = start_day + step_offsets  # (B, T) — broadcast

        phi_parts = []
        for k in range(K):
            freq = float(k + 1) * omega
            phi_parts.append(keras.ops.cos(freq * doy_t))   # (B, T)
            phi_parts.append(keras.ops.sin(freq * doy_t))   # (B, T)

        return keras.ops.stack(phi_parts, axis=-1)  # (B, T, 2K)

    def call(self, inputs):
        """
        Args:
            inputs: [coeffs, start_doy]
                coeffs:    (B, H, W, n_bands, 2K)
                start_doy: (B, 1) normalised DOY in [0, 1]
        Returns:
            deltas: (B, T, H, W, n_bands)
        """
        coeffs, start_doy = inputs

        phi = self._build_design_matrix(start_doy)  # (B, T, 2K)

        coeffs_5d = keras.ops.expand_dims(coeffs, axis=1)  # (B, 1, H, W, n_bands, 2K)

        # Expand phi from (B, T, 2K) to (B, T, 1, 1, 1, 2K) using keras.ops.expand_dims
        # Avoids tf.reshape with dynamic batch dimension which breaks Keras 3 tracing.
        phi_6d = keras.ops.expand_dims(
            keras.ops.expand_dims(
                keras.ops.expand_dims(phi, axis=2),
                axis=3),
            axis=4)

        deltas = keras.ops.sum(coeffs_5d * phi_6d, axis=-1)  # (B, T, H, W, n_bands)
        return deltas

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_steps": self.output_steps,
            "n_harmonics": self.n_harmonics,
            "cycle_length_days": self.cycle_length_days,
            "step_interval": self.step_interval,
        })
        return config


def build_eo_convlstm_model(
        input_shape=(10, 128, 128, 4),
        landcover_shape=(128, 128, 10),
        temporal_shape=(10, 3),
        weather_shape=(20, 21),
        target_start_doy_shape=(1,),
        output_shape=(20, 128, 128, 4),
        n_harmonics=4,
        cycle_length_days=365,
        step_interval=5,
):
    """
    Constructs the Fourier Harmonics model for satellite reflectance forecasting.

    Architecture:
    1. Encode BAP-filled Sentinel-2 + per-frame DOY through stacked ConvLSTM.
    2. Fuse latent state with global weather/temporal context + landcover.
    3. Predict Fourier harmonic coefficients per pixel per band.
    4. Synthesise temporal deltas via DOY-conditioned Fourier basis.
    5. Apply weather FiLM modulation to harmonics.
    6. Add DC offset (sustained trend) — separate Conv2D broadcast across time.

    Changes from previous architecture:
    - Landcover moved from encoder to decoder (latent_fused concat) — avoids
      wasting 16.4M redundant values per sample in the recurrent path.
    - Per-frame DOY encoding added to encoder — gives ConvLSTM explicit
      temporal positioning (season awareness per frame).
    - Weather FiLM modulation applied to harmonics before DC offset — scale-only
      ``harmonics * (1 + gamma)`` where gamma is zero-initialised.
    """
    n_bands = output_shape[-1]
    output_steps = output_shape[0]
    img_h = input_shape[1]
    img_w = input_shape[2]
    time_steps = input_shape[0]

    # --- Inputs ---
    img_input = Input(shape=input_shape, name='sentinel2_sequence')
    landcover_input = Input(shape=landcover_shape, name='landcover_map')
    temporal_input = Input(shape=temporal_shape, name='temporal_metadata')
    weather_input = Input(shape=weather_shape, name='weather_sequence')
    doy_input = Input(shape=target_start_doy_shape, name='target_start_doy')

    # --- Per-frame DOY encoding: tile (B, 10, 3) spatially for encoder ---
    # (B, 10, 3) -> (B, 10, 1, 1, 3) -> (B, 10, 128, 128, 3)
    doy_reshaped = layers.Reshape(
        (time_steps, 1, 1, temporal_shape[-1]), name='doy_reshape')(temporal_input)
    doy_tiled = layers.UpSampling3D(
        size=(1, img_h, img_w), name='doy_tile')(doy_reshaped)

    # --- Encoder input: BAP-filled S2 + per-frame DOY ---
    # Landcover is no longer tiled here — it joins at the decoder stage.
    x = layers.Concatenate(axis=-1, name='input_concat')([img_input, doy_tiled])
    # Shape: (B, 10, 128, 128, 4 + 3 = 7)

    # --- ConvLSTM encoder ---
    # recurrent_dropout removed: it disables cuDNN fusion, causing a ~3-5×
    # slowdown (per-timestep Python loop instead of fused kernel).
    # Regular dropout=0.3 between layers + GroupNorm is sufficient regularisation.
    enc1 = layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding='same',
        dropout=0.3,
        return_sequences=True, name='encoder_lstm1'
    )(x)
    enc1 = layers.GroupNormalization(groups=4, axis=-1, name='encoder_norm1')(enc1)

    enc2 = layers.ConvLSTM2D(
        filters=48, kernel_size=(3, 3), padding='same',
        dropout=0.3,
        return_sequences=True, name='encoder_lstm2'
    )(enc1)
    enc2 = layers.GroupNormalization(groups=4, axis=-1, name='encoder_norm2')(enc2)

    latent = layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3), padding='same',
        dropout=0.3,
        return_sequences=False, name='encoder_lstm3'
    )(enc2)
    latent = layers.GroupNormalization(groups=4, axis=-1, name='latent_norm')(latent)
    # Shape: (B, H, W, 64)

    skip = LastTimestepLayer(name='skip_last')(enc2)  # (B, H, W, 48)

    # --- Global context: weather summary + temporal metadata ---
    weather_flat = layers.Flatten(name='weather_flatten')(weather_input)
    weather_enc = layers.Dense(
        32, activation='relu', name='weather_enc')(weather_flat)

    # Extract last timestep from per-frame temporal: (B, 10, 3) -> (B, 3)
    temporal_last = layers.Cropping1D(
        cropping=(time_steps - 1, 0), name='temporal_last')(temporal_input)
    temporal_last = layers.Reshape((temporal_shape[-1],), name='temporal_last_reshape')(temporal_last)
    temporal_enc = layers.Dense(
        16, activation='relu', name='temporal_enc')(temporal_last)
    context = layers.Concatenate(name='context_concat')([weather_enc, temporal_enc])

    context_reshaped = layers.Reshape((1, 1, 48), name='context_reshape')(context)
    context_tiled = layers.UpSampling2D(
        size=(img_h, img_w), name='context_tile')(context_reshaped)
    # Shape: (B, H, W, 48)

    # --- Fuse latent, skip, global context, and landcover ---
    latent_fused = layers.Concatenate(
        axis=-1, name='latent_fused')([latent, skip, context_tiled, landcover_input])
    # Shape: (B, H, W, 64 + 48 + 48 + 10 = 170)

    # --- Fourier coefficient head ---
    coefficients = FourierCoefficientHead(
        n_bands=n_bands,
        n_harmonics=n_harmonics,
        hidden_units=64,
        name='fourier_head'
    )(latent_fused)
    # Shape: (B, H, W, n_bands, 2K)

    # --- DC offset: separate Conv2D from latent_fused ---
    dc_offset = layers.Conv2D(
        n_bands, (1, 1), activation=None,
        kernel_initializer='zeros', bias_initializer='zeros',
        name='dc_offset'
    )(latent_fused)
    # Shape: (B, H, W, n_bands)

    # --- Fourier synthesis ---
    outputs_harmonics = FourierSynthesisLayer(
        output_steps=output_steps,
        n_harmonics=n_harmonics,
        cycle_length_days=cycle_length_days,
        step_interval=step_interval,
        name='fourier_synthesis'
    )([coefficients, doy_input])
    # Shape: (B, T, H, W, n_bands)

    # --- Weather FiLM modulation ---
    # Scale-only FiLM: harmonics * (1 + gamma) where gamma starts at zero.
    # Zero-init ensures identity (no modulation) at start of training.
    # Scale-only (no shift) means weather can't create signal from nothing.
    # Applied before DC offset so weather modulates oscillatory harmonics,
    # not the sustained trend.
    weather_gamma = layers.TimeDistributed(
        layers.Dense(
            n_bands, activation=None,
            kernel_initializer='zeros', bias_initializer='zeros',
            name='film_dense'
        ),
        name='weather_film'
    )(weather_input)
    # Shape: (B, 20, n_bands)

    # Broadcast gamma spatially: (B, 20, 4) -> (B, 20, 1, 1, 4) -> (B, 20, 128, 128, 4)
    gamma_reshaped = layers.Reshape(
        (output_steps, 1, 1, n_bands), name='film_reshape')(weather_gamma)
    gamma_tiled = layers.UpSampling3D(
        size=(1, img_h, img_w), name='film_tile')(gamma_reshaped)

    # harmonics * (1 + gamma)
    film_modulated = layers.Multiply(
        name='film_multiply')([outputs_harmonics, gamma_tiled])
    outputs_modulated = layers.Add(
        name='film_add')([outputs_harmonics, film_modulated])
    # Shape: (B, T, H, W, n_bands)

    # --- Broadcast DC offset across time ---
    # (B, H, W, n_bands) -> (B, 1, H, W, n_bands) -> (B, T, H, W, n_bands)
    dc_4d = layers.Reshape((1, img_h, img_w, n_bands), name='dc_reshape')(dc_offset)
    dc_5d = layers.UpSampling3D(size=(output_steps, 1, 1), name='dc_tile')(dc_4d)
    outputs = layers.Add(name='dc_add')([outputs_modulated, dc_5d])

    model = models.Model(
        inputs=[
            img_input,
            landcover_input,
            temporal_input,
            weather_input,
            doy_input,
        ],
        outputs=outputs
    )
    return model


def load_model(model_path: str, compile: bool = True) -> keras.Model:
    """
    Load a saved Keras model with all custom layers registered.

    :param model_path: Path to the .keras model file.
    :param compile: Whether to compile the model after loading.
    :raises FileNotFoundError: If model_path does not exist.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    custom_objects = {
        'CloudAwareGatingLayer': CloudAwareGatingLayer,
        'LearnableScale': LearnableScale,
        'LastTimestepLayer': LastTimestepLayer,
        'FourierCoefficientHead': FourierCoefficientHead,
        'FourierSynthesisLayer': FourierSynthesisLayer,
    }

    return keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=compile
    )
