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

    No L2 regularisation on conv layers — AdamW weight_decay handles that.
    coeff_scale initialised at 0.1 (not 1.0) so the model starts near-zero
    and learns to grow, rather than collapsing from 1.0.

    Note: DC offset is a separate Conv2D in build_eo_convlstm_model so this
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

        # Learnable amplitude scale — init at 0.1 so predictions start small
        # and grow as the model learns, rather than collapsing from 1.0.
        self.coeff_scale = self.add_weight(
            name='coeff_scale', shape=(1, 1, 1, n_bands, self.n_coeffs),
            initializer=keras.initializers.Constant(0.1),
            trainable=True
        )

    def call(self, latent, training=None):
        x = self.conv1(latent)
        x = self.conv2(x)

        raw = self.conv_coeffs(x)  # (B, H, W, n_bands * 2K)

        # Reshape using keras.ops — tf.reshape with tf.shape() breaks Keras 3
        # symbolic graph tracing; keras.ops.reshape preserves tensor connectivity.
        s = keras.ops.shape(raw)
        coeffs = keras.ops.reshape(raw, (s[0], s[1], s[2], self.n_bands, self.n_coeffs))
        coeffs = coeffs * self.coeff_scale

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
        step_offsets = keras.ops.convert_to_tensor(step_values, dtype='float32')  # (T,)
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
        temporal_shape=(3,),
        weather_shape=(20, 21),
        target_start_doy_shape=(1,),
        output_shape=(20, 128, 128, 4),
        n_harmonics=6,
        cycle_length_days=365,
        step_interval=5,
):
    """
    Constructs the Fourier Harmonics model for satellite reflectance forecasting.

    Architecture:
    1. Encode BAP-filled Sentinel-2 + landcover through stacked ConvLSTM.
    2. Fuse latent state with global weather/temporal context.
    3. Predict Fourier harmonic coefficients per pixel per band.
    4. Synthesise temporal deltas via DOY-conditioned Fourier basis.
    5. Add DC offset (sustained trend) — separate Conv2D broadcast across time.
    6. Add per-timestep weather residual — TimeDistributed Dense broadcast spatially.

    Key changes vs previous version:
    - CloudAwareGating removed: input is BAP-filled, gating was zeroing valid data.
    - L2 regularisation removed from all heads: AdamW weight_decay handles it.
    - DC offset added: model can represent sustained seasonal trends.
    - Per-timestep weather residual: short-term anomalies modulate each forecast step.
    - coeff_scale initialised at 0.1 instead of 1.0.
    - All custom layer ops use keras.ops (not tf.*) for Keras 3 graph tracing.
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

    # --- Tile landcover across time ---
    landcover_reshaped = layers.Reshape(
        (1,) + landcover_shape, name='lc_reshape')(landcover_input)
    landcover_tiled = layers.UpSampling3D(
        size=(time_steps, 1, 1), name='lc_tile')(landcover_reshaped)

    # --- Encoder input: BAP-filled S2 + landcover (no cloud gating) ---
    # The sentinel2_sequence is already BAP-filled in the data pipeline.
    # Cloud gating was previously zeroing out those clean observations, depriving
    # the encoder of temporal signal whenever cloud cover was high.
    x = layers.Concatenate(axis=-1, name='input_concat')([img_input, landcover_tiled])

    # --- ConvLSTM encoder ---
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

    latent = layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3), padding='same',
        dropout=0.3, recurrent_dropout=0.2,
        return_sequences=False, name='encoder_lstm3'
    )(enc2)
    latent = layers.GroupNormalization(groups=4, axis=-1, name='latent_norm')(latent)
    # Shape: (B, H, W, 64)

    skip = LastTimestepLayer(name='skip_last')(enc2)  # (B, H, W, 48)

    # --- Global context: weather summary + temporal metadata ---
    weather_flat = layers.Flatten(name='weather_flatten')(weather_input)
    weather_enc = layers.Dense(
        32, activation='relu', name='weather_enc')(weather_flat)
    temporal_enc = layers.Dense(
        16, activation='relu', name='temporal_enc')(temporal_input)
    context = layers.Concatenate(name='context_concat')([weather_enc, temporal_enc])

    context_reshaped = layers.Reshape((1, 1, 48), name='context_reshape')(context)
    context_tiled = layers.UpSampling2D(
        size=(img_h, img_w), name='context_tile')(context_reshaped)
    # Shape: (B, H, W, 48)

    # --- Fuse latent, skip, and global context ---
    latent_fused = layers.Concatenate(
        axis=-1, name='latent_fused')([latent, skip, context_tiled])
    # Shape: (B, H, W, 64 + 48 + 48 = 160)

    # --- Fourier coefficient head ---
    coefficients = FourierCoefficientHead(
        n_bands=n_bands,
        n_harmonics=n_harmonics,
        hidden_units=64,
        name='fourier_head'
    )(latent_fused)
    # Shape: (B, H, W, n_bands, 2K)

    # --- DC offset: separate Conv2D from latent_fused ---
    # Predicts a per-pixel per-band constant added to every forecast step,
    # enabling sustained trend predictions that pure harmonics cannot represent.
    # Kept as a standalone layer (not inside FourierCoefficientHead) so that
    # FourierCoefficientHead stays single-output — Keras 3 cannot trace
    # multi-tensor returns from custom layers through the functional graph.
    dc_offset = layers.Conv2D(
        n_bands, (1, 1), activation=None, name='dc_offset'
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

    # --- Broadcast DC offset across time ---
    # (B, H, W, n_bands) -> (B, 1, H, W, n_bands) -> (B, T, H, W, n_bands)
    dc_4d = layers.Reshape((1, img_h, img_w, n_bands), name='dc_reshape')(dc_offset)
    dc_5d = layers.UpSampling3D(size=(output_steps, 1, 1), name='dc_tile')(dc_4d)
    outputs_fourier = layers.Add(name='dc_add')([outputs_harmonics, dc_5d])

    # --- Per-timestep weather residual ---
    # Allows short-term weather anomalies to modulate individual forecast steps,
    # rather than being collapsed into a single global context vector.
    weather_corr = layers.TimeDistributed(
        layers.Dense(n_bands, activation=None),
        name='weather_residual'
    )(weather_input)  # (B, T, n_bands)

    # (B, T, n_bands) -> (B, T, 1, 1, n_bands) -> (B, T, H, W, n_bands)
    weather_corr_4d = layers.Reshape(
        (output_steps, 1, 1, n_bands), name='weather_corr_reshape')(weather_corr)
    weather_corr_5d = layers.UpSampling3D(
        size=(1, img_h, img_w), name='weather_corr_tile')(weather_corr_4d)

    outputs = layers.Add(name='output_add')([outputs_fourier, weather_corr_5d])

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
        'LastTimestepLayer': LastTimestepLayer,
        'FourierCoefficientHead': FourierCoefficientHead,
        'FourierSynthesisLayer': FourierSynthesisLayer,
    }

    return keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=compile
    )
