import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class DeltaRegressionLoss(keras.losses.Loss):
    """
    Multi-component loss for reflectance delta prediction.

    Components:
    1. Stable log-cosh regression with per-band weights (primary)
    2. Edge penalty — spatial structure matching
    3. NDVI consistency — NIR/Red inter-band coupling
    4. Spatial diversity (MAD-based) — anti-collapse regulariser
    5. Temporal NDVI smoothness — phenological regularisation
    6. Asymmetric under-prediction penalty (per-band)

    y_true structure: (B, T, H, W, 9) — [cloudmask(1), deltas(4), BAP(4)]
    y_pred structure: (B, T, H, W, 4) — predicted reflectance deltas
    """
    def __init__(
        self,
        regression_weight=5.0,
        edge_weight=0.1,
        ndvi_weight=1.0,
        diversity_weight=0.5,
        temporal_weight=0.1,
        under_penalty=0.5,
        ndvi_eps=0.05,
        band_weights=(0.76, 0.87, 0.90, 1.47),
        name='delta_regression_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.regression_weight = regression_weight
        self.edge_weight = edge_weight
        self.ndvi_weight = ndvi_weight
        self.diversity_weight = diversity_weight
        self.temporal_weight = temporal_weight
        self.under_penalty = under_penalty
        self.ndvi_eps = ndvi_eps
        self.band_weights = tuple(band_weights)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_log_cosh(x):
        """Numerically stable log(cosh(x)) = |x| + softplus(-2|x|) - log(2).

        Finite for all finite x. Unlike tf.math.log(tf.math.cosh(x)),
        this never hits float32-inf (cosh overflows at |x| > 88.7).
        """
        abs_x = tf.abs(x)
        return abs_x + tf.math.softplus(-2.0 * abs_x) - tf.cast(
            tf.math.log(2.0), x.dtype
        )

    def _weighted_log_cosh(self, error):
        """Per-band weighted stable log-cosh, returns (B, T, H, W).

        error: (B, T, H, W, 4) — per-band errors
        """
        bw = tf.constant(self.band_weights, dtype=error.dtype)  # (4,)
        per_band = self._stable_log_cosh(error) * bw             # (B,T,H,W,4)
        # Normalise so mean weight = 1 (sum of weights / n_bands)
        return tf.reduce_sum(per_band, axis=-1) / tf.reduce_sum(bw)

    def _compute_ndvi(self, reflectance):
        """NDVI from 4-band reflectance [..., (B02, B03, B04, B8A)].

        Uses self.ndvi_eps for denominator stability.
        """
        red = reflectance[..., 2:3]  # B04
        nir = reflectance[..., 3:4]  # B8A
        return (nir - red) / (nir + red + self.ndvi_eps)

    # ------------------------------------------------------------------
    # Loss call
    # ------------------------------------------------------------------

    def call(self, y_true, y_pred):
        cloudmask = y_true[..., 0:1]       # (B, T, H, W, 1)
        true_deltas = y_true[..., 1:5]     # (B, T, H, W, 4)
        bap = y_true[..., 5:9]             # (B, T, H, W, 4)
        pred_deltas = y_pred               # (B, T, H, W, 4)

        clear = 1.0 - cloudmask            # 1 = clear, 0 = cloudy
        clear_sq = tf.squeeze(clear, axis=-1)  # (B, T, H, W)
        n_clear = tf.reduce_sum(clear_sq) + 1e-6

        # ==============================================================
        # 1. Stable log-cosh regression with per-band weights
        # ==============================================================
        error = pred_deltas - true_deltas
        reg_loss = self._weighted_log_cosh(error)  # (B, T, H, W)

        # --- 6. Per-band asymmetric under-prediction penalty ---
        if self.under_penalty > 0:
            # Compare per-band: under-predicted when |pred| < |true|
            under_mask = tf.cast(
                tf.abs(pred_deltas) < tf.abs(true_deltas), tf.float32
            )  # (B, T, H, W, 4)
            # Per-band weight for asymmetry
            bw = tf.constant(self.band_weights, dtype=error.dtype)
            asym_mult = tf.reduce_sum(
                under_mask * bw, axis=-1
            ) / tf.reduce_sum(bw)  # (B, T, H, W)
            reg_loss = reg_loss * (1.0 + self.under_penalty * asym_mult)

        total_reg = tf.reduce_sum(reg_loss * clear_sq) / n_clear

        # ==============================================================
        # 2. Edge penalty (stabilised)
        # ==============================================================
        shape = tf.shape(true_deltas)
        H, W = shape[2], shape[3]

        true_flat = tf.reshape(true_deltas, [-1, H, W, shape[4]])
        pred_flat = tf.reshape(pred_deltas, [-1, H, W, shape[4]])
        w_flat = tf.reshape(clear, [-1, H, W, 1])

        true_dy, true_dx = tf.image.image_gradients(true_flat)
        pred_dy, pred_dx = tf.image.image_gradients(pred_flat)

        edge_y = self._weighted_log_cosh(pred_dy - true_dy)  # (-1, H, W)
        edge_x = self._weighted_log_cosh(pred_dx - true_dx)
        w_flat_sq = tf.squeeze(w_flat, axis=-1)  # (-1, H, W)
        total_edge = tf.reduce_sum(
            (edge_y + edge_x) * w_flat_sq
        ) / (tf.reduce_sum(w_flat_sq) + 1e-6)

        # ==============================================================
        # 3. NDVI consistency loss
        # ==============================================================
        full_true = tf.maximum(bap + true_deltas, 0.0)
        full_pred = tf.maximum(bap + pred_deltas, 0.0)

        ndvi_true = self._compute_ndvi(full_true)  # (B, T, H, W, 1)
        ndvi_pred = self._compute_ndvi(full_pred)

        ndvi_err = self._stable_log_cosh(ndvi_pred - ndvi_true)  # (B,T,H,W,1)
        ndvi_err = tf.squeeze(ndvi_err, axis=-1)                 # (B,T,H,W)
        total_ndvi = tf.reduce_sum(ndvi_err * clear_sq) / n_clear

        # ==============================================================
        # 4. Spatial diversity (MAD-based)
        # ==============================================================
        # Mean Absolute Deviation over spatial dims per (batch, time, band)
        # clear: (B, T, H, W, 1)
        n_clear_spatial = tf.reduce_sum(clear, axis=[2, 3]) + 1e-6  # (B,T,1)

        pred_masked = pred_deltas * clear      # (B, T, H, W, 4)
        true_masked = true_deltas * clear

        pred_mean = tf.reduce_sum(pred_masked, axis=[2, 3]) / n_clear_spatial  # (B,T,4)
        true_mean = tf.reduce_sum(true_masked, axis=[2, 3]) / n_clear_spatial

        # Expand back to spatial dims for deviation
        pred_mean_exp = pred_mean[:, :, tf.newaxis, tf.newaxis, :]  # (B,T,1,1,4)
        true_mean_exp = true_mean[:, :, tf.newaxis, tf.newaxis, :]

        pred_mad = tf.reduce_sum(
            tf.abs(pred_deltas - pred_mean_exp) * clear, axis=[2, 3]
        ) / n_clear_spatial  # (B, T, 4)
        true_mad = tf.reduce_sum(
            tf.abs(true_deltas - true_mean_exp) * clear, axis=[2, 3]
        ) / n_clear_spatial

        diversity_gap = tf.nn.relu(true_mad - pred_mad)  # (B, T, 4)
        total_diversity = tf.reduce_mean(diversity_gap)

        # ==============================================================
        # 5. Temporal NDVI smoothness (2nd-order difference, Huber)
        # ==============================================================
        # ndvi_pred: (B, T, H, W, 1) — use all pixels (regularises model)
        ndvi_seq = tf.squeeze(ndvi_pred, axis=-1)        # (B, T, H, W)
        d_ndvi = ndvi_seq[:, 1:] - ndvi_seq[:, :-1]     # 1st diff (T-1)
        dd_ndvi = d_ndvi[:, 1:] - d_ndvi[:, :-1]        # 2nd diff (T-2)
        total_temporal = tf.reduce_mean(tf.keras.losses.huber(
            tf.zeros_like(dd_ndvi), dd_ndvi, delta=0.01
        ))

        # ==============================================================
        # Combine
        # ==============================================================
        return (
            self.regression_weight * total_reg
            + self.edge_weight * total_edge
            + self.ndvi_weight * total_ndvi
            + self.diversity_weight * total_diversity
            + self.temporal_weight * total_temporal
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'regression_weight': self.regression_weight,
            'edge_weight': self.edge_weight,
            'ndvi_weight': self.ndvi_weight,
            'diversity_weight': self.diversity_weight,
            'temporal_weight': self.temporal_weight,
            'under_penalty': self.under_penalty,
            'ndvi_eps': self.ndvi_eps,
            'band_weights': list(self.band_weights),
        })
        return config
