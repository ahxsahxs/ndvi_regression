import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class DeltaRegressionLoss(keras.losses.Loss):
    """
    Multi-component loss for reflectance delta prediction.

    Components:
    1. Stable log-cosh regression with per-band weights (primary)
    2. NDVI consistency — NIR/Red inter-band coupling
    3. Temporal smoothness — 2nd-order Huber on delta curvature

    y_true structure: (B, T, H, W, 9) — [cloudmask(1), deltas(4), BAP(4)]
    y_pred structure: (B, T, H, W, 4) — predicted reflectance deltas
    """
    def __init__(
        self,
        regression_weight=5.0,
        ndvi_weight=1.5,
        temporal_weight=0.1,
        ndvi_eps=0.05,
        band_weights=(0.75, 0.75, 1.25, 3.5),
        horizon_w_min=0.5,
        horizon_w_max=2.0,
        name='delta_regression_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.regression_weight = regression_weight
        self.ndvi_weight = ndvi_weight
        self.temporal_weight = temporal_weight
        self.ndvi_eps = ndvi_eps
        self.band_weights = tuple(band_weights)
        self.horizon_w_min = horizon_w_min
        self.horizon_w_max = horizon_w_max

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

        # ==============================================================
        # Horizon temporal weights: linear ramp from w_min to w_max
        # Later timesteps (harder, larger deltas) get more gradient.
        # Normalised so mean weight = 1.0 to preserve loss magnitude.
        # ==============================================================
        T = tf.shape(y_true)[1]
        t_idx = tf.cast(tf.range(T), dtype=y_true.dtype)
        t_norm = t_idx / tf.cast(T - 1, dtype=y_true.dtype)
        raw_hw = self.horizon_w_min + (self.horizon_w_max - self.horizon_w_min) * t_norm
        horizon_w = raw_hw / tf.reduce_mean(raw_hw)      # mean = 1.0
        horizon_w = tf.reshape(horizon_w, (1, -1, 1, 1))  # (1, T, 1, 1)

        weighted_clear = clear_sq * horizon_w  # (B, T, H, W)
        n_clear_w = tf.reduce_sum(weighted_clear) + 1e-6

        # ==============================================================
        # 1. Stable log-cosh regression with per-band weights
        # ==============================================================
        error = pred_deltas - true_deltas
        reg_loss = self._weighted_log_cosh(error)  # (B, T, H, W)
        total_reg = tf.reduce_sum(reg_loss * weighted_clear) / n_clear_w

        # ==============================================================
        # 2. NDVI consistency loss
        # ==============================================================
        full_true = tf.maximum(bap + true_deltas, 0.0)
        full_pred = tf.maximum(bap + pred_deltas, 0.0)

        ndvi_true = self._compute_ndvi(full_true)  # (B, T, H, W, 1)
        ndvi_pred = self._compute_ndvi(full_pred)

        ndvi_err = self._stable_log_cosh(ndvi_pred - ndvi_true)  # (B,T,H,W,1)
        ndvi_err = tf.squeeze(ndvi_err, axis=-1)                 # (B,T,H,W)
        total_ndvi = tf.reduce_sum(ndvi_err * weighted_clear) / n_clear_w

        # ==============================================================
        # 3. Temporal smoothness (2nd-order difference, Huber)
        # ==============================================================
        # Penalises curvature (changes in rate) across ALL bands — prevents
        # non-physical jumps while allowing linear trends (greenup/senescence).
        d_pred = pred_deltas[:, 1:] - pred_deltas[:, :-1]   # (B, T-1, H, W, 4)
        dd_pred = d_pred[:, 1:] - d_pred[:, :-1]            # (B, T-2, H, W, 4)
        total_temporal = tf.reduce_mean(tf.keras.losses.huber(
            tf.zeros_like(dd_pred), dd_pred, delta=0.02
        ))

        # ==============================================================
        # Combine
        # ==============================================================
        return (
            self.regression_weight * total_reg
            + self.ndvi_weight * total_ndvi
            + self.temporal_weight * total_temporal
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'regression_weight': self.regression_weight,
            'ndvi_weight': self.ndvi_weight,
            'temporal_weight': self.temporal_weight,
            'ndvi_eps': self.ndvi_eps,
            'band_weights': list(self.band_weights),
            'horizon_w_min': self.horizon_w_min,
            'horizon_w_max': self.horizon_w_max,
        })
        return config
