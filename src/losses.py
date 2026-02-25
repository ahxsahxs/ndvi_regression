import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class DeltaRegressionLoss(keras.losses.Loss):
    """
    Log-Cosh regression loss on reflectance deltas with spatial edge penalty.

    Components:
    - Log-Cosh loss on masked (cloud-free) reflectance deltas — primary objective.
    - Edge penalty to encourage spatial structure matching — regulariser.
    - Optional asymmetric under-prediction penalty.

    y_true structure: (B, T, H, W, 9) — [cloudmask(1), deltas(4), BAP(4)]
    y_pred structure: (B, T, H, W, 4) — predicted reflectance deltas
    """
    def __init__(
        self,
        regression_weight=5.0,
        edge_weight=0.1,
        under_penalty=0.5,
        name='delta_regression_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.regression_weight = regression_weight
        self.edge_weight = edge_weight
        self.under_penalty = under_penalty

    def _log_cosh(self, y_true, y_pred):
        """Log-Cosh over last axis (bands), returning (B, T, H, W)."""
        error = y_pred - y_true
        return tf.reduce_mean(tf.math.log(tf.math.cosh(error)), axis=-1)

    def call(self, y_true, y_pred):
        cloudmask = y_true[..., 0:1]      # (B, T, H, W, 1)
        true_deltas = y_true[..., 1:5]    # (B, T, H, W, 4)
        pred_deltas = y_pred              # (B, T, H, W, 4)

        sample_weight = 1.0 - cloudmask   # 1 = clear, 0 = cloudy
        weight_sum = tf.reduce_sum(sample_weight) + 1e-6

        # --- 1. Regression loss ---
        reg_loss = self._log_cosh(true_deltas, pred_deltas)  # (B, T, H, W)

        # Asymmetric penalty: scale up loss when |pred| < |true|
        if self.under_penalty > 0:
            abs_true = tf.reduce_mean(tf.abs(true_deltas), axis=-1)
            abs_pred = tf.reduce_mean(tf.abs(pred_deltas), axis=-1)
            under_mask = tf.cast(abs_pred < abs_true, tf.float32)
            reg_loss = reg_loss * (1.0 + self.under_penalty * under_mask)

        masked_reg = reg_loss * tf.squeeze(sample_weight, axis=-1)
        total_reg = tf.reduce_sum(masked_reg) / weight_sum

        # --- 2. Edge penalty ---
        shape = tf.shape(true_deltas)
        H, W = shape[2], shape[3]

        true_flat = tf.reshape(true_deltas, [-1, H, W, shape[4]])
        pred_flat = tf.reshape(pred_deltas, [-1, H, W, shape[4]])
        w_flat = tf.reshape(sample_weight, [-1, H, W, 1])

        true_dy, true_dx = tf.image.image_gradients(true_flat)
        pred_dy, pred_dx = tf.image.image_gradients(pred_flat)

        edge_y = tf.expand_dims(self._log_cosh(true_dy, pred_dy), -1) * w_flat
        edge_x = tf.expand_dims(self._log_cosh(true_dx, pred_dx), -1) * w_flat
        total_edge = tf.reduce_sum(edge_y + edge_x) / weight_sum

        return self.regression_weight * total_reg + self.edge_weight * total_edge

    def get_config(self):
        config = super().get_config()
        config.update({
            'regression_weight': self.regression_weight,
            'edge_weight': self.edge_weight,
            'under_penalty': self.under_penalty,
        })
        return config
