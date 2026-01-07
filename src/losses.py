import keras
from keras import layers
import tensorflow as tf

@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class MaskedHuberLoss(keras.losses.Loss):
    def __init__(self, delta=0.1, name='masked_huber_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        # Reduced delta to 0.1 to provide stronger gradients for small delta errors (typically in [-0.2, 0.2])
        self.huber = keras.losses.Huber(delta=delta, reduction="none")

    def call(self, y_true, y_pred):
        # y_true structure: (batch, time, w, h, 5)
        # channel 0: cloudmask
        # channels 1-4: deltas
        
        # y_pred structure: (batch, time, w, h, 4) matches deltas
        
        cloudmask = y_true[..., 0:1]
        y_true_deltas = y_true[..., 1:]
        
        # Invert mask: 1 for clear (valid), 0 for cloudy (invalid)
        sample_weight = 1.0 - cloudmask
        
        # Compute huber loss
        # Note: Huber loss with reduction=NONE returns (batch, temp_dims)
        loss = self.huber(y_true_deltas, y_pred) # (batch, time, w, h) (if channels are averaged by Huber default?)
        
        # sample_weight is (B, T, W, H, 1)
        # squeze sample_weight to (B, T, W, H)
        weights = tf.squeeze(sample_weight, axis=-1)
        
        masked_loss = loss * weights
        
        # Simple mean over the batch/time/space
        return tf.reduce_mean(masked_loss)
        
    def get_config(self):
        config = super().get_config()
        config.update({'delta': self.delta})
        return config

@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class kNDVILoss(keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, sigma=0.5, name='kndvi_loss', **kwargs):
        """
        Args:
            alpha: Weight for the regression (masked huber) component.
            beta: Weight for the kNDVI component.
            sigma: Sigma parameter for the kNDVI RBF kernel.
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.fn_huber = keras.losses.Huber(delta=0.1, reduction="none")

    def kernel_function(self, n, r):
        """
        RBF Kernel k(n, r) = exp(-(n-r)^2 / (2*sigma^2))
        """
        diff = n - r
        return tf.exp(- tf.square(diff) / (2 * self.sigma ** 2))

    def compute_kndvi(self, nir, red):
        """
        kNDVI = (1 - k(n,r)) / (1 + k(n,r))
        """
        k_val = self.kernel_function(nir, red)
        return (1.0 - k_val) / (1.0 + k_val + 1e-6)

    def call(self, y_true, y_pred):
        # y_true structure: (batch, time, w, h, 9)
        # channel 0: cloudmask
        # channels 1-4: deltas (true)
        # channels 5-8: BAP (base image)
        
        # y_pred structure: (batch, time, w, h, 4) matches deltas
        
        # --- Unpack ---
        cloudmask = y_true[..., 0:1] # (B, T, W, H, 1)
        true_deltas = y_true[..., 1:5] # (B, T, W, H, 4)
        bap = y_true[..., 5:9] # (B, T, W, H, 4)
        
        pred_deltas = y_pred # (B, T, W, H, 4)
        
        # --- 1. Regression Loss (Huber on Deltas) ---
        sample_weight = 1.0 - cloudmask
        # Note: We must squeeze the weight for Huber if reduction=None handles basic dims, 
        # but here we can just do elementwise mult.
        
        reg_loss = self.fn_huber(true_deltas, pred_deltas) # (B, T, W, H) likely averaged over channels
        # Expand dims for broadcasting if needed, but huber usually reduces last dim.
        # Let's ensure shape consistency.
        
        # Weighted mean
        masked_reg_loss = reg_loss * tf.squeeze(sample_weight, axis=-1)
        total_reg_loss = tf.reduce_sum(masked_reg_loss) / (tf.reduce_sum(sample_weight) + 1e-6)
        
        # --- 2. kNDVI Loss ---
        # Reconstruct Full Images
        # Bands: B02, B03, B04 (Red), B8A (NIR)
        # Indx:   0,   1,   2,         3
        
        true_full = bap + true_deltas
        pred_full = bap + pred_deltas
        
        # Extract Red and NIR
        true_red = true_full[..., 2]
        true_nir = true_full[..., 3]
        
        pred_red = pred_full[..., 2]
        pred_nir = pred_full[..., 3]
        
        true_kndvi = self.compute_kndvi(true_nir, true_red)
        pred_kndvi = self.compute_kndvi(pred_nir, pred_red)
        
        # L1 Loss on kNDVI
        kndvi_diff = tf.abs(true_kndvi - pred_kndvi)
        masked_kndvi_loss = kndvi_diff * tf.squeeze(sample_weight, axis=-1)
        total_kndvi_loss = tf.reduce_sum(masked_kndvi_loss) / (tf.reduce_sum(sample_weight) + 1e-6)
        
        return self.alpha * total_reg_loss + self.beta * total_kndvi_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'sigma': self.sigma
        })
        return config


@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class ImprovedkNDVILoss(keras.losses.Loss):
    """
    Enhanced kNDVI loss with improved convergence properties:
    - Gradient clipping: Caps extreme kNDVI differences to prevent saturation
    - Learned weights: Automatic loss component balancing via uncertainty weighting
    - Cumulative loss: Supervises the trajectory to fix magnitude/sign bias
    
    Note: Temporal smoothness was removed as it encouraged constant predictions.
    """
    def __init__(
        self, 
        kndvi_clip=0.5,      # Max kNDVI difference (gradient clipping)
        sigma=0.5,           # RBF kernel sigma
        learn_weights=True,  # Enable learned loss weights
        name='improved_kndvi_loss', 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.kndvi_clip = kndvi_clip
        self.sigma = sigma
        self.learn_weights = learn_weights
        
        # Learned loss weights (log-variance parameterization)
        if learn_weights:
            self.log_var_reg = tf.Variable(0.0, trainable=True, name='log_var_reg')
            self.log_var_kndvi = tf.Variable(0.0, trainable=True, name='log_var_kndvi')
        
        # Reduced Huber delta to 0.1 for stronger gradients on small delta errors
        self.fn_huber = keras.losses.Huber(delta=0.1, reduction="none")



    def kernel_function(self, n, r):
        """RBF Kernel k(n, r) = exp(-(n-r)^2 / (2*sigma^2))"""
        diff = n - r
        return tf.exp(-tf.square(diff) / (2 * self.sigma ** 2))

    def compute_kndvi(self, nir, red):
        """kNDVI = (1 - k) / (1 + k)"""
        k_val = self.kernel_function(nir, red)
        return (1.0 - k_val) / (1.0 + k_val + 1e-6)

    def call(self, y_true, y_pred):
        # y_true: (batch, time, w, h, 9) - [mask(1), deltas(4), BAP(4)]
        # y_pred: (batch, time, w, h, 4) - predicted deltas
        
        # --- Unpack ---
        cloudmask = y_true[..., 0:1]
        true_deltas = y_true[..., 1:5]
        bap = y_true[..., 5:9]
        pred_deltas = y_pred
        
        sample_weight = 1.0 - cloudmask
        weight_sum = tf.reduce_sum(sample_weight) + 1e-6
        
        # --- 1. Regression Loss (Huber on Deltas) ---
        reg_loss = self.fn_huber(true_deltas, pred_deltas)
        masked_reg_loss = reg_loss * tf.squeeze(sample_weight, axis=-1)
        total_reg_loss = tf.reduce_sum(masked_reg_loss) / weight_sum
        
        # --- 2. kNDVI Loss on Deltas ---
        true_full = bap + true_deltas
        pred_full = bap + pred_deltas
        
        true_red, true_nir = true_full[..., 2], true_full[..., 3]
        pred_red, pred_nir = pred_full[..., 2], pred_full[..., 3]
        
        true_kndvi = self.compute_kndvi(true_nir, true_red)
        pred_kndvi = self.compute_kndvi(pred_nir, pred_red)
        
        # Clipped L1 to prevent gradient explosion
        kndvi_diff = tf.abs(true_kndvi - pred_kndvi)
        kndvi_diff = tf.minimum(kndvi_diff, self.kndvi_clip)
        
        masked_kndvi_loss = kndvi_diff * tf.squeeze(sample_weight, axis=-1)
        total_kndvi_loss = tf.reduce_sum(masked_kndvi_loss) / weight_sum
        
        # --- 3. Cumulative Loss (Trajectory Supervision) ---
        # "Area under the curve" / Integral supervision
        # Determines the total change over time, fixing magnitude/sign bias
        true_cumsum = tf.cumsum(true_deltas, axis=1) # (B, Time, H, W, 4)
        pred_cumsum = tf.cumsum(pred_deltas, axis=1)
        
        # We use Huber loss on the cumulative sums as well
        cum_loss = self.fn_huber(true_cumsum, pred_cumsum)
        masked_cum_loss = cum_loss * tf.squeeze(sample_weight, axis=-1)
        total_cum_loss = tf.reduce_sum(masked_cum_loss) / weight_sum

        # --- 4. Combine Losses ---
        # Temporal smoothness was REMOVED as it encouraged constant predictions
        if self.learn_weights:
            # Uncertainty weighting
            loss = (
                tf.exp(-self.log_var_reg) * (total_reg_loss + total_cum_loss) + self.log_var_reg +
                tf.exp(-self.log_var_kndvi) * total_kndvi_loss + self.log_var_kndvi
            )
        else:
            loss = (total_reg_loss + total_cum_loss) + total_kndvi_loss
        
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'kndvi_clip': self.kndvi_clip,
            'sigma': self.sigma,
            'learn_weights': self.learn_weights,
        })
        return config

