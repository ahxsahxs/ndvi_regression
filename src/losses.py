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
    Combined loss for delta prediction with balanced component scaling.
    
    Components (with default weights prioritizing regression):
    - Huber loss on deltas for regression (weight=10.0, primary objective)
    - Variance penalty to prevent mode collapse (weight=1.0, regularizer)
    - kNDVI loss for spectral consistency (weight=0.1 when enabled, auxiliary)
    
    All components are normalized to similar scales before weighting to ensure
    no single component dominates the gradient signal.
    
    The variance penalty computes the absolute difference between the spatial
    variance of true deltas and predicted deltas, encouraging the model to
    preserve spatial variability rather than collapsing to constant predictions.
    """
    def __init__(
        self, 
        regression_weight=5.0,   # Primary objective weight
        variance_weight=2.0,     # Regularizer weight
        kndvi_weight=0.0,        # Auxiliary loss weight (0 = disabled, set via callback)
        kndvi_clip=0.5,          # Max kNDVI difference (gradient clipping)
        sigma=1.0,               # RBF kernel sigma
        name='improved_kndvi_loss', 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.regression_weight = regression_weight
        self.variance_weight = variance_weight
        self.kndvi_weight = tf.Variable(kndvi_weight, trainable=False, dtype=tf.float32, name='kndvi_weight')
        self.kndvi_clip = kndvi_clip
        self.sigma = sigma
        
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
        # Expected range: ~0.001 to 0.05 for well-behaved deltas
        reg_loss = self.fn_huber(true_deltas, pred_deltas)
        masked_reg_loss = reg_loss * tf.squeeze(sample_weight, axis=-1)
        total_reg_loss = (tf.reduce_sum(masked_reg_loss) / weight_sum) * 10.0
        
        # --- 2. Variance Penalty (prevent mode collapse) ---
        # Compute spatial variance of true and predicted deltas per sample
        # Expected range: ~0.0001 to 0.01 (variance of small deltas)
        weights_expanded = sample_weight  # (B, T, H, W, 1)
        
        # Weighted mean per sample per band
        true_weighted = true_deltas * weights_expanded
        pred_weighted = pred_deltas * weights_expanded
        
        # Mean across spatial dims (H, W) - shape becomes (B, T, 1, 1, 4)
        spatial_weight_sum = tf.reduce_sum(weights_expanded, axis=[2, 3], keepdims=True) + 1e-6
        true_mean = tf.reduce_sum(true_weighted, axis=[2, 3], keepdims=True) / spatial_weight_sum
        pred_mean = tf.reduce_sum(pred_weighted, axis=[2, 3], keepdims=True) / spatial_weight_sum
        
        # Variance = E[(x - mean)^2]
        spatial_weight_sum_squeezed = tf.reduce_sum(weights_expanded, axis=[2, 3]) + 1e-6
        true_var = tf.reduce_sum(weights_expanded * tf.square(true_deltas - true_mean), axis=[2, 3]) / spatial_weight_sum_squeezed
        pred_var = tf.reduce_sum(weights_expanded * tf.square(pred_deltas - pred_mean), axis=[2, 3]) / spatial_weight_sum_squeezed
        
        # Variance penalty: abs difference between true and pred variance
        # Scale by 100 to bring into similar range as regression loss (~0.01-0.1)
        variance_penalty = tf.reduce_mean(tf.abs(true_var - pred_var)) * 100.0
        
        # --- 3. kNDVI Loss (optional, controlled by kndvi_weight) ---
        # Expected range: ~0.01 to 0.3 (kNDVI differences)
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
        # Scale by 1.0 (no scaling needed as kNDVI is already in [0,1] range approx, diffs are small)
        total_kndvi_loss = tf.reduce_sum(masked_kndvi_loss) / weight_sum
        
        # --- Combine Losses with Weights ---
        # All components now normalized to ~0.01-0.1 range before weighting
        total_loss = (
            self.regression_weight * total_reg_loss +
            self.variance_weight * variance_penalty +
            self.kndvi_weight * total_kndvi_loss
        )
        
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'regression_weight': self.regression_weight,
            'variance_weight': self.variance_weight,
            'kndvi_weight': float(self.kndvi_weight.numpy()),
            'kndvi_clip': self.kndvi_clip,
            'sigma': self.sigma,
        })
        return config


class EnablekNDVICallback(keras.callbacks.Callback):
    """
    Callback to enable kNDVI loss component after a warmup period.
    
    During the warmup epochs, only regression loss and variance penalty are active.
    After warmup, the kNDVI loss is enabled with the specified weight.
    
    :param enable_epoch: Epoch at which to enable kNDVI loss (default 20).
    :type enable_epoch: int
    :param kndvi_weight: Weight for kNDVI loss when enabled (default 1.0).
    :type kndvi_weight: float
    """
    def __init__(self, enable_epoch=20, kndvi_weight=1.0):
        super().__init__()
        self.enable_epoch = enable_epoch
        self.kndvi_weight = kndvi_weight
        self.enabled = False
    
    def on_epoch_begin(self, epoch, logs=None):
        if not self.enabled and epoch >= self.enable_epoch:
            loss_fn = self.model.loss
            if hasattr(loss_fn, 'kndvi_weight'):
                loss_fn.kndvi_weight.assign(self.kndvi_weight)
                self.enabled = True
                print(f'\nEnablekNDVICallback: Enabled kNDVI loss with weight={self.kndvi_weight} at epoch {epoch}')
