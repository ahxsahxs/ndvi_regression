import keras
from keras import layers
import tensorflow as tf
@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class SplineLoss(keras.losses.Loss):
    """
    Custom loss for spline-based predictions with multiple regularization terms.
    Ignores cloud areas (where cloudmask == 1) when computing losses.
    """
    def __init__(self, smoothness_weight=0, curvature_weight=0, 
                 name='spline_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smoothness_weight = smoothness_weight
        self.curvature_weight = curvature_weight
    
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: dict with keys:
                - "deltas": (batch, output_frames, h, w, channels)
                - "cloudmask": (batch, output_frames, h, w, channels, 1)
            y_pred: dict with key:
                - "deltas": (batch, output_frames, h, w, channels)
        """
        y_true_img = y_true["deltas"]
        y_true_cloudmask = y_true["cloudmask"]    
        y_pred_img = y_pred["deltas"]
        
        # Create binary mask: 0 where clouds (mask==1), 1 where clear (mask==0)
        # Squeeze the last dimension to match y_true_img shape
        clear_mask = 1.0 - y_true_cloudmask  # (batch, frames, h, w, 1)
        
        # Main reconstruction loss (MSE) - only on clear pixels
        squared_error = tf.square(y_true_img - y_pred_img)
        masked_squared_error = squared_error * clear_mask
        
        # Compute mean over non-cloud pixels
        num_clear_pixels = tf.reduce_sum(clear_mask) + 1e-8  # Add epsilon to avoid division by zero
        mse_loss = tf.reduce_sum(masked_squared_error) / num_clear_pixels
        
        total_loss = mse_loss
        
        # First-order smoothness (velocity) - mask temporal differences
        if self.smoothness_weight > 0:
            d1 = y_pred_img[:, 1:, :, :, :] - y_pred_img[:, :-1, :, :, :]
            
            # Mask for temporal differences: both frames must be clear
            clear_mask_t1 = clear_mask[:, 1:, :, :, :]
            clear_mask_t0 = clear_mask[:, :-1, :, :, :]
            clear_mask_temporal = clear_mask_t1 * clear_mask_t0
            
            masked_d1 = tf.square(d1) * clear_mask_temporal
            num_clear_temporal = tf.reduce_sum(clear_mask_temporal) + 1e-8
            smoothness_loss = tf.reduce_sum(masked_d1) / num_clear_temporal
            
            total_loss += self.smoothness_weight * smoothness_loss
        
        # Second-order smoothness (acceleration/curvature)
        if self.curvature_weight > 0:
            d1 = y_pred_img[:, 1:, :, :, :] - y_pred_img[:, :-1, :, :, :]
            d2 = d1[:, 1:, :, :, :] - d1[:, :-1, :, :, :]
            
            # Mask for second-order differences: all three frames must be clear
            clear_mask_t2 = clear_mask[:, 2:, :, :, :]
            clear_mask_t1 = clear_mask[:, 1:-1, :, :, :]
            clear_mask_t0 = clear_mask[:, :-2, :, :, :]
            clear_mask_curvature = clear_mask_t2 * clear_mask_t1 * clear_mask_t0
            
            masked_d2 = tf.square(d2) * clear_mask_curvature
            num_clear_curvature = tf.reduce_sum(clear_mask_curvature) + 1e-8
            curvature_loss = tf.reduce_sum(masked_d2) / num_clear_curvature
            
            total_loss += self.curvature_weight * curvature_loss
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "smoothness_weight": self.smoothness_weight,
            "curvature_weight": self.curvature_weight
        })
        return config

@keras.utils.register_keras_serializable(package="ConvLSTMSpline")
class kNDVILoss(keras.losses.Loss):
    """
    Custom loss that combines L1 loss on spectral bands and kNDVI.
    
    Loss = 0.125*L1(B02) + 0.125*L1(B03) + 0.125*L1(B04) + 0.125*L1(B8a) + 0.5*L1(kNDVI)
    
    Also supports smoothness and curvature regularization.
    """
    def __init__(self, smoothness_weight=0, curvature_weight=0, 
                 name='kndvi_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smoothness_weight = smoothness_weight
        self.curvature_weight = curvature_weight
        self.epsilon = 1e-5
    
    def kndvi(self, b04, b8a):
        """
        Compute kNDVI = tanh[(B8a-B04)/(B8a+B04+epsilon)]
        """
        # Using tanh for kNDVI as specified
        # Note: B04 is index 2, B8a is index 3 in the 4-channel input
        diff = b8a - b04
        add = b8a + b04 + self.epsilon
        return tf.math.tanh(diff / add)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: dict with keys:
                - "deltas": (batch, output_frames, h, w, channels)
                - "sentinel2": (batch, output_frames, h, w, channels) [Target Absolute Values]
                - "cloudmask": (batch, output_frames, h, w, channels, 1)
            y_pred: dict with key:
                - "deltas": (batch, output_frames, h, w, channels)
        """
        y_true_deltas = y_true["deltas"]
        y_true_sentinel2 = y_true["sentinel2"]
        y_true_cloudmask = y_true["cloudmask"]    
        y_pred_deltas = y_pred["deltas"]
        
        # Reconstruct absolute predictions
        # We know y_true_sentinel2 = last_img + y_true_deltas
        # So last_img = y_true_sentinel2 - y_true_deltas
        # And y_pred_sentinel2 = last_img + y_pred_deltas
        #                      = (y_true_sentinel2 - y_true_deltas) + y_pred_deltas
        
        last_img = y_true_sentinel2 - y_true_deltas
        y_pred_sentinel2 = last_img + y_pred_deltas
        
        # Create binary mask: 0 where clouds (mask==1), 1 where clear (mask==0)
        clear_mask = 1.0 - y_true_cloudmask  # (batch, frames, h, w, 1)
        
        # --- Spectral Band L1 Loss ---
        # Bands: B02(0), B03(1), B04(2), B8a(3)
        # Weights: 0.125 each
        
        abs_error = tf.abs(y_true_sentinel2 - y_pred_sentinel2)
        masked_abs_error = abs_error * clear_mask
        
        # We can compute mean L1 per band or sum and divide.
        # The prompt says: "aggregated (by weighted sum) with weights 0.125..."
        # This implies we sum the errors (or mean errors) weighted by these factors.
        # Usually losses are means over the batch/pixels.
        
        num_clear_pixels = tf.reduce_sum(clear_mask) + 1e-8
        
        # Calculate L1 loss per band
        # masked_abs_error is (B, T, H, W, 4)
        
        # Split channels
        l1_b02 = tf.reduce_sum(masked_abs_error[..., 0]) / num_clear_pixels
        l1_b03 = tf.reduce_sum(masked_abs_error[..., 1]) / num_clear_pixels
        l1_b04 = tf.reduce_sum(masked_abs_error[..., 2]) / num_clear_pixels
        l1_b8a = tf.reduce_sum(masked_abs_error[..., 3]) / num_clear_pixels
        
        spectral_loss = 0.125 * (l1_b02 + l1_b03 + l1_b04 + l1_b8a)
        
        # --- kNDVI Loss ---
        # kNDVI uses B04 (idx 2) and B8a (idx 3)
        
        # True kNDVI
        kndvi_true = self.kndvi(y_true_sentinel2[..., 2], y_true_sentinel2[..., 3])
        
        # Pred kNDVI
        kndvi_pred = self.kndvi(y_pred_sentinel2[..., 2], y_pred_sentinel2[..., 3])
        
        # L1 Loss for kNDVI
        # kNDVI shape is (B, T, H, W) -> expand to (B, T, H, W, 1) for masking
        kndvi_diff = tf.abs(kndvi_true - kndvi_pred)
        kndvi_diff = tf.expand_dims(kndvi_diff, axis=-1)
        
        masked_kndvi_diff = kndvi_diff * clear_mask
        l1_kndvi = tf.reduce_sum(masked_kndvi_diff) / num_clear_pixels
        
        kndvi_loss_term = 0.5 * l1_kndvi
        
        total_loss = spectral_loss + kndvi_loss_term
        
        # --- Regularization (Smoothness/Curvature) ---
        # Applied on DELTAS (velocity) as in SplineLoss
        
        # First-order smoothness (velocity)
        if self.smoothness_weight > 0:
            d1 = y_pred_deltas[:, 1:, :, :, :] - y_pred_deltas[:, :-1, :, :, :]
            
            clear_mask_t1 = clear_mask[:, 1:, :, :, :]
            clear_mask_t0 = clear_mask[:, :-1, :, :, :]
            clear_mask_temporal = clear_mask_t1 * clear_mask_t0
            
            masked_d1 = tf.square(d1) * clear_mask_temporal
            num_clear_temporal = tf.reduce_sum(clear_mask_temporal) + 1e-8
            smoothness_loss = tf.reduce_sum(masked_d1) / num_clear_temporal
            
            total_loss += self.smoothness_weight * smoothness_loss
        
        # Second-order smoothness (acceleration/curvature)
        if self.curvature_weight > 0:
            d1 = y_pred_deltas[:, 1:, :, :, :] - y_pred_deltas[:, :-1, :, :, :]
            d2 = d1[:, 1:, :, :, :] - d1[:, :-1, :, :, :]
            
            clear_mask_t2 = clear_mask[:, 2:, :, :, :]
            clear_mask_t1 = clear_mask[:, 1:-1, :, :, :]
            clear_mask_t0 = clear_mask[:, :-2, :, :, :]
            clear_mask_curvature = clear_mask_t2 * clear_mask_t1 * clear_mask_t0
            
            masked_d2 = tf.square(d2) * clear_mask_curvature
            num_clear_curvature = tf.reduce_sum(clear_mask_curvature) + 1e-8
            curvature_loss = tf.reduce_sum(masked_d2) / num_clear_curvature
            
            total_loss += self.curvature_weight * curvature_loss
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "smoothness_weight": self.smoothness_weight,
            "curvature_weight": self.curvature_weight
        })
        return config