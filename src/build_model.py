
import tensorflow as tf
from keras import layers, models, Input, regularizers
import keras


@keras.utils.register_keras_serializable(package="EOModel")
class BSplineBasisLayer(layers.Layer):
    """
    Evaluates a Cubic B-Spline curve for a given time sequence.

    The curve is defined by control points. This layer computes the B-spline
    basis functions and combines them with the learned control points.

    Curve: :math:`S(t) = \\sum P_i \\cdot N_{i,3}(t)`
    where :math:`P_i` are control points and :math:`N_{i,3}` are cubic B-spline basis functions.

    :param output_steps: Number of future time steps to predict (e.g. 12).
    :type output_steps: int
    :param n_bands: Number of spectral bands (e.g. 4).
    :type n_bands: int
    :param n_control_points: Number of control points (knots + degree - 1).
                             Default 6 for 12 steps gives reasonable flexibility.
    :type n_control_points: int
    """
    def __init__(self, output_steps=12, n_bands=4, n_control_points=6, **kwargs):
        super().__init__(**kwargs)
        self.output_steps = output_steps
        self.n_bands = n_bands
        self.n_control_points = n_control_points
        self.degree = 3  # Cubic B-Splines
        
        # Precompute basis functions matrix
        self._basis_matrix = self._compute_basis_matrix() # (n_steps, n_control_points)
        
    def _compute_basis_matrix(self):
        """
        Computes the B-spline basis functions for uniform knots.
        Returns a matrix M of shape (output_steps, n_control_points)
        where M[t, i] is the value of the i-th basis function at time t.
        """
        # Time points normalized to [0, n_control_points - degree]
        # This range maps the full spline support to the time steps
        max_knot = self.n_control_points - self.degree
        t_seq = tf.linspace(0.0, float(max_knot), self.output_steps)
        
        # Knots for uniform B-spline
        # We need (n_control_points + degree + 1) knots
        # Typically for clamped B-splines or open uniform:
        # 0, 0, 0, 0, 1, 2, ..., k, k, k, k
        # Here we use uniform knots for simplicity: 0, 1, 2, ...
        # Standard uniform B-spline definition
        knots = tf.range(-self.degree, self.n_control_points + 1, dtype=tf.float32)
        
        basis_values = []
        for i in range(self.n_control_points):
            # Compute N_{i,3}(t) for all t in t_seq
            vals = self._cox_de_boor(t_seq, i, self.degree, knots)
            basis_values.append(vals)
            
        basis_matrix = tf.stack(basis_values, axis=1) # (Steps, BO)
        return basis_matrix

    def _cox_de_boor(self, t, i, k, knots):
        """
        Recursive Cox-de Boor formula for B-spline basis functions.
        i: basis function index
        k: degree
        t: time points (tensor)
        knots: knot vector
        """
        if k == 0:
            # 1 if knots[i] <= t < knots[i+1], else 0
            return tf.cast((t >= knots[i]) & (t < knots[i+1]), tf.float32)
        
        # Term 1 denominator
        denom1 = knots[i+k] - knots[i]
        term1 = 0.0
        if denom1 > 0:
            term1 = ((t - knots[i]) / denom1) * self._cox_de_boor(t, i, k-1, knots)
            
        # Term 2 denominator
        denom2 = knots[i+k+1] - knots[i+1]
        term2 = 0.0
        if denom2 > 0:
            term2 = ((knots[i+k+1] - t) / denom2) * self._cox_de_boor(t, i+1, k-1, knots)
            
        return term1 + term2

    def call(self, inputs):
        # inputs: (Batch, H, W, n_bands * n_control_points)
        shape = tf.shape(inputs)
        batch, h, w = shape[0], shape[1], shape[2]
        
        # Reshape to separate bands and control points
        # (Batch, H, W, n_bands, n_control_points)
        coeffs = tf.reshape(inputs, (batch, h, w, self.n_bands, self.n_control_points))
        
        # Perform spline evaluation: S(t) = sum(coeffs_i * basis_i(t))
        # We can implement this as a matrix multiplication
        
        # basis_matrix: (output_steps, n_control_points)
        
        # We want output: (Batch, H, W, n_bands, output_steps)
        # Einsum: 
        # b: batch, h: height, w: width, c: channels(bands), p: control_points, t: time
        # coeffs: bhwcp, basis: tp -> output: bhwct
        
        basis = tf.cast(self._basis_matrix, inputs.dtype)
        
        outputs = tf.einsum('bhwcp,tp->bhwct', coeffs, basis)
        
        # Permute to (Batch, Time, H, W, Bands)
        outputs = tf.transpose(outputs, perm=[0, 4, 1, 2, 3])
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_steps": self.output_steps,
            "n_bands": self.n_bands,
            "n_control_points": self.n_control_points,
        })
        return config




@keras.utils.register_keras_serializable(package="EOModel")
class BroadcastFusionLayer(layers.Layer):
    """
    Fuses a spatial tensor (Batch, H, W, C1) with a global feature vector (Batch, C2)
    by broadcasting the vector across the spatial dimensions and concatenating.

    The global vector is expanded and tiled to match the spatial dimensions of the
    spatial tensor, then they are concatenated along the channel axis.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: [spatial_tensor, global_vector]
        spatial_tensor, global_vector = inputs
        
        # global_vector: (Batch, Features)
        # spatial_tensor: (Batch, H, W, Channels)
        
        h = tf.shape(spatial_tensor)[1]
        w = tf.shape(spatial_tensor)[2]
        
        # Reshape global_vector to (Batch, 1, 1, Features)
        vector_expanded = tf.expand_dims(tf.expand_dims(global_vector, 1), 1)
        
        # Tile to (Batch, H, W, Features)
        vector_tiled = tf.tile(vector_expanded, [1, h, w, 1])
        
        # Concatenate along channel axis
        return tf.concat([spatial_tensor, vector_tiled], axis=-1)



def build_eo_convlstm_model(
        input_shape=(18, 128, 128, 4), 
        cloudmask_shape=(18, 128, 128, 1),
        landcover_shape=(128, 128, 10),
        temporal_shape=(3,),
        weather_shape=(12, 21),
        output_shape=(12, 128, 128, 4),
    ):
    """
    Constructs a Spatiotemporal Encoder-Decoder for Satellite Imagery Forecasting.

    This model fuses satellite imagery (Sentinel-2), cloud masks, landcover maps,
    temporal metadata, and weather variables to predict future satellite imagery.

    :param input_shape: Shape of the input image sequence (Time, H, W, C).
                        Default (18, 128, 128, 4).
    :type input_shape: tuple
    :param cloudmask_shape: Shape of the cloud mask sequence (Time, H, W, 1).
                            Default (18, 128, 128, 1).
    :type cloudmask_shape: tuple
    :param landcover_shape: Shape of the landcover map (H, W, Classes).
                           Default (128, 128, 10).
    :type landcover_shape: tuple
    :param temporal_shape: Shape of the temporal metadata (Features,).
                           Default (3,) for [Year, Cos_DOY, Sin_DOY].
    :type temporal_shape: tuple
    :param weather_shape: Shape of the weather sequence (Output_Steps, Features).
                          Default (12, 21).
    :type weather_shape: tuple
    :param output_shape: Shape of the output sequence (Time, H, W, C).
                         Default (12, 128, 128, 4).
    :type output_shape: tuple
    :return: A Keras Model instance.
    :rtype: keras.models.Model
    """
    
    # --- Inputs ---
    # Image Sequence: 4 Spectral Bands
    img_input = Input(shape=input_shape, name='sentinel2_sequence')
    # Cloudmask: Single band where 1 means cloudy and 0 means clear
    cloudmask_input = Input(shape=cloudmask_shape, name='cloudmask_sequence')
    # Landcover classification: One hot encoding for different classes
    landcover_input = Input(shape=landcover_shape, name='landcover_map')
    # Temporal data: Year, Cosine Day of Year, Sine Day of Year
    temporal_input = Input(shape=temporal_shape, name='temporal_metadata')
    # Weather data: E-OBS climate variables for each output step
    weather_input = Input(shape=weather_shape, name='weather_sequence')
    
    # Tile Landcover: (batch, H, W, 10) -> (batch, frames, H, W, 10)
    # Tile Landcover: (batch, H, W, 10) -> (batch, frames, H, W, 10)
    # 1. Add time dimension: (batch, 1, H, W, 10)
    target_shape = (1,) + landcover_shape # (1, 128, 128, 10)
    landcover_reshaped = layers.Reshape(target_shape, name='lc_reshape')(landcover_input)
    
    # 2. Tile along time dimension using UpSampling3D
    # UpSampling3D expects (batch, D, H, W, C). Here D is Time.
    # We want to repeat 'input_shape[0]' times along the time axis.
    time_steps = input_shape[0]
    landcover_tiled = layers.UpSampling3D(size=(time_steps, 1, 1), name='lc_tile')(landcover_reshaped)

    x = layers.Concatenate(axis=-1)([img_input, cloudmask_input, landcover_tiled])

    # --- Encoder (ConvLSTM) ---
    # Extract spatiotemporal features from past 18 frames
    # Reduced dropout (15%/10%) to prevent underfitting on small datasets
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', dropout=0.15, recurrent_dropout=0.1, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    
    # Additional ConvLSTM layer for deeper feature extraction
    x = layers.ConvLSTM2D(filters=48, kernel_size=(3, 3), padding='same', dropout=0.15, recurrent_dropout=0.1, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    
    # Reduce temporal dimension to a single hidden state volume
    # This 'context volume' summarizes the movement/vegetation trends of the past
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', dropout=0.15, recurrent_dropout=0.1, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    # Shape: (Batch, H, W, 64)

    # --- Multi-modal Fusion ---
    # 1. Embed temporal scalar vector
    temporal_emb = layers.Dense(16, activation='relu')(temporal_input)
    
    # 2. Embed weather features (reduce dimensionality)
    # Weather: (Batch, output_steps, 21) -> (Batch, output_steps, 16)
    weather_emb = layers.Dense(16, activation='relu', name='weather_embed')(weather_input)
    # Take mean across output steps to get a single vector for fusion with encoder
    # (Batch, output_steps, 16) -> (Batch, 16)
    weather_emb_mean = layers.GlobalAveragePooling1D(name='weather_mean')(weather_emb)
    
    # 3. Broadcast Strategy: Spatially tile the metadata to match image dimensions
    # Fuse the Encoder context with the Temporal metadata
    # Result Shape: (Batch, H, W, 64 + 16)
    context_fused = BroadcastFusionLayer(name='fusion_temporal')([x, temporal_emb])
    
    # Fuse with Weather embedding
    # Result Shape: (Batch, H, W, 64 + 16 + 16) = (Batch, H, W, 96)
    context_fused = BroadcastFusionLayer(name='fusion_weather')([context_fused, weather_emb_mean])
   
    # --- Prediction Head (B-Spline) ---
    # Using B-Spline for smooth temporal trajectory prediction.
    
    # Feature extraction with deeper prediction head
    # context_fused is (Batch, H, W, 96)
    
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(context_fused)
    x = layers.BatchNormalization()(x)
    
    # Additional Conv2D layer for better feature extraction
    x = layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    n_bands = output_shape[-1]
    n_ctrl_pts = 8  # Increased from 6 for more temporal flexibility
    
    # Predict B-Spline Control Points
    # Linear activation allows unbounded values (necessary for correct magnitude)
    coeffs = layers.Conv2D(filters=n_bands * n_ctrl_pts, kernel_size=(1, 1), name='coeff_prediction',
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    
    # Evaluate B-Spline
    outputs = BSplineBasisLayer(
        output_steps=output_shape[0], 
        n_bands=n_bands, 
        n_control_points=n_ctrl_pts,
        name='bspline_eval'
    )(coeffs) # (Batch, 12, H, W, 4)

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