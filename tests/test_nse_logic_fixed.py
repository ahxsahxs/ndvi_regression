
import unittest
import numpy as np

LANDCOVER_CLASSES = [
    'Tree cover', 'Shrubland', 'Grassland', 'Cropland', 
    'Built-up', 'Bare/sparse', 'Snow/ice', 'Water',
    'Wetland', 'Mangroves', 'Moss/lichen'
]

def calculate_vegetation_score_stats_fixed(true_kndvi, pred_kndvi, mask, landcover_flat):
    """
    Fixed/Refined logic for Vegetation Score statistics.
    """
    T, N = true_kndvi.shape
    
    lc_nnse = {lc: [] for lc in LANDCOVER_CLASSES}
    
    # Numerator: sum((obs - pred)^2)
    diff_sq = (true_kndvi - pred_kndvi) ** 2
    valid = (mask == 0)
    
    valid_counts = np.sum(valid, axis=0) # (N,)
    valid_pixels_mask = valid_counts > 2
    
    # Denominator: sum((obs - mean)^2)
    obs_sum = np.sum(true_kndvi * valid, axis=0)
    obs_mean = np.divide(obs_sum, valid_counts, out=np.zeros_like(obs_sum), where=valid_counts>0)
    term2 = (true_kndvi - obs_mean[np.newaxis, :]) ** 2
    
    numerator = np.sum(diff_sq * valid, axis=0)
    denominator = np.sum(term2 * valid, axis=0)
    
    # --- New Logic ---
    # Initialize NSE with "Infinite Error" value (will result in NNSE ~ 0)
    nse = np.full(N, -9999.0, dtype=np.float32)
    
    # Case 1: Normal Variance
    # We use a small epsilon for variance stability
    var_epsilon = 1e-6
    mask_normal = denominator > var_epsilon
    
    # standard nse formula for normal pixels
    # nse = 1 - num/den
    nse[mask_normal] = 1.0 - (numerator[mask_normal] / denominator[mask_normal])
    
    # Case 2: Zero/Low Variance
    # If variance is zero, NSE is undefined usually. 
    # But if Pred matches Obs (Numerator is also zero/small), then it's a Perfect Prediction.
    mask_low_var = ~mask_normal
    
    # Threshold for "Perfect Match" in numerator
    # If sum_sq_diff is very small, we consider it perfect.
    num_epsilon = 1e-6
    mask_perfect_match = (numerator < num_epsilon)
    
    # Pixels that are Low Variance AND Perfect Match get Max Score
    mask_low_var_perfect = mask_low_var & mask_perfect_match
    nse[mask_low_var_perfect] = 1.0
    
    # Pixels that are Low Variance but BAD Match (numerator large) stay at -9999 (Equivalent to -Inf)
    
    # Compute NNSE
    # NNSE = 1 / (2 - NSE)
    # Clip NSE to reasonable range [-10, 1]
    # -9999 becomes -10 -> NNSE = 1/12 ~ 0.08
    nse_clipped = np.clip(nse, -10, 1)
    nnse = 1 / (2 - nse_clipped)
    
    # Aggregate
    for i, lc_name in enumerate(LANDCOVER_CLASSES):
        indices = np.where((landcover_flat == i) & valid_pixels_mask)[0]
        if len(indices) > 0:
            lc_nnse[lc_name].extend(nnse[indices])
            
    return lc_nnse

class TestNSELogicFixed(unittest.TestCase):
    def setUp(self):
        self.lc_classes = LANDCOVER_CLASSES
        self.lc_flat = np.zeros(100, dtype=int) 
        
    def test_perfect_prediction_constant_signal(self):
        """Test case where signal is constant and prediction is perfect."""
        true_kndvi = np.full((10, 100), 0.5)
        pred_kndvi = np.full((10, 100), 0.5)
        mask = np.zeros((10, 100))
        
        stats = calculate_vegetation_score_stats_fixed(true_kndvi, pred_kndvi, mask, self.lc_flat)
        mean_nnse = np.mean(stats['Tree cover'])
        print(f"Constant Signal (Perfect) -> Mean NNSE: {mean_nnse}")
        self.assertAlmostEqual(mean_nnse, 1.0, places=4)

    def test_bad_prediction_constant_signal(self):
        """Test case where signal is constant but prediction is wrong. NOW should be 0."""
        true_kndvi = np.full((10, 100), 0.5)
        pred_kndvi = np.full((10, 100), 0.6)
        mask = np.zeros((10, 100))
        
        stats = calculate_vegetation_score_stats_fixed(true_kndvi, pred_kndvi, mask, self.lc_flat)
        mean_nnse = np.mean(stats['Tree cover'])
        print(f"Constant Signal (Bad) -> Mean NNSE: {mean_nnse}")
        
        # 1 / (2 - (-10)) = 1/12 = 0.0833
        expected = 1.0 / 12.0 
        self.assertAlmostEqual(mean_nnse, expected, places=4)
        
    def test_low_variance_signal_bad(self):
        """Test case with low variance and bad prediction. Should be low."""
        true_kndvi = np.full((10, 100), 0.5)
        true_kndvi[0, :] += 0.002 # Slightly > epsilon
        pred_kndvi = np.full((10, 100), 0.6) 
        mask = np.zeros((10, 100))
        
        stats = calculate_vegetation_score_stats_fixed(true_kndvi, pred_kndvi, mask, self.lc_flat)
        mean_nnse = np.mean(stats['Tree cover'])
        print(f"Low Var Signal (Bad) -> Mean NNSE: {mean_nnse}")
        self.assertLess(mean_nnse, 0.15)
        
    def test_low_variance_signal_good(self):
        """Test case with low variance but Good (perfect) prediction. Should be 1."""
        true_kndvi = np.full((10, 100), 0.5)
        true_kndvi[0, :] += 0.002
        pred_kndvi = true_kndvi.copy()
        mask = np.zeros((10, 100))
        
        stats = calculate_vegetation_score_stats_fixed(true_kndvi, pred_kndvi, mask, self.lc_flat)
        mean_nnse = np.mean(stats['Tree cover'])
        print(f"Low Var Signal (Good) -> Mean NNSE: {mean_nnse}")
        self.assertAlmostEqual(mean_nnse, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
