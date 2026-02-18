
import unittest
import numpy as np
from src.error_analysis import calculate_vegetation_score_stats, LANDCOVER_CLASSES

class TestNSELogic(unittest.TestCase):
    def setUp(self):
        self.lc_classes = LANDCOVER_CLASSES
        # Create dummy landcover map (all 0 -> Tree cover)
        self.lc_flat = np.zeros(100, dtype=int) 
        
    def test_perfect_prediction_constant_signal(self):
        """Test case where signal is constant and prediction is perfect."""
        # T=10, N=100
        # Constant signal 0.5
        true_kndvi = np.full((10, 100), 0.5)
        pred_kndvi = np.full((10, 100), 0.5)
        mask = np.zeros((10, 100)) # All valid
        
        # Calculate stats
        stats = calculate_vegetation_score_stats(true_kndvi, pred_kndvi, mask, self.lc_flat)
        
        # Should get NNSE = 1.0 for Tree cover (index 0)
        nnse_values = stats['Tree cover']
        mean_nnse = np.mean(nnse_values)
        
        print(f"\nConstant Signal (Perfect) -> Mean NNSE: {mean_nnse}")
        
        # Currently this likely fails or returns low value/NaN depending on implementation
        # We expect it to be 1.0 ideally
        self.assertAlmostEqual(mean_nnse, 1.0, places=4, msg="Perfect constant prediction should yield NNSE=1.0")

    def test_bad_prediction_constant_signal(self):
        """Test case where signal is constant but prediction is wrong."""
        true_kndvi = np.full((10, 100), 0.5)
        pred_kndvi = np.full((10, 100), 0.6) # Wrong by 0.1
        mask = np.zeros((10, 100))
        
        stats = calculate_vegetation_score_stats(true_kndvi, pred_kndvi, mask, self.lc_flat)
        nnse_values = stats['Tree cover']
        mean_nnse = np.mean(nnse_values)
        
        print(f"Constant Signal (Bad) -> Mean NNSE: {mean_nnse}")
        
        # Should be 0.0 (Worst possible score)
        self.assertAlmostEqual(mean_nnse, 0.0, places=4, msg="Bad constant prediction should yield NNSE=0.0")
        
    def test_standard_case(self):
        """Standard varying signal case."""
        # Sine wave signal
        t = np.linspace(0, 4*np.pi, 10)
        true_kndvi = np.tile(np.sin(t), (100, 1)).T # (10, 100)
        pred_kndvi = true_kndvi.copy() # Perfect
        mask = np.zeros((10, 100))
        
        stats = calculate_vegetation_score_stats(true_kndvi, pred_kndvi, mask, self.lc_flat)
        mean_nnse = np.mean(stats['Tree cover'])
        
        print(f"Varying Signal (Perfect) -> Mean NNSE: {mean_nnse}")
        self.assertAlmostEqual(mean_nnse, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
