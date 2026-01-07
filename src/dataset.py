import os
import numpy as np
import glob
import xarray as xr
import pandas as pd
from pathlib import Path
import tensorflow as tf

def percentile_contrast(data: np.ndarray, p_min:int, p_max:int):
    val_min, val_max = np.nanpercentile(data, [p_min, p_max])
    clipped_data = np.clip(data, val_min, val_max)
    return (clipped_data - val_min) / (val_max - val_min)

def scale_to_01(data: np.ndarray):
    val_min = np.nanmin(data)
    val_max = np.nanmax(data)
    return (data - val_min) / (val_max - val_min)

class DatasetGenerator:
    S2_BANDS = ['s2_B02', 's2_B03', 's2_B04', 's2_B8A']
    S2_MASK = 's2_mask'
    S2_AVAIL = 's2_avail'
    ESA_CLASSES = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    # E-OBS weather variables (excluding eobs_fg which contains NaN)
    EOBS_VARS = ['eobs_tg', 'eobs_hu', 'eobs_pp', 'eobs_qq', 'eobs_rr', 'eobs_tn', 'eobs_tx']
    # Normalization ranges for each E-OBS variable (min, max)
    EOBS_NORM = {
        'eobs_tg': (-20, 45),   # Mean temperature (°C)
        'eobs_hu': (0, 100),    # Relative humidity (%)
        'eobs_pp': (950, 1050), # Sea level pressure (hPa)
        'eobs_qq': (0, 400),    # Global radiation (W/m²)
        'eobs_rr': (0, 50),     # Precipitation (mm)
        'eobs_tn': (-30, 35),   # Min temperature (°C)
        'eobs_tx': (-10, 50),   # Max temperature (°C)
    }

    def __init__(
            self,
            data_dir: str | Path,
            input_days: int = 90,
            target_days: int = 60
        ) -> None:
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.nc"), recursive=True))
        
        if not self.files:
            raise ValueError(f"The data_dir {data_dir} does not contains any .nc file")

        self.input_days = input_days
        self.target_days = target_days

    def prepare_sentinel_for_slice(self, dataset: xr.Dataset, time_slice:slice)->tuple[np.ndarray, np.ndarray]:
        # 1. Sentinel-2 Bands
        s2_data = []
        for band in self.S2_BANDS:
            b_data = dataset[band].isel(time=time_slice).values
            s2_data.append(b_data)
        
        # (10, 128, 128, 4)
        sentinel2 = np.stack(s2_data, axis=-1)
        # sentinel2 = scale_to_01(sentinel2)

        # If any band is NaN, the pixel is missing/invalid
        s2_nans = np.isnan(sentinel2).any(axis=-1, keepdims=True) # (10, 128, 128, 1)

        # 2. Cloud Mask
        # mask > 0 means cloud/shadow etc.
        mask = dataset[self.S2_MASK].isel(time=time_slice).values
        s2_mask = (mask > 0).astype(np.float32)
        s2_mask = np.expand_dims(s2_mask, axis=-1) # (10, 128, 128, 1)

        # Update mask to include NaNs (missing data)
        # If s2_nans is True, s2_mask should be 1 (masked)
        s2_mask = np.maximum(s2_mask, s2_nans.astype(np.float32))

        # Replace NaNs with 0
        sentinel2 = np.nan_to_num(sentinel2, nan=0)

        return sentinel2, s2_mask

    def prepare_x(self, dataset: xr.Dataset) -> dict:
        # --- Inputs (First 50 days, starting at index 4) ---
        input_slice = slice(4, self.input_days, 5)
        
        sentinel2, s2_mask = self.prepare_sentinel_for_slice(dataset, input_slice)

        # 6. Landcover (ESA WorldCover)
        esawc_lc = dataset['esawc_lc'].values # (128, 128)
        
        # One-hot encoding
        lc_indices = np.searchsorted(self.ESA_CLASSES, esawc_lc)
        lc_indices = np.clip(lc_indices, 0, len(self.ESA_CLASSES) - 1)
        lc_one_hot = np.eye(len(self.ESA_CLASSES))[lc_indices] # (128, 128, 11)
        
        # Drop the first class to get n_classes-1
        esawc_lc = lc_one_hot[..., 1:].astype(np.float32) # (128, 128, 10)
        
        # 7. Time
        times = dataset.time.isel(time=input_slice).values
        ts = pd.to_datetime(times)
        
        # Normalize year (2017-2021)
        year_norm = (ts.year - 2017) / (2021 - 2017)
        
        # Cyclic day of year
        days_in_year = np.where(ts.is_leap_year, 366, 365)
        doy_rad = 2 * np.pi * ts.day_of_year / days_in_year
        sin_doy = np.sin(doy_rad)
        cos_doy = np.cos(doy_rad)
        
        time_feats = np.stack([year_norm, sin_doy, cos_doy], axis=-1).astype(np.float32) 
        
        # Prepare weather features
        weather_feats = self.prepare_weather(dataset)
        
        return {
            "time": time_feats,
            "sentinel2": sentinel2,
            "cloudmask": s2_mask,
            "landcover": esawc_lc,
            "weather": weather_feats
        }
    
    def prepare_weather(self, dataset: xr.Dataset) -> np.ndarray:
        """
        Prepares weather features from E-OBS variables using climatology-based aggregation.
        
        For each 5-day forecast step, computes 3 aggregations per variable:
        - min_detrend: Minimum anomaly (value - climatology) over 5 days
        - max_detrend: Maximum anomaly over 5 days
        - mean_clima: Mean climatology value over 5 days
        
        Returns:
            weather_feats: (target_steps, 21) where 21 = 7 vars × 3 aggregations
        """
        target_days = self.target_days
        n_output_steps = target_days // 5
        n_vars = len(self.EOBS_VARS)
        n_aggs = 3  # min_detrend, max_detrend, mean_clima
        
        # Rolling window size for climatology (21-day centered window)
        clima_window = 21
        
        weather_feats = np.zeros((n_output_steps, n_vars * n_aggs), dtype=np.float32)
        
        for var_idx, var_name in enumerate(self.EOBS_VARS):
            # Get all daily values
            daily_values = dataset[var_name].values  # (150,)
            
            # Handle NaN values with forward/backward fill
            if np.isnan(daily_values).any():
                valid_mask = ~np.isnan(daily_values)
                if valid_mask.sum() > 0:
                    # Interpolate NaN values
                    daily_values = np.interp(
                        np.arange(len(daily_values)),
                        np.where(valid_mask)[0],
                        daily_values[valid_mask]
                    )
                else:
                    # All NaN - fill with 0
                    daily_values = np.zeros_like(daily_values)
            
            # Compute rolling climatology (21-day centered window mean)
            climatology = np.zeros_like(daily_values)
            half_window = clima_window // 2
            for i in range(len(daily_values)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(daily_values), i + half_window + 1)
                climatology[i] = np.mean(daily_values[start_idx:end_idx])
            
            # Compute anomalies (detrended values)
            anomalies = daily_values - climatology
            
            # Get normalization range for this variable
            norm_min, norm_max = self.EOBS_NORM[var_name]
            
            # For each 5-day output step, compute aggregations
            for step in range(n_output_steps):
                # Target window: days [input_days + step*5, input_days + (step+1)*5)
                start_day = self.input_days + step * 5
                end_day = start_day + 5
                
                if end_day > len(daily_values):
                    break
                
                window_anomalies = anomalies[start_day:end_day]
                window_climatology = climatology[start_day:end_day]
                
                # Compute aggregations
                min_detrend = np.min(window_anomalies)
                max_detrend = np.max(window_anomalies)
                mean_clima = np.mean(window_climatology)
                
                # Normalize: anomalies to [-1, 1] range, climatology to [0, 1]
                range_val = norm_max - norm_min
                min_detrend_norm = np.clip(min_detrend / (range_val / 2), -1, 1)
                max_detrend_norm = np.clip(max_detrend / (range_val / 2), -1, 1)
                mean_clima_norm = np.clip((mean_clima - norm_min) / range_val, 0, 1)
                
                # Store in output: [var0_min, var0_max, var0_mean, var1_min, ...]
                base_idx = var_idx * n_aggs
                weather_feats[step, base_idx] = min_detrend_norm
                weather_feats[step, base_idx + 1] = max_detrend_norm
                weather_feats[step, base_idx + 2] = mean_clima_norm
        
        return weather_feats


    def compute_bap(self, sentinel2: np.ndarray, cloudmask: np.ndarray) -> np.ndarray:
        """
        Computes the Best Available Pixel (BAP) composite from the input sequence.
        Iterates backwards from the last image. If a pixel is cloudy/missing in the
        current composite but clear in the previous image, it is updated.
        """
        # Initialize BAP with the last image and its mask
        bap = sentinel2[-1].copy()
        bap_mask = cloudmask[-1].copy()

        # Iterate backwards from the second to last image
        for i in range(sentinel2.shape[0] - 2, -1, -1):
            # Identify pixels that are currently masked in BAP but clear in the current frame
            # bap_mask == 1 (bad), cloudmask[i] == 0 (good)
            update_mask = (bap_mask == 1) & (cloudmask[i] == 0)
            
            # Update BAP where we found better pixels
            bap = np.where(update_mask, sentinel2[i], bap)
            
            # Update the mask: if we found a good pixel, it's no longer masked
            bap_mask = np.where(update_mask, 0, bap_mask)
            
            # If no more masked pixels, we can stop early
            if np.sum(bap_mask) == 0:
                break
                
        return bap

    def prepare_y(self, last_img: np.ndarray, dataset: xr.Dataset):
        target_slice = slice(self.input_days+4, self.input_days+self.target_days, 5)
        # dataset[self.S2_BANDS] # Unused access
        sentinel2, s2_mask = self.prepare_sentinel_for_slice(dataset, target_slice)
        
        sentinel2_deltas = sentinel2.copy()
        
        # Tile last_img (BAP) to match time dimension of target
        # last_img: (128, 128, 4)
        # sentinel2: (T, 128, 128, 4)
        bap_tiled = np.tile(
            np.expand_dims(last_img, 0), 
            (sentinel2.shape[0], 1, 1, 1)
        ) # (T, 128, 128, 4)

        sentinel2_deltas = sentinel2 - bap_tiled

        # Output: [Mask (1), Deltas (4), BAP (4)] -> Total 9 channels
        return np.concatenate([s2_mask, sentinel2_deltas, bap_tiled], axis=-1)
        
    def __call__(self):
        for file_path in self.files:
            try:
                with xr.open_dataset(file_path) as ds:
                    # Check if we have enough time steps
                    if len(ds.time) < self.input_days + self.target_days:
                        # print(f"File {file_path} does not contains enough days")
                        continue

                    x = self.prepare_x(dataset=ds)
                    
                    # Compute Best Available Pixel (BAP) composite for the last image
                    last_img = self.compute_bap(x["sentinel2"], x["cloudmask"])
                    y = self.prepare_y(last_img=last_img, dataset=ds)

                    yield x, y

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    def get_dataset(self) -> tf.data.Dataset:
        actual_input_days = int(self.input_days / 5)
        actual_target_days = int(self.target_days / 5)
        # Define output signature
        # Weather: 7 vars × 3 aggregations = 21 features
        n_weather_features = len(self.EOBS_VARS) * 3
        output_signature = (
            {
                'time': tf.TensorSpec(shape=(actual_input_days, 3), dtype=tf.float32),
                'sentinel2': tf.TensorSpec(shape=(actual_input_days, 128, 128, 4), dtype=tf.float32),
                'cloudmask': tf.TensorSpec(shape=(actual_input_days, 128, 128, 1), dtype=tf.float32),
                'landcover': tf.TensorSpec(shape=(128, 128, 10), dtype=tf.float32),
                'weather': tf.TensorSpec(shape=(actual_target_days, n_weather_features), dtype=tf.float32),
            },
            # y: [Mask (1), Deltas (4), BAP (4)] = 9 channels
            tf.TensorSpec(shape=(actual_target_days, 128, 128, 9), dtype=tf.float32)
        )
        
        return tf.data.Dataset.from_generator(
            self.__call__,
            output_signature=output_signature
        )


if __name__ == "__main__":
    DATASET_PATH = Path("/home/me/workspace/probformer/data/greenearthnet/sub_train")

    generator = DatasetGenerator(DATASET_PATH)
    dataset = generator.get_dataset()

    for x_dict, y_tensor in dataset.take(5):
        x_landcover: np.ndarray = x_dict["landcover"].numpy()
        x_time: np.ndarray = x_dict["time"].numpy()
        x_sentinel2: np.ndarray = x_dict["sentinel2"].numpy()
        x_cloudmask: np.ndarray = x_dict["cloudmask"].numpy()
        x_weather: np.ndarray = x_dict["weather"].numpy()
        
        y_out: np.ndarray = y_tensor.numpy()

        print("X Sentinel2 Shape:", x_sentinel2.shape)
        print("Time Shape:", x_time.shape)
        print("Landcover Shape:", x_landcover.shape)
        print("Weather Shape:", x_weather.shape)
        print("Weather sample (step 0):", x_weather[0, :6])  # First 2 variables
        print("Y Shape:", y_out.shape)
        print("-" * 40)
