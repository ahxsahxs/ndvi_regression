import os
import numpy as np
import glob
import xarray as xr
import pandas as pd
from pathlib import Path
import tensorflow as tf

def percentile_contrast(data: np.ndarray, p_min: int, p_max: int) -> np.ndarray:
    """
    Applies percentile-based contrast stretching to data.

    :param data: Input array to normalize.
    :type data: np.ndarray
    :param p_min: Lower percentile for clipping.
    :type p_min: int
    :param p_max: Upper percentile for clipping.
    :type p_max: int
    :return: Normalized array scaled to [0, 1].
    :rtype: np.ndarray
    """
    val_min, val_max = np.nanpercentile(data, [p_min, p_max])
    clipped_data = np.clip(data, val_min, val_max)
    return (clipped_data - val_min) / (val_max - val_min)


def scale_to_01(data: np.ndarray) -> np.ndarray:
    """
    Scales data to [0, 1] range using min-max normalization.

    :param data: Input array to normalize.
    :type data: np.ndarray
    :return: Normalized array scaled to [0, 1].
    :rtype: np.ndarray
    """
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
            input_days: int = 50,
            target_days: int = 100
        ) -> None:
        """
        Initializes the DatasetGenerator.

        :param data_dir: Directory containing GreenEarthNet .nc files.
        :type data_dir: str | Path
        :param input_days: Number of input days (sampled every 5 days).
        :type input_days: int
        :param target_days: Number of target days to predict.
        :type target_days: int
        :raises ValueError: If no .nc files are found in data_dir.
        """
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.nc"), recursive=True))
        
        if not self.files:
            raise ValueError(f"The data_dir {data_dir} does not contains any .nc file")

        self.input_days = input_days
        self.target_days = target_days

    def prepare_sentinel_for_slice(
            self, dataset: xr.Dataset, time_slice: slice
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts Sentinel-2 bands and cloud mask for a time slice.

        :param dataset: xarray Dataset containing satellite data.
        :type dataset: xr.Dataset
        :param time_slice: Slice object specifying time range.
        :type time_slice: slice
        :return: Tuple of (sentinel2, cloudmask) arrays.
                 sentinel2: (T, H, W, 4) reflectance values.
                 cloudmask: (T, H, W, 1) where 1=cloudy/invalid.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        s2_data = []
        for band in self.S2_BANDS:
            b_data = dataset[band].isel(time=time_slice).values
            s2_data.append(b_data)
        
        sentinel2 = np.stack(s2_data, axis=-1)  # (T, 128, 128, 4)
        s2_nans = np.isnan(sentinel2).any(axis=-1, keepdims=True)

        # Cloud mask: mask > 0 means cloud/shadow
        mask = dataset[self.S2_MASK].isel(time=time_slice).values
        s2_mask = (mask > 0).astype(np.float32)
        s2_mask = np.expand_dims(s2_mask, axis=-1)

        # Include NaNs in mask
        s2_mask = np.maximum(s2_mask, s2_nans.astype(np.float32))
        sentinel2 = np.nan_to_num(sentinel2, nan=0)

        return sentinel2, s2_mask

    def prepare_x(self, dataset: xr.Dataset) -> dict:
        """
        Prepares all input features for the model.

        :param dataset: xarray Dataset containing satellite and ancillary data.
        :type dataset: xr.Dataset
        :return: Dictionary with keys: 'time', 'sentinel2', 'cloudmask',
                 'landcover', 'weather'.
        :rtype: dict
        """
        input_slice = slice(4, self.input_days, 5)
        sentinel2, s2_mask = self.prepare_sentinel_for_slice(dataset, input_slice)
        
        # Fill cloudy pixels with last clear observation
        sentinel2_bap = self.compute_bap_sequence(sentinel2, s2_mask)

        # Landcover one-hot encoding
        esawc_lc = dataset['esawc_lc'].values
        lc_indices = np.searchsorted(self.ESA_CLASSES, esawc_lc)
        lc_indices = np.clip(lc_indices, 0, len(self.ESA_CLASSES) - 1)
        lc_one_hot = np.eye(len(self.ESA_CLASSES))[lc_indices]
        esawc_lc = lc_one_hot[..., 1:].astype(np.float32)  # (128, 128, 10)
        
        # Temporal features
        times = dataset.time.isel(time=input_slice).values
        ts = pd.to_datetime(times)
        year_norm = (ts.year - 2017) / (2021 - 2017)
        days_in_year = np.where(ts.is_leap_year, 366, 365)
        doy_rad = 2 * np.pi * ts.day_of_year / days_in_year
        time_feats = np.stack([year_norm, np.sin(doy_rad), np.cos(doy_rad)], axis=-1).astype(np.float32)
        
        weather_feats = self.prepare_weather(dataset)
        
        return {
            "time": time_feats,
            "sentinel2": sentinel2_bap,
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

        :param dataset: xarray Dataset containing E-OBS weather variables.
        :type dataset: xr.Dataset
        :return: Weather features array (target_steps, 21) where 21 = 7 vars × 3 aggs.
        :rtype: np.ndarray
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


    def compute_bap_sequence(self, sentinel2: np.ndarray, cloudmask: np.ndarray) -> np.ndarray:
        """
        Computes a BAP sequence where each frame has cloudy pixels replaced.
        
        For each frame t, iterates backwards through previous frames to find
        clear pixels. Cloudy/missing pixels are replaced with the most recent
        clear observation. This provides realistic spectral values for the model
        instead of zeros in cloudy regions.
        
        :param sentinel2: Input sequence (T, H, W, C).
        :type sentinel2: np.ndarray
        :param cloudmask: Cloud mask sequence (T, H, W, 1) where 1=cloudy.
        :type cloudmask: np.ndarray
        :return: BAP-filled sequence with same shape as input.
        :rtype: np.ndarray
        """
        bap_sequence = sentinel2.copy()
        
        for t in range(1, sentinel2.shape[0]):  # Start from frame 1
            # Current frame's cloud mask
            current_mask = cloudmask[t]  # (H, W, 1)
            
            # Look backwards for clear pixels to fill cloudy regions
            for prev_t in range(t - 1, -1, -1):
                prev_mask = cloudmask[prev_t]
                
                # Update where current frame is cloudy but previous is clear
                update_mask = (current_mask == 1) & (prev_mask == 0)
                
                bap_sequence[t] = np.where(update_mask, sentinel2[prev_t], bap_sequence[t])
                
                # Update the effective mask for this frame (pixels now filled)
                current_mask = np.where(update_mask, 0, current_mask)
                
                # Early exit if all pixels are now clear
                if np.sum(current_mask) == 0:
                    break
        
        return bap_sequence

    def compute_bap(self, sentinel2: np.ndarray, cloudmask: np.ndarray) -> np.ndarray:
        """
        Computes the Best Available Pixel (BAP) composite for the last frame.

        Returns the last frame of the BAP sequence, where cloudy pixels are
        filled with the most recent clear observation from previous frames.

        :param sentinel2: Input sequence (T, H, W, C).
        :type sentinel2: np.ndarray
        :param cloudmask: Cloud mask sequence (T, H, W, 1) where 1=cloudy.
        :type cloudmask: np.ndarray
        :return: BAP composite for the last frame (H, W, C).
        :rtype: np.ndarray
        """
        # Reuse compute_bap_sequence and return the last frame
        bap_sequence = self.compute_bap_sequence(sentinel2, cloudmask)
        return bap_sequence[-1]

    def prepare_y(self, last_img: np.ndarray, dataset: xr.Dataset) -> np.ndarray:
        """
        Prepares target output with mask, deltas, and BAP reference.

        :param last_img: BAP composite from the last input frame (H, W, C).
        :type last_img: np.ndarray
        :param dataset: xarray Dataset containing target satellite data.
        :type dataset: xr.Dataset
        :return: Target array (T, H, W, 9) with [mask(1), deltas(4), BAP(4)].
        :rtype: np.ndarray
        """
        target_slice = slice(self.input_days + 4, self.input_days + self.target_days, 5)
        sentinel2, s2_mask = self.prepare_sentinel_for_slice(dataset, target_slice)
        
        # Tile BAP to match time dimension
        bap_tiled = np.tile(np.expand_dims(last_img, 0), (sentinel2.shape[0], 1, 1, 1))
        sentinel2_deltas = sentinel2 - bap_tiled

        return np.concatenate([s2_mask, sentinel2_deltas, bap_tiled], axis=-1)
        
    def __call__(self):
        """
        Generator that yields (x, y) tuples for each valid sample.

        :yields: Tuple of (x_dict, y_array) for each sample.
        :ytype: tuple[dict, np.ndarray]
        """
        for file_path in self.files:
            try:
                with xr.open_dataset(file_path) as ds:
                    if len(ds.time) < self.input_days + self.target_days:
                        continue

                    x = self.prepare_x(dataset=ds)
                    last_img = self.compute_bap(x["sentinel2"], x["cloudmask"])
                    y = self.prepare_y(last_img=last_img, dataset=ds)

                    yield x, y

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    def get_dataset(self) -> tf.data.Dataset:
        """
        Creates a tf.data.Dataset from the generator.

        :return: TensorFlow Dataset yielding (x_dict, y_array) tuples.
        :rtype: tf.data.Dataset
        """
        actual_input_days = int(self.input_days / 5)
        actual_target_days = int(self.target_days / 5)
        n_weather_features = len(self.EOBS_VARS) * 3
        
        output_signature = (
            {
                'time': tf.TensorSpec(shape=(actual_input_days, 3), dtype=tf.float32),
                'sentinel2': tf.TensorSpec(shape=(actual_input_days, 128, 128, 4), dtype=tf.float32),
                'cloudmask': tf.TensorSpec(shape=(actual_input_days, 128, 128, 1), dtype=tf.float32),
                'landcover': tf.TensorSpec(shape=(128, 128, 10), dtype=tf.float32),
                'weather': tf.TensorSpec(shape=(actual_target_days, n_weather_features), dtype=tf.float32),
            },
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
