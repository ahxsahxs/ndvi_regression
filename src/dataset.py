import os
import numpy as np
import glob
import xarray as xr
import pandas as pd
from pathlib import Path
import tensorflow as tf

def percentile_contrast(data: np.ndarray, p_min:int, p_max:int):
    val_min, val_max = np.nanpercentile(data, [p_min, p_max])
    return (data - val_min) / (val_max - val_min)

def scale_to_0_255(data: np.ndarray):
    val_min = np.nanmin(data)
    val_max = np.nanmax(data)
    return ((data - val_min) / (val_max - val_min)*255).astype(np.int32)

class DatasetGenerator:
    S2_BANDS = ['s2_B02', 's2_B03', 's2_B04', 's2_B8A']
    S2_MASK = 's2_mask'
    S2_AVAIL = 's2_avail'
    ESA_CLASSES = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])

    def __init__(
            self,
            data_dir: str | Path,
            input_days: int = 50,
            target_days: int = 100
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
        sentinel2 = percentile_contrast(sentinel2, 1, 99)

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
        
        return {
            "time": time_feats,
            "sentinel2": sentinel2,
            "cloudmask": s2_mask,
            "landcover": esawc_lc
        }
    

    def prepare_y(self, last_img: np.ndarray, dataset: xr.Dataset):
        # --- Inputs (First 50 days, starting at index 4) ---
        target_slice = slice(self.input_days+4, self.input_days+self.target_days, 5)
        dataset[self.S2_BANDS]
        sentinel2, s2_mask = self.prepare_sentinel_for_slice(dataset, target_slice)
        
        sentinel2_deltas = sentinel2.copy()
        for i in range(sentinel2.shape[0]):
            sentinel2_deltas[i, :, :, :] -= last_img

        return {
            "sentinel2": sentinel2,
            "cloudmask": s2_mask,
            "deltas": sentinel2_deltas
        }
        
    def __call__(self):
        for file_path in self.files:
            try:
                with xr.open_dataset(file_path) as ds:
                    # Check if we have enough time steps
                    if len(ds.time) < self.input_days + self.target_days:
                        print(f"File {file_path} does not contains enough days")
                        continue

                    x = self.prepare_x(dataset=ds)
                    
                    last_img = x["sentinel2"][-1]
                    y = self.prepare_y(last_img=last_img, dataset=ds)

                    yield x, y

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                print(f"Error processing {file_path}: {e}")
                continue

    def get_dataset(self) -> tf.data.Dataset:
        actual_input_days = int(self.input_days / 5)
        actual_target_days = int(self.target_days / 5)
        # Define output signature
        output_signature = (
            {
                'time': tf.TensorSpec(shape=(actual_input_days, 3), dtype=tf.float32),
                'sentinel2': tf.TensorSpec(shape=(actual_input_days, 128, 128, 4), dtype=tf.float32),
                'cloudmask': tf.TensorSpec(shape=(actual_input_days, 128, 128, 1), dtype=tf.float32),
                'cloudmask': tf.TensorSpec(shape=(actual_input_days, 128, 128, 1), dtype=tf.float32),
                'landcover': tf.TensorSpec(shape=(128, 128, 10), dtype=tf.float32),
            },
            {
                "sentinel2": tf.TensorSpec(shape=(actual_target_days, 128, 128, 4), dtype=tf.float32),
                "cloudmask": tf.TensorSpec(shape=(actual_target_days, 128, 128, 1), dtype=tf.float32),
                "deltas": tf.TensorSpec(shape=(actual_target_days, 128, 128, 4), dtype=tf.float32),
            }
        )
        
        return tf.data.Dataset.from_generator(
            self.__call__,
            output_signature=output_signature
        )


if __name__ == "__main__":
    DATASET_PATH = Path("/home/me/workspace/probformer/data/greenearthnet/train_test")

    generator = DatasetGenerator(DATASET_PATH)
    dataset = generator.get_dataset()

    for x_dict, y_dict in dataset.take(5):
        x_landcover: np.ndarray = x_dict["landcover"].numpy()
        x_time: np.ndarray = x_dict["time"].numpy()
        x_sentinel2: np.ndarray = x_dict["sentinel2"].numpy()
        x_cloudmask: np.ndarray = x_dict["cloudmask"].numpy()
        
        y_sentinel2: np.ndarray = y_dict["sentinel2"].numpy()
        y_cloudmask: np.ndarray = y_dict["cloudmask"].numpy()

        print("X Shape:", x_sentinel2.shape)
        print("Time Shape:", x_time.shape)
        print("Landcover Shape:", x_landcover.shape)
        print("Y Shape:", y_sentinel2.shape)

