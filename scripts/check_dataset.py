import numpy as np
from src.dataset import DatasetGenerator
from src.config import DATASET_PATH, VALIDATION_PATH

def check_dataset(path, name):
    print(f"Checking {name} dataset at {path}")
    generator = DatasetGenerator(path)
    dataset = generator.get_dataset()
    
    mins, maxs = [], []
    for x, y in dataset.take(50):
        s2 = x["sentinel2"].numpy()
        mins.append(s2.min())
        maxs.append(s2.max())
    
    print(f"{name} Sentinel-2 Reflectance: Min = {np.min(mins):.2f}, Max = {np.max(maxs):.2f}")

if __name__ == "__main__":
    check_dataset(DATASET_PATH, "Train")
    check_dataset(VALIDATION_PATH, "Validation")
