import torch
import json
import os
import rasterio
import numpy as np
import config
import matplotlib.pyplot as plt
import csv  # Import the csv module
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds, get_marida_means_and_stds
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Callable # Added Callable for transform type hint
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images
from sklearn.cluster import KMeans

class HydroMoCoGeoDataset(Dataset):
    def __init__(
            self, path_dataset: Path, bands: List[str] = None,
            transforms: Optional[Callable] = None, compute_stats: bool = False, # Changed 'transforms' to type Callable
            model_name="mae", num_geo_clusters=100, save_csv=False, csv_path="geo_labels.csv"):
        self.path_dataset = Path(path_dataset)

        # This line is crucial: it should now find files directly in `self.path_dataset`
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))

        self.bands = bands
        if self.bands is None: # Corrected from `if self.bands == None:`
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        self.transforms = transforms

        self.model_name = model_name
        self.num_geo_clusters = num_geo_clusters
        self.save_csv = save_csv  # Store whether to save to CSV
        self.csv_path = csv_path  # Store the path to the CSV file
        
        # Ensure that NORM_PARAM_DEPTH and NORM_PARAM_PATHS are defined in config
        # and self.location is set if needed for these lookups.
        # Assuming a default location if not provided.
        self.location = "agia_napa" # Default location, can be passed as argument if needed
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        if self.model_name == "moco-geo":
            self.geo_to_label = self._create_geo_labels()

    def __len__(self):
        return len(self.file_paths)

    def _load_data(self):
        all_data = []
        for file_path in self.file_paths:
            try:
                with rasterio.open(file_path) as src:
                    row, col = src.height // 2, src.width // 2
                    lon, lat = src.transform * (col, row)
                    all_data.append((str(file_path), lat, lon))  # Store path and coordinates
            except rasterio.errors.RasterioIOError as e:
                print(f"Error opening {file_path}: {e}")
        return all_data

    def _create_geo_labels(self):
        geo_coords = np.array([[item[2], item[1]] for item in self._load_data()])  # [lon, lat]
        kmeans = KMeans(n_clusters=self.num_geo_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(geo_coords)
        geo_to_label_map = {}
        if self.save_csv:  # Only create and save CSV if save_csv is True
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['label', 'file_dir', 'lat', 'lon'])  # Write the header row
                for i, (path, lat, lon) in enumerate(self._load_data()):
                    label = cluster_assignments[i]
                    geo_to_label_map[path] = label
                    writer.writerow([label, path, lat, lon])  # Write data to CSV
        else:
            for i, (path, lat, lon) in enumerate(self._load_data()):
                geo_to_label_map[path] = cluster_assignments[i]
        return geo_to_label_map

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        try:
            with rasterio.open(file_path) as src:
                bands_data = [] # Changed from 'bands' to 'bands_data' to avoid conflict with self.bands
                for i in range(1, len(self.bands) + 1):
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands_data.append(band_tensor)
                sample = torch.cat(bands_data, dim=0).contiguous()

                # Pre-normalization and NaN handling
                if len(self.bands) == 11:
                    nan_mask = torch.isnan(sample) # Using torch.isnan directly
                    means, stds, _ = get_marida_means_and_stds()
                    # Reshape means for tiling correctly with 256x256 image
                    impute_nan_val = means.astype(np.float32) # Ensure float32
                    impute_nan_tensor = torch.from_numpy(impute_nan_val).unsqueeze(-1).unsqueeze(-1) # (C, 1, 1)
                    # Expand to (C, H, W) and use it for imputation
                    impute_nan_expanded = impute_nan_tensor.expand_as(sample)
                    sample[nan_mask] = impute_nan_expanded[nan_mask]

                elif len(self.bands) == 3:
                    # No specific NaN handling or imputation mentioned for 3 bands, assuming data is clean
                    pass # Placeholder if no specific pre-processing for 3 bands is needed

                else:
                    # No specific NaN handling or imputation mentioned for other band counts
                    pass # Placeholder if no specific pre-processing for other bands is needed

                if self.transforms is not None:
                    sample = self.transforms(sample)

                # Post-augmentation normalization for both transformed crops
                if len(self.bands) == 11:
                    means, stds, _ = get_marida_means_and_stds()
                    means_tensor = torch.tensor(means[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                    stds_tensor = torch.tensor(stds[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                elif len(self.bands) == 3:
                    means, stds = self.norm_param[0], self.norm_param[1]
                    means_tensor = torch.tensor(means, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                    stds_tensor = torch.tensor(stds, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                else:
                    means, stds = get_means_and_stds()
                    means_tensor = torch.tensor(means, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                    stds_tensor = torch.tensor(stds, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                
                # Apply normalization to the transformed sample(s)
                # This logic assumes self.transforms returns a tuple (q, k) for moco-geo,
                # or a single tensor otherwise.
                if self.model_name == "moco-geo":
                    # Ensure sample is a tuple of two tensors if transforms returned two crops
                    if isinstance(sample, tuple) and len(sample) == 2:
                        q_raw, k_raw = sample
                        q = (q_raw - means_tensor) / stds_tensor
                        k = (k_raw - means_tensor) / stds_tensor
                        pseudo_label = self.geo_to_label[str(file_path)]
                        return (q, k, torch.tensor(pseudo_label, dtype=torch.long))
                    else: # Handle case where transform might return a single sample for moco-geo, though less common
                        sample_normalized = (sample - means_tensor) / stds_tensor
                        pseudo_label = self.geo_to_label[str(file_path)]
                        return sample_normalized, torch.tensor(pseudo_label, dtype=torch.long)
                else:
                    sample_normalized = (sample - means_tensor) / stds_tensor
                    return sample_normalized.float()

        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening {file_path}: {e}")
            return None # Return None if file cannot be opened
        except KeyError as e:
            print(f"KeyError in __getitem__: {e}")
            raise # Re-raise KeyError for debugging specific missing keys