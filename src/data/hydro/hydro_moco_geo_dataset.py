import torch
import json
import os
import rasterio
import numpy as np
import config
import matplotlib.pyplot as plt
import csv  # Import the csv module

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds, get_marida_means_and_stds
from pytorch_lightning import LightningDataModule
from typing import Optional, List
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images
from sklearn.cluster import KMeans

class HydroDataset(Dataset):
    def __init__(
            self, path_dataset: Path, bands: List[str] = None,
            transforms=None, compute_stats: bool = False, location="agia_napa",
            model_name="mae", num_geo_clusters=100, save_csv=False, csv_path="geo_labels.csv"): # Added save_csv and csv_path
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands
        self.location = location
        if self.bands is None:
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        self.transforms = transforms

        self.model_name = model_name
        self.num_geo_clusters = num_geo_clusters
        self.save_csv = save_csv  # Store whether to save to CSV
        self.csv_path = csv_path  # Store the path to the CSV file
        if self.model_name == "moco-geo":
            self.geo_to_label = self._create_geo_labels()
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

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
                bands = []
                for i in range(1, len(self.bands) + 1):
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands.append(band_tensor)
                sample = torch.cat(bands, dim=0).contiguous()

                if len(self.bands) == 11:
                    nan_mask = torch.from_numpy(np.isnan(sample.numpy()))
                    num_nans = torch.sum(nan_mask).item()
                    means, stds, _ = get_marida_means_and_stds()
                    self.impute_nan = np.tile(means, (256, 256, 1))
                    sample[nan_mask] = torch.from_numpy(self.impute_nan.transpose(2, 1, 0))[nan_mask]
                    sample = (sample - means[:11, None, None]) / stds[:11, None, None]
                elif len(self.bands) == 3:
                    means, stds = self.norm_param[0][:, np.newaxis, np.newaxis], self.norm_param[1][:, np.newaxis, np.newaxis]
                    sample = (sample - means) / stds
                else:
                    means, stds = get_means_and_stds()
                    sample = (sample - means[:, None, None]) / stds[:, None, None]

                if self.transforms is not None:
                    sample = self.transforms(sample)
                    if len(sample) == 2 and self.model_name == "moco-geo":
                        q, k = sample[0][1:4, :, :], sample[1][1:4, :, :]
                        pseudo_label = self.geo_to_label[str(file_path)]  # Ensure key is string
                        return (q, k, torch.tensor(pseudo_label, dtype=torch.long))
                    elif self.model_name == "moco-geo":  # Handle single crop case too, if needed
                        sample_transformed = sample[1:4, :, :]  # Or however you define it
                        pseudo_label = self.geo_to_label[str(file_path)]  # Ensure key is string
                        return sample_transformed, torch.tensor(pseudo_label, dtype=torch.long)

                return sample.float()

        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening {file_path}: {e}")
            return None
        except KeyError as e:
            print(f"KeyError in __getitem__: {e}")
            raise
