import torch
import json
import os
import rasterio
import numpy as np
import config
import matplotlib.pyplot as plt
import csv
import random 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds, get_marida_means_and_stds
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Callable
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images
from sklearn.cluster import KMeans

class HydroMoCoDataset(Dataset):
    def __init__(
            self, path_dataset: Path, bands: List[str] = None,
            compute_stats: bool = False, location="agia_napa",
            model_name="mae",transform: Optional[Callable] = None):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands
        self.location = location
        if self.bands is None:
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def _load_data(self):
        all_data = []
        for file_path in self.file_paths:
            try:
                with rasterio.open(file_path) as src:
                    row, col = src.height // 2, src.width // 2
                    lon, lat = src.transform * (col, row)
                    all_data.append((str(file_path), lat, lon)) # Store path and coordinates
            except rasterio.errors.RasterioIOError as e:
                print(f"Error opening {file_path}: {e}")
        return all_data

    def _load_and_preprocess(self, file_path: Path) -> Optional[torch.Tensor]:
        try:
            with rasterio.open(file_path) as src:
                bands_data = []
                for i in range(1, len(self.bands) + 1):
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands_data.append(band_tensor)
                sample = torch.cat(bands_data, dim=0).contiguous()

                if len(self.bands) == 11:
                    nan_mask = torch.isnan(sample)
                    means, stds, _ = get_marida_means_and_stds()
                    impute_nan = np.tile(means, (256, 256, 1)).transpose(2, 1, 0)
                    sample[nan_mask] = torch.from_numpy(impute_nan)[nan_mask]
                    # Normalization will happen after augmentation
                    return sample.float()
                elif len(self.bands) == 3:
                    means, stds = self.norm_param[0][:, None, None], self.norm_param[1][:, None, None]
                    # Normalization will happen after augmentation
                    return sample.float()
                else:
                    means, stds = get_means_and_stds()
                    # Normalization will happen after augmentation
                    return sample.float()

        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening {file_path}: {e}")
            return None
        except KeyError as e:
            print(f"KeyError in _load_and_preprocess: {e}")
            raise

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        img = self._load_and_preprocess(file_path)

        if img is None:
            return None 

        img_q, img_k = self.transform(img) 

        # Now, apply normalization to each of the two tensors individually.
        if len(self.bands) == 11:
            means, stds, _ = get_marida_means_and_stds()
            # Convert means/stds to tensors once for efficiency within __getitem__
            means_tensor = torch.tensor(means[:11, None, None], dtype=torch.float32)
            stds_tensor = torch.tensor(stds[:11, None, None], dtype=torch.float32)
            img_q = (img_q - means_tensor) / stds_tensor
            img_k = (img_k - means_tensor) / stds_tensor
        elif len(self.bands) == 3:
            means, stds = self.norm_param[0][:, None, None], self.norm_param[1][:, None, None]
            means_tensor = torch.tensor(means, dtype=torch.float32) # `means` is already (C,1,1)
            stds_tensor = torch.tensor(stds, dtype=torch.float32) # `stds` is already (C,1,1)
            img_q = (img_q - means_tensor) / stds_tensor
            img_k = (img_k - means_tensor) / stds_tensor
        else:
            means, stds = get_means_and_stds()
            means_tensor = torch.tensor(means[:, None, None], dtype=torch.float32)
            stds_tensor = torch.tensor(stds[:, None, None], dtype=torch.float32)
            img_q = (img_q - means_tensor) / stds_tensor
            img_k = (img_k - means_tensor) / stds_tensor

        return img_q, img_k 