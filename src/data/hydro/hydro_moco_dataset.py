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

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        gaussian_blur = T.GaussianBlur(kernel_size=7, sigma=sigma)
        x = gaussian_blur(x)
        return x

class HydroMoCoDataset(Dataset):
    def __init__(
            self, path_dataset: Path, bands: List[str] = None,
            compute_stats: bool = False, location="agia_napa",
            model_name="mae"):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands
        self.location = location
        if self.bands is None:
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        self.augmentations = T.Compose([
            T.Resize(256 * 2),
            T.RandomResizedCrop(256, scale=(0.2, 1.)),
            T.RandomApply([
                # T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            # T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [GaussianBlur([.1, 2.])], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # Normalization will be applied in __getitem__
        ])

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

        img1 = self.augmentations(img)
        img2 = self.augmentations(img)

        # Apply normalization after augmentation
        if len(self.bands) == 11:
            means, stds, _ = get_marida_means_and_stds()
            img1 = (img1 - torch.tensor(means[:11, None, None])) / torch.tensor(stds[:11, None, None])
            img2 = (img2 - torch.tensor(means[:11, None, None])) / torch.tensor(stds[:11, None, None])
        elif len(self.bands) == 3:
            means, stds = self.norm_param[0][:, None, None], self.norm_param[1][:, None, None]
            img1 = (img1 - torch.tensor(means)) / torch.tensor(stds)
            img2 = (img2 - torch.tensor(means)) / torch.tensor(stds)
        else:
            means, stds = get_means_and_stds()
            img1 = (img1 - torch.tensor(means[:, None, None])) / torch.tensor(stds[:, None, None])
            img2 = (img2 - torch.tensor(means[:, None, None])) / torch.tensor(stds[:, None, None])

        return img1, img2