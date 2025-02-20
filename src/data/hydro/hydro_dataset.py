import torch
import json
import os
import rasterio
import numpy as np
import config

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds
from pytorch_lightning import LightningDataModule
from typing import Optional, List

class HydroDataset(Dataset):
    def __init__(self, path_dataset: Path, bands: List[str] = None, transforms=None,compute_stats: bool = False,):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands or ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11"]#, "B12"]
        self.band_means, self.band_stds = config.get_means_and_stds()
        self.transforms = transforms


    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        means, stds = get_means_and_stds()
        with rasterio.open(file_path) as src:
            bands = []
            bands = [
                torch.from_numpy(src.read(i).astype(np.float32)).unsqueeze(0)
                if src.read(i).shape[-2:] != (256, 256) else torch.from_numpy(src.read(i).astype(np.float32)).unsqueeze(0)
                for i in range(1, len(self.bands) + 1)
            ]

        sample = torch.cat(bands, dim=0) 
        if len(self.bands) == 11:
            sample = (sample - means[:11,None,None]) /stds[:11,None,None]
        else:
            sample = (sample - means[1:4,None,None]) /stds[1:4,None,None]

        if self.transforms is not None:
            sample = self.transforms(sample)
        if len(self.bands) ==12:    
            sample = sample[:11,:,:] 
        return sample.float()