import torch
import json
import os
import rasterio
import numpy as np

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
        print(f"Found {len(self.file_paths)} .tif files in {self.path_dataset}")
        self.bands = bands or ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

        self.band_means, self.band_stds = self._compute_dataset_stats()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transforms = transforms

        # Store min and max values for each sample
        self.min_vals = []
        self.max_vals = []
        if compute_stats:
            # Compute min/max values over the entire dataset
            self._compute_dataset_stats
        print(f"Dataset initialized with {len(self.bands)} bands.")

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        means, stds = get_means_and_stds()
        with rasterio.open(file_path) as src:
            bands = []
            for i in range(1, len(self.bands) + 1):
                band = src.read(i).astype(np.float32)  
                band = torch.tensor(band, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                band = self.resize(band) if band.shape[-2:] != (256, 256) else band
                bands.append(band)

        # Stack bands into a multi-channel tensor
        sample = torch.cat(bands, dim=0) # shape (C, 256, 256)
        assert not torch.isnan(sample).any(), "Input contains NaN"
        assert not torch.isinf(sample).any(), "Input contains Inf"
        # Min-Max Scaling (with epsilon for stability)
        #min_value = np.percentile(sample.reshape(-1), 1)
        #max_value = np.percentile(sample.reshape(-1), 99)
        sample = (sample - means[:,None,None]) /stds[:,None,None]#- min_value) / (max_value - min_value+ 1e-7)

        # Store min and max values for this sample
        #self.min_vals.append(min_value)
        #self.max_vals.append(max_value)

        if self.transforms is not None:
            sample = self.transforms(sample)
        sample = sample[1:4,:,:] # BGR # shape (3, 256, 256)
        return sample.float()

    def _compute_dataset_stats(self, stats_file: str = "dataset_stats.json"):
        """Compute per-band mean and standard deviation, then save to a file."""
        # Check if stats file exists
        if os.path.exists(stats_file):
            print(f"Loading dataset statistics from {stats_file}...")
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            return stats['means'], stats['stds']