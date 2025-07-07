import torch
import json
import os
import rasterio
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds,get_marida_means_and_stds,NORM_PARAM_DEPTH,NORM_PARAM_PATHS
from pytorch_lightning import LightningDataModule
from typing import Optional, List

class HydroDataset(Dataset):
    def __init__(self, path_dataset: Path, bands: List[str] = None, transforms=None,compute_stats: bool = False,
                 location="agia_napa"):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands 
        self.location = location

        self.band_means, self.band_stds = config.get_means_and_stds()
        self.transforms = transforms
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])


    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        
        with rasterio.open(file_path) as src:
            bands = []
            bands = [
                torch.from_numpy(src.read(i).astype(np.float32)).unsqueeze(0)
                if src.read(i).shape[-2:] != (256, 256) else torch.from_numpy(src.read(i).astype(np.float32)).unsqueeze(0)
                for i in range(1, len(self.bands) + 1)
            ]

        sample = torch.cat(bands, dim=0) 
        if len(self.bands) == 12:
            means, stds = get_means_and_stds()
            sample = (sample - means[:,None,None]) /stds[:,None,None]
        elif len(self.bands) == 11:
            means, stds,_ = get_marida_means_and_stds()
            sample = (sample - means[:11,None,None]) /stds[:11,None,None]
        else:
            means,stds = self.norm_param[0][:, np.newaxis, np.newaxis],self.norm_param[1][:, np.newaxis, np.newaxis]

            sample = (sample - means[1:4,None,None]) /stds[1:4,None,None]

        if self.transforms is not None:
            sample = self.transforms(sample)
        #if len(self.bands) == 12:    
        #    sample = sample[1:4,:,:] 
        return sample.float()