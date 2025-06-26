import torch
import json
import os
import rasterio
import numpy as np
import config

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds, get_marida_means_and_stds
from pytorch_lightning import LightningDataModule
from typing import Optional, List
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images

class HydroDataset(Dataset):
    def __init__(self, path_dataset: Path, bands: List[str] = None, transforms=None,compute_stats: bool = False,location="agia_napa"):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands 
        self.location = location
        if self.bands == None:
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        self.band_means, self.band_stds = config.get_means_and_stds()
        self.band_means, self.band_stds,_ = config.get_marida_means_and_stds()
        self.transforms = transforms
        self.impute_nan = np.tile(self.band_means, (256,256,1))
        #print("Get Parameters for location:",self.location)
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location] #puck_lagoon
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location]) #puck_lagoon

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        try:
            with rasterio.open(file_path) as src:
                bands = []
                for i in range(1, len(self.bands) + 1):
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands.append(band_tensor)
                sample = torch.cat(bands, dim=0)
                sample = sample.contiguous()

                if len(self.bands) == 11:
                    nan_mask = torch.from_numpy(np.isnan(sample.numpy()))
                    num_nans = torch.sum(nan_mask).item()
                    sample[nan_mask] = torch.from_numpy(self.impute_nan.transpose(2, 1, 0))[nan_mask]
                    means, stds,_ = get_marida_means_and_stds()#get_means_and_stds()
                    sample = (sample - means[:11, None, None]) / stds[:11, None, None]
                elif len(self.bands) == 3:
                        means, stds = self.norm_param[0][:, np.newaxis, np.newaxis], self.norm_param[1][:, np.newaxis, np.newaxis] 
                        sample = (sample - means) / stds
                else:    
                    means, stds =  get_means_and_stds()
                    sample = (sample - means[:, None, None]) / stds[:, None, None]
                    
                if self.transforms is not None:
                    sample = self.transforms(sample)
                return sample.float()
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening {file_path}: {e}")
            return None
