import torch
import json
import os
import rasterio
import numpy as np
import config
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from config import get_means_and_stds, get_marida_means_and_stds
from pytorch_lightning import LightningDataModule
from typing import Optional, List
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images

class HydroDataset(Dataset):
    def __init__(self,
                 path_dataset: Path,
                 bands: List[str] = None,
                 transforms=None,
                 compute_stats: bool = False,
                 location="agia_napa",
                 ocean_flag=True,
                 csv_features_path: str = "/home/joanna/SSLORS2/src/utils/train_ocean_labels_3_clusters_correct.csv"):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        #self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands 
        self.location = location
        if self.bands == None:
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        self.band_means, self.band_stds = config.get_means_and_stds()
        self.band_means, self.band_stds,_ = config.get_marida_means_and_stds()
        self.transforms = transforms
        self.impute_nan = np.tile(self.band_means, (256,256,1))
        self.ocean_flag=ocean_flag
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location] 
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location]) 
        
        self.csv_features_path = Path(csv_features_path)
        
        if self.ocean_flag:
            self.file_path_to_csv_row_map = {}
            self.file_paths = []

            self.csv_df = None
            
            self._load_ocean_features_and_map(all_file_paths)
        else:
            self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
            
        print("len self.file_paths",len(self.file_paths))
        
    def _load_ocean_features_and_map(self, all_file_paths: List[Path]):
        print(f"Starting data mapping. CSV path: {self.csv_features_path}")
        if not self.csv_features_path.exists():
            raise FileNotFoundError(
                f"Ocean features CSV not found at '{self.csv_features_path}'. "
                "Please ensure the CSV file is in the correct location and name."
            )
        try:
            self.csv_df = pd.read_csv(self.csv_features_path)
            print(f"Loaded CSV with {len(self.csv_df)} rows.")

            if 'file_dir' not in self.csv_df.columns:
                raise ValueError("CSV must contain a 'file_dir' column for direct file path mapping.")
            
            csv_file_dir_map = {Path(p).resolve(): row for p, row in self.csv_df.set_index('file_dir').iterrows()}
            print(f"Created CSV file_dir to row map with {len(csv_file_dir_map)} entries.")

            successful_matches = []
            print(f"Starting direct mapping process for {len(all_file_paths)} TIF files.")
            
            for i, file_path in enumerate(all_file_paths):
                if i % 1000 == 0:
                    print(f"Matching TIF file {i}/{len(all_file_paths)}")
                
                resolved_file_path = file_path.resolve()
                if resolved_file_path in csv_file_dir_map:
                    matched_row = csv_file_dir_map[resolved_file_path]
                    self.file_path_to_csv_row_map[file_path] = matched_row
                    successful_matches.append(file_path)
            
            self.file_paths = successful_matches
            print(f"Finished direct mapping. Successfully matched {len(self.file_paths)} TIF files.")

        except Exception as e:
            raise RuntimeError(
                f"Error loading or processing ocean features from '{self.csv_features_path}': {e}. "
                "Please ensure the CSV format includes a 'file_dir' column with correct paths."
            ) from e

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
    
                sample = sample[0:11,:,:]
                return sample.float()
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening {file_path}: {e}")
            return None