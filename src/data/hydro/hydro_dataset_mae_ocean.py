import torch
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
import sys
import warnings

from typing import Optional, List, Callable, Tuple
from torch.utils.data import Dataset
from torchvision import transforms as T
from rasterio.warp import transform as rasterio_transform
import pyproj

from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_means_and_stds, get_marida_means_and_stds

class HydroMaeOceanFeaturesDataset(Dataset):
    def __init__(
        self,
        path_dataset: Path,
        bands: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        location: str = "agia_napa",
        model_name: str = "mae",
        csv_features_path: str = "/home/joanna/SSLORS2/src/utils/train_ocean_labels_3_clusters_correct.csv",
        ocean_flag=True
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        self.transforms = transforms
        self.model_name = model_name
        self.ocean_flag = ocean_flag
        self.location = location
        self.csv_features_path = Path(csv_features_path)

        self._load_normalization_params()
        
        if self.ocean_flag:
            self.file_path_to_csv_row_map = {}
            self.file_paths = []

            self.csv_df = None

            self._load_ocean_features_and_map(all_file_paths)
        else:
            self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))


    def _load_normalization_params(self):  
        means_np, stds_np = get_means_and_stds()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)


    def _load_ocean_features_and_map(self, all_file_paths: List[Path]):
        if not self.csv_features_path.exists():
            raise FileNotFoundError(
                f"Ocean features CSV not found at '{self.csv_features_path}'. "
                "Please ensure the CSV file is in the correct location and name."
            )
        
        self.csv_df = pd.read_csv(self.csv_features_path)

        if 'file_dir' not in self.csv_df.columns:
            raise ValueError("CSV must contain a 'file_dir' column for direct file path mapping.")
        
        csv_file_dir_map = {Path(p).resolve(): row for p, row in self.csv_df.set_index('file_dir').iterrows()}

        successful_matches = []
        
        for file_path in all_file_paths:
            resolved_file_path = file_path.resolve()
            if resolved_file_path in csv_file_dir_map:
                matched_row = csv_file_dir_map[resolved_file_path]
                self.file_path_to_csv_row_map[file_path] = matched_row
                successful_matches.append(file_path)
        
        self.file_paths = successful_matches


    def __len__(self) -> int:
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> Optional[torch.Tensor]:
        with rasterio.open(file_path) as src:
            bands = []
            for i in range(1, len(self.bands) + 1):
                band_data = src.read(i)
                band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                bands.append(band_tensor)
            sample = torch.cat(bands, dim=0)
            sample = sample.contiguous()
            return sample


    def _normalize_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        return normalized_tensor
                    
    
    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        sample = self._read_and_process_image(file_path)
        
        if sample is None:
            return None 

        sample = self._normalize_tensor(sample)

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        features_row = self.file_path_to_csv_row_map.get(file_path)
        if features_row is None:
            warnings.warn(f"No features found for {file_path}. Returning None for this item.")
            return None
        
        secchi = float(features_row['secchi'])
        bathy = float(features_row['bathy'])
        chlorophyll = float(features_row['chlorophyll'])
            
        ocean_features = torch.tensor([secchi, bathy, chlorophyll], dtype=torch.float32)
        sample = sample[1:4,:,:]    
        return sample, ocean_features