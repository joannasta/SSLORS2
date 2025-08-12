import torch
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
import sys
import warnings

from typing import Optional, List, Callable, Tuple, Union
from torch.utils.data import Dataset
from torchvision import transforms as T
from rasterio.warp import transform as rasterio_transform
import pyproj

from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_means_and_stds, get_marida_means_and_stds

class HydroMocoGeoOceanFeaturesDataset(Dataset):
    def __init__(
        self,
        path_dataset: Path,
        bands: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        location: str = "agia_napa",
        model_name: str = "ocean_aware",
        csv_features_path: str = "/home/joanna/SSLORS2/src/utils/ocean_features//ocean_featues_nans_bathy.csv",#"#train_ocean_labels_3_clusters.csv",
        num_geo_clusters: int = 3,
        ocean_flag=True
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.num_geo_clusters = num_geo_clusters
        self.ocean_flag = ocean_flag
        
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        self.transforms = transforms
        self.model_name = model_name
        self.location = location
        self.csv_features_path = Path(csv_features_path)

        self._load_normalization_params()
        
        self.file_path_to_csv_row_map = {}
        self.file_paths = []

        if self.ocean_flag:
            self._load_ocean_features_and_map(all_file_paths)
            print(f"Mapped {len(self.file_paths)} TIF files to CSV entries after initial filtering.")

    def _load_normalization_params(self):
        if len(self.bands) == 11:
            means_np, stds_np, _ = get_marida_means_and_stds()
            self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        elif len(self.bands) == 3:
            means_np = np.load(NORM_PARAM_PATHS[self.location])[0]
            stds_np = np.load(NORM_PARAM_PATHS[self.location])[1]
            self.means_tensor = torch.tensor(means_np[:3], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np[:3], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        else:
            means_np, stds_np = get_means_and_stds()
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)


    def _load_ocean_features_and_map(self, all_file_paths: List[Path]):
        print(f"Starting data mapping. CSV path: {self.csv_features_path}")

        self.csv_df = pd.read_csv(self.csv_features_path)
        print(f"Loaded CSV with {len(self.csv_df)} rows.")

        csv_file_dir_map = {Path(p).resolve(): row for p, row in self.csv_df.set_index('file_dir').iterrows()}
        successful_matches = []
        for i, file_path in enumerate(all_file_paths):
            resolved_file_path = file_path.resolve()
            if resolved_file_path in csv_file_dir_map:
                matched_row = csv_file_dir_map[resolved_file_path]
                self.file_path_to_csv_row_map[file_path] = matched_row
                successful_matches.append(file_path)
            
        self.file_paths = successful_matches
        print(f"Finished direct mapping. Successfully matched {len(self.file_paths)} TIF files.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> torch.Tensor:
        with rasterio.open(file_path) as src:
            band_indices = list(range(1, len(self.bands) + 1))
            sample_data = src.read(band_indices)
            sample = torch.from_numpy(sample_data.astype(np.float32))

            if len(self.bands) == 11:
                nan_mask = torch.isnan(sample)
                means, _, _ = get_marida_means_and_stds()
                impute_nan_val = torch.tensor(means.astype(np.float32)).unsqueeze(-1).unsqueeze(-1)
                impute_nan_expanded = impute_nan_val.expand_as(sample)
                sample[nan_mask] = impute_nan_expanded[nan_mask]
            return sample

    def _normalize_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor], Tuple[torch.Tensor, torch.LongTensor]]:
        file_path = self.file_paths[idx]
        sample = self._read_and_process_image(file_path)

        if self.transforms is not None:
            sample = self.transforms(sample)
            
        q_raw, k_raw = sample
        
        q_normalized = self._normalize_tensor(q_raw)
        k_normalized = self._normalize_tensor(k_raw)
        
        features_row = self.file_path_to_csv_row_map[file_path]
        cluster_label = torch.tensor(features_row['label'], dtype=torch.long)
            
        return (q_normalized, k_normalized, cluster_label)