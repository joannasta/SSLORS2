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
        csv_features_path: str = "/home/joanna/SSLORS2/src/utils/ocean_features_capped_bathy.csv",
        ocean_flag=True
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        print(f"Found {len(all_file_paths)} TIF files initially.")
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        self.transforms = transforms
        self.model_name = model_name
        self.ocean_flag = ocean_flag
        self.location = location
        self.csv_features_path = Path(csv_features_path)

        self._load_normalization_params()
        
        self.file_path_to_csv_row_map = {}
        self.file_paths = []

        if self.ocean_flag:
            self._load_ocean_features_and_map(all_file_paths)
            print(f"Mapped {len(self.file_paths)} TIF files to CSV entries after initial filtering.")

        self._filter_valid_files()

        if not self.file_paths:
            raise ValueError(
                "No valid TIF files found in the dataset after all filtering steps. "
                "Check TIF file existence, readability, CSV matching, and 'label' column availability."
            )
        print(f"Dataset initialization complete. Total matched and validated files: {len(self.file_paths)}")


    def _load_normalization_params(self):
        if len(self.bands) == 11:
            means_np, stds_np, _ = get_marida_means_and_stds()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        elif len(self.bands) == 3:
            means_np = np.load(NORM_PARAM_PATHS[self.location])[0]
            stds_np = np.load(NORM_PARAM_PATHS[self.location])[1]
            if means_np.shape[0] < 3:
                raise ValueError(f"Normalization parameters for 3 bands at '{self.location}' are insufficient ({means_np.shape[0]} channels).")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.means_tensor = torch.tensor(means_np[:3], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                self.stds_tensor = torch.tensor(stds_np[:3], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        else:
            means_np, stds_np = get_means_and_stds()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)


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
            if 'label' not in self.csv_df.columns:
                raise ValueError("CSV must contain a 'label' column for ocean features.")
            
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
                "Please ensure the CSV format includes a 'file_dir' column with correct paths and a 'label' column."
            ) from e

    def _filter_valid_files(self):
        print(f"Starting validity check for {len(self.file_paths)} mapped TIF files...")
        valid_file_paths = []
        for i, file_path in enumerate(self.file_paths):
            if i % 1000 == 0:
                print(f"Validating file {i}/{len(self.file_paths)}")
            
            try:
                with rasterio.open(file_path) as src:
                    pass
            except (rasterio.errors.RasterioIOError, Exception) as e:
                continue

            features_row = self.file_path_to_csv_row_map.get(file_path)
            if features_row is None or 'label' not in features_row:
                continue

            valid_file_paths.append(file_path)
        
        self.file_paths = valid_file_paths
        print(f"Finished validity check. {len(self.file_paths)} valid TIF files remain.")


    def __len__(self) -> int:
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> torch.Tensor:
        with rasterio.open(file_path) as src:
            bands_data = []
            for i in range(1, len(self.bands) + 1):
                band_data = src.read(i)
                band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                bands_data.append(band_tensor)
            sample = torch.cat(bands_data, dim=0).contiguous()

            if len(self.bands) == 11:
                nan_mask = torch.isnan(sample)
                if torch.any(nan_mask):
                    means, _, _ = get_marida_means_and_stds()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        impute_nan_val = torch.tensor(means.astype(np.float32)).unsqueeze(-1).unsqueeze(-1)
                    impute_nan_expanded = impute_nan_val.expand_as(sample)
                    sample[nan_mask] = impute_nan_expanded[nan_mask]
            return sample

    def _normalize_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor for normalization, but got {type(img_tensor)}")
        
        if img_tensor.shape[0] != self.means_tensor.shape[0]:
            if img_tensor.shape[0] == 3 and self.means_tensor.shape[0] > 3:
                sliced_means = self.means_tensor[1:4, :, :]
                sliced_stds = self.stds_tensor[1:4, :, :]
                normalized_tensor = (img_tensor - sliced_means) / sliced_stds
                return normalized_tensor
            else:
                raise ValueError(
                    f"Image tensor channels ({img_tensor.shape[0]}) do not match "
                    f"normalization parameter channels ({self.means_tensor.shape[0]}). "
                    "Check `self.bands` configuration and image reading. "
                    "No automatic channel slicing available for this mismatch."
                )

        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        
        if normalized_tensor.shape[0] < 4:
            if normalized_tensor.shape[0] == 3:
                return normalized_tensor
            else:
                raise ValueError(f"Cannot slice channels [1:4]. Normalized tensor only has {normalized_tensor.shape[0]} channels. Check `self.bands` configuration or `_normalize_tensor` logic.")
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor], Tuple[torch.Tensor, torch.LongTensor]]:
        file_path = self.file_paths[idx]
        
        sample = self._read_and_process_image(file_path)

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        features_row = self.file_path_to_csv_row_map[file_path]
        
        cluster_label = torch.tensor(features_row['label'], dtype=torch.long)

        if self.model_name == "ocean_aware":
            if not (isinstance(sample, (list, tuple)) and len(sample) == 2 and 
                    isinstance(sample[0], torch.Tensor) and isinstance(sample[1], torch.Tensor)):
                raise TypeError(
                    f"For 'moco-geo-ocean' model, transforms must return a tuple/list of two torch.Tensors, "
                    f"but got {type(sample)} with content types: "
                    f"{[type(s) for s in sample] if isinstance(sample, (list, tuple)) else 'N/A'}"
                )
            
            q_raw, k_raw = sample
            q_normalized = self._normalize_tensor(q_raw)
            k_normalized = self._normalize_tensor(k_raw)
            
            print("q_normalized",q_normalized.shape)
            print("k_normalized",k_normalized.shape)
            print("cluster_label",cluster_label)
            return (q_normalized, k_normalized, cluster_label)
        else:
            if not isinstance(sample, torch.Tensor):
                 raise TypeError(
                    f"For non-'moco-geo-ocean' models, transforms must return a single torch.Tensor, "
                    f"but got {type(sample)}"
                )
            normalized_sample = self._normalize_tensor(sample)
            return (normalized_sample, cluster_label)