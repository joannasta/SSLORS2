import torch
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from typing import Optional, List, Callable

from torch.utils.data import Dataset
from torchvision import transforms as T

# Assuming these are available from your 'config' module
# It's good practice to import only what's needed, but keeping existing imports
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_means_and_stds, get_marida_means_and_stds

class HydroMoCoGeoDataset(Dataset):
    def __init__(
            self,
            path_dataset: Path,
            bands: Optional[List[str]] = None,
            transforms: Optional[Callable] = None,
            location: str = "agia_napa",
            model_name: str = "mae",
            csv_path: str = "/home/joanna/SSLORS2/src/data/hydro/train_ocean_labels10_projected.csv",
            csv_features_path: str = "/home/joanna/SSLORS2/src/utils/train_ocean_labels_3_clusters_correct.csv",
            num_geo_clusters: int = 10,
            ocean_flag=True
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        #self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.num_geo_clusters = num_geo_clusters
        self.ocean_flag=ocean_flag
        
        # Default bands if not provided
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        
        self.transforms = transforms
        self.model_name = model_name
        self.location = location 
        self.csv_path = Path(csv_path)

        # Load normalization parameters upfront
        self._load_normalization_params()

        self.geo_to_label = {}
        
        self.csv_features_path = Path(csv_features_path)
        
        if self.ocean_flag:
            self.file_path_to_csv_row_map = {}
            self.file_paths = []

            self.csv_df = None
            
            self._load_ocean_features_and_map(all_file_paths)
        else:
            self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
            
        if self.model_name == "geo_aware":
            self._load_geo_labels()

    def _load_normalization_params(self):
        """Pre-computes and stores normalization tensors."""
        if len(self.bands) == 11:
            means_np, stds_np, _ = get_marida_means_and_stds()
            self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        elif len(self.bands) == 3:
            means_np, stds_np = np.load(NORM_PARAM_PATHS[self.location])[0], np.load(NORM_PARAM_PATHS[self.location])[1]
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        else:
            means_np, stds_np = get_means_and_stds()
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    def _load_geo_labels(self):
        """Loads geo-labels from CSV for 'moco-geo' model."""
        df = pd.read_csv(self.csv_path)
        self.geo_to_label = {row['file_dir']: row['label'] for _, row in df.iterrows()}
        print(f"Successfully loaded {len(self.geo_to_label)} geo-labels from '{self.csv_path}'.")
            
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

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> Optional[torch.Tensor]:
        """Reads a TIFF image and performs initial processing."""
        try:
            with rasterio.open(file_path) as src:
                bands_data = []
                for i in range(1, len(self.bands) + 1):
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands_data.append(band_tensor)
                sample = torch.cat(bands_data, dim=0).contiguous()

                # NaN handling for 11-band MARIDA data
                if len(self.bands) == 11:
                    nan_mask = torch.isnan(sample)
                    if torch.any(nan_mask):
                        means, _, _ = get_marida_means_and_stds()
                        impute_nan_val = torch.tensor(means.astype(np.float32)).unsqueeze(-1).unsqueeze(-1)
                        impute_nan_expanded = impute_nan_val.expand_as(sample)
                        sample[nan_mask] = impute_nan_expanded[nan_mask]
                return sample
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening image file '{file_path}': {e}. This sample will be skipped.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during image reading for '{file_path}': {e}. This sample will be skipped.")
            return None

    def _normalize_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Applies normalization (mean/std) to a tensor and selects channels 1:4."""    
        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        print("file_path",file_path)
        sample = self._read_and_process_image(file_path)

        if sample is None:
            return None # Skip this sample if reading failed

        # Apply data augmentations/transformations if provided
        if self.transforms is not None:
            sample = self.transforms(sample)

        # Expecting a list of two tensors from TwoCropsTransform
        if isinstance(sample, list) and len(sample) == 2:
            q_raw, k_raw = sample

            # Normalize query and key crops
            q_normalized = self._normalize_tensor(q_raw)
            k_normalized = self._normalize_tensor(k_raw)
                
            pseudo_label = self.geo_to_label.get(str(file_path), -1)
            print("pseudo_label",pseudo_label)
            if pseudo_label == -1:
                print(f"WARNING: No geo-label found for '{file_path}' in CSV. Using default label -1.")
            
            # Return as a tuple: (query_image, key_image, label)
            return (q_normalized, k_normalized, torch.tensor(pseudo_label, dtype=torch.long))
            