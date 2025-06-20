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
            csv_path: str = "/home/joanna/SSLORS/src/data/hydro/train_geo_labels10_projected.csv",
            num_geo_clusters: int = 100
    ):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.num_geo_clusters = num_geo_clusters
        
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
        if self.model_name == "moco-geo":
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
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Required geo-labels CSV not found at '{self.csv_path}'. "
                "For 'moco-geo' model, this CSV must pre-exist as automatic generation is disabled. "
                "Please ensure the CSV file is in the correct location."
            )
        try:
            df = pd.read_csv(self.csv_path)
            self.geo_to_label = {row['file_dir']: row['label'] for _, row in df.iterrows()}
            print(f"Successfully loaded {len(self.geo_to_label)} geo-labels from '{self.csv_path}'.")
        except Exception as e:
            raise RuntimeError(
                f"Error loading geo-labels from '{self.csv_path}': {e}. "
                "Please ensure the CSV format (label,file_dir,lat,lon) is correct."
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
        # Ensure img_tensor is indeed a Tensor before subtraction
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor for normalization, but got {type(img_tensor)}")
        
        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        # Select channels from 1 to 3 (inclusive), which is 1:4 in Python slicing
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        sample = self._read_and_process_image(file_path)

        if sample is None:
            return None # Skip this sample if reading failed

        # Apply data augmentations/transformations if provided
        if self.transforms is not None:
            sample = self.transforms(sample)

        # Handle different model types and transformation outputs
        if self.model_name == "moco-geo":
            # Expecting a list of two tensors from TwoCropsTransform
            if isinstance(sample, list) and len(sample) == 2:
                q_raw, k_raw = sample
                
                # Normalize query and key crops
                q_normalized = self._normalize_tensor(q_raw)[1:4, :, :]  # Select channels 1:4
                k_normalized = self._normalize_tensor(k_raw)[1:4, :, :] 
                
                pseudo_label = self.geo_to_label.get(str(file_path), -1)
                if pseudo_label == -1:
                    print(f"WARNING: No geo-label found for '{file_path}' in CSV. Using default label -1.")
                
                # Return as a tuple: (query_image, key_image, label)
                return (q_normalized, k_normalized, torch.tensor(pseudo_label, dtype=torch.long))
            
            # Fallback for moco-geo if transforms somehow returned a single tensor (less common)
            elif isinstance(sample, torch.Tensor):
                sample_normalized = self._normalize_tensor(sample)
                pseudo_label = self.geo_to_label.get(str(file_path), -1)
                if pseudo_label == -1:
                    print(f"WARNING: No geo-label found for '{file_path}' in CSV. Using default label -1.")
                return sample_normalized.float(), torch.tensor(pseudo_label, dtype=torch.long)
            else:
                # If the transformed 'sample' is neither a list of two Tensors nor a single Tensor
                raise TypeError(
                    f"Unexpected type returned by transforms for moco-geo model: {type(sample)}. "
                    "Expected a list of two Tensors or a single Tensor."
                )
        else: # For other models (e.g., 'mae'), expect a single tensor
            sample_normalized = self._normalize_tensor(sample)
            sample_normalized = sample_normalized[1:4, :, :]  
            return sample_normalized.float()