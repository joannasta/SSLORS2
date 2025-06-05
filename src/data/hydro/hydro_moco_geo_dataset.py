import torch
import numpy as np
import rasterio
import pandas as pd # Essential for loading your CSVs
import os 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List, Callable

# Assuming these are available from your 'config' module
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_means_and_stds, get_marida_means_and_stds

class HydroMoCoGeoDataset(Dataset):
    def __init__(
            self, path_dataset: Path, bands: List[str] = None,
            transforms: Optional[Callable] = None, 
            location="agia_napa", 
            model_name="mae", 
            csv_path="/home/joanna/SSLORS/src/data/hydro/train_geo_labels10_projected.csv", # No 'compute_stats' or 'save_csv' as they are not used for this functionality
            num_geo_clusters=100):
        self.path_dataset = Path(path_dataset)
        # Efficiently get all .tif file paths. This happens once.
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.num_geo_clusters = num_geo_clusters # Store number of geo clusters for potential future use
        self.bands = bands
        if self.bands is None:
            self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

        self.transforms = transforms
        self.model_name = model_name
        
        self.csv_path = Path(csv_path) # Convert to Path object for easier handling
        self.location = location 

        # Load normalization parameters relevant to the specified location
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        self.geo_to_label = {} # Initialize an empty dictionary for geo-labels

        # --- Geo-Label Loading Logic (Strictly from CSV) ---
        if self.model_name == "moco-geo":
            if self.csv_path.exists():
                print(f"Model is 'moco-geo'. Attempting to load geo-labels from: {self.csv_path}")
                self._load_geo_labels_from_csv() # Call the method to load labels
            else:
                # If the CSV doesn't exist, raise an error immediately
                raise FileNotFoundError(
                    f"Required geo-labels CSV not found at '{self.csv_path}'. "
                    "For 'moco-geo' model, this CSV must pre-exist as automatic generation is disabled. "
                    "Please ensure the CSV file is in the correct location."
                )
        # --- End Geo-Label Loading Logic ---

        # --- Pre-compute normalization tensors once in __init__ for efficiency ---
        # This avoids repeated tensor creation in __getitem__
        if len(self.bands) == 11:
            means_np, stds_np, _ = get_marida_means_and_stds()
            self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        elif len(self.bands) == 3: # Assuming your config's norm_param contains means/stds for 3 bands
            means_np, stds_np = self.norm_param[0], self.norm_param[1]
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        else: # Default case for other band counts (e.g., if get_means_and_stds() is generic)
            means_np, stds_np = get_means_and_stds()
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        # --- End Pre-computation ---

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def _load_geo_labels_from_csv(self):
        """
        Internal method to load geo-labels from the specified CSV file.
        This method is called only if `model_name` is 'moco-geo' and the CSV exists.
        It assumes the CSV has 'file_dir' and 'label' columns.
        """
        try:
            df = pd.read_csv(self.csv_path)
            # Efficiently create a dictionary mapping file_dir strings to their labels
            self.geo_to_label = {row['file_dir']: row['label'] for index, row in df.iterrows()}
            print(f"Successfully loaded {len(self.geo_to_label)} geo-labels from '{self.csv_path}'.")
        except Exception as e:
            # Catch any exception during CSV loading (e.g., wrong format)
            raise RuntimeError(f"Error loading geo-labels from '{self.csv_path}': {e}. "
                               "Please ensure the CSV format (label,file_dir,lat,lon) is correct.") from e

    def __getitem__(self, idx: int):
        """Retrieves a single sample (image data and optionally geo-label) by index."""
        file_path = self.file_paths[idx]
        try:
            with rasterio.open(file_path) as src:
                bands_data = []
                for i in range(1, len(self.bands) + 1):
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands_data.append(band_tensor)
                sample = torch.cat(bands_data, dim=0).contiguous()

                # Pre-processing: NaN handling for 11-band MARIDA data
                if len(self.bands) == 11:
                    nan_mask = torch.isnan(sample)
                    if torch.any(nan_mask): # Only impute if NaNs are actually present
                        means, _, _ = get_marida_means_and_stds() # Get means for imputation value
                        impute_nan_val = torch.tensor(means.astype(np.float32)).unsqueeze(-1).unsqueeze(-1)
                        # Expand mean values to fill the NaN positions
                        impute_nan_expanded = impute_nan_val.expand_as(sample)
                        sample[nan_mask] = impute_nan_expanded[nan_mask]
                
                # Apply data augmentations/transformations if provided
                if self.transforms is not None:
                    sample = self.transforms(sample)

                # Apply post-augmentation normalization using the pre-computed tensors
                if self.model_name == "moco-geo":
                    # Assumes self.transforms returns a tuple (q, k) for MoCo-Geo
                    if isinstance(sample, tuple) and len(sample) == 2:
                        q_raw, k_raw = sample
                        q_normalized = (q_raw - self.means_tensor) / self.stds_tensor
                        k_normalized = (k_raw - self.means_tensor) / self.stds_tensor
                        q_normalized = q_normalized[1:4,:,:] # Assuming q is a 4D tensor (C, H, W)
                        k_normalized = k_normalized[1:4,:,:] # Assuming k is a 4D tensor (C, H, W)
                        # Retrieve pseudo_label; use .get() with a default value to prevent KeyError
                        pseudo_label = self.geo_to_label.get(str(file_path), -1) 
                        if pseudo_label == -1: # Warn if a label was not found for a file
                            print(f"WARNING: No geo-label found for '{file_path}' in CSV. Using default label -1.")
                        
                        return (q_normalized, k_normalized, torch.tensor(pseudo_label, dtype=torch.long))
                    else: # Fallback if transform yields single crop for moco-geo (less common)
                        sample_normalized = (sample - self.means_tensor) / self.stds_tensor
                        sample_normalized = sample_normalized[1:4,:,:]
                        pseudo_label = self.geo_to_label.get(str(file_path), -1)
                        if pseudo_label == -1:
                            print(f"WARNING: No geo-label found for '{file_path}' in CSV. Using default label -1.")
                        return sample_normalized.float(), torch.tensor(pseudo_label, dtype=torch.long)
                else: # For other models (e.g., 'mae'), return a single normalized sample
                    sample_normalized = (sample - self.means_tensor) / self.stds_tensor
                    sample_normalized = sample_normalized[1:4,:,:]
                    return sample_normalized.float()

        except rasterio.errors.RasterioIOError as e:
            # Catch errors when a TIFF file cannot be opened (e.g., corrupt file, permissions)
            print(f"Error opening image file '{file_path}': {e}. This sample will be skipped.")
            return None # Return None; the DataLoader's collate_fn must handle filtering these out
        except Exception as e: 
            # Catch any other unexpected errors during item processing
            print(f"An unexpected error occurred while processing '{file_path}': {e}. This sample will be skipped.")
            return None # Return None; the DataLoader's collate_fn must handle filtering these out