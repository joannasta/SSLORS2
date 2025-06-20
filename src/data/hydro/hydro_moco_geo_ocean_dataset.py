import torch
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from typing import Optional, List, Callable, Tuple

from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.spatial import KDTree # Import KDTree for nearest neighbor search

# Assuming these are available from your 'config' module
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_means_and_stds, get_marida_means_and_stds

class HydroMocoGeoOceanFeaturesDataset(Dataset): # Renamed for clarity and purpose
    def __init__(
            self,
            path_dataset: Path,
            bands: Optional[List[str]] = None,
            transforms: Optional[Callable] = None,
            location: str = "agia_napa",
            model_name: str = "moco-ocean-features", # Changed default model_name for this task
            csv_path: str = "/home/joanna/SSLORS/src/data/ocean_features_projected.csv", # Updated default CSV path
            # New parameter for KDTree matching
            max_match_distance: float = 0.001 
    ):
        self.path_dataset = Path(path_dataset)
        # Initially get all tif files, then filter based on CSV matching in _load_ocean_features_and_map
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif"))) 
        
        # num_geo_clusters is removed as it's not relevant for regression
        
        # Default bands if not provided
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        
        self.transforms = transforms
        self.model_name = model_name
        self.location = location 
        self.csv_path = Path(csv_path) # Now refers to the ocean features CSV
        self.max_match_distance = max_match_distance

        # Load normalization parameters upfront
        self._load_normalization_params()

        # Replaced geo_to_label with mapping for regression features
        self.file_path_to_features_map = {} 
        # This will store only the valid file paths after successful feature mapping
        self.file_paths = [] 

        # Load oceanographic features and create mapping (replaces _load_geo_labels)
        self._load_ocean_features_and_map(all_file_paths)

        if not self.file_paths and self.csv_path:
            raise ValueError(
                "No TIF files could be matched to ocean features in the CSV. "
                "Check TIF file validity, CSV data, `max_match_distance`, or if there are TIF files at all."
            )

    def _load_normalization_params(self):
        """Pre-computes and stores normalization tensors."""
        # The logic here remains largely as your provided base, ensuring proper tensor shapes
        if len(self.bands) == 11:
            means_np, stds_np, _ = get_marida_means_and_stds()
            self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        elif len(self.bands) == 3:
            means_np = np.load(NORM_PARAM_PATHS[self.location])[0]
            stds_np = np.load(NORM_PARAM_PATHS[self.location])[1]
            # Ensure the loaded means/stds have enough channels for the requested 3 bands
            if means_np.shape[0] < 3: 
                raise ValueError(f"Normalization parameters for 3 bands at '{self.location}' are insufficient ({means_np.shape[0]} channels).")
            self.means_tensor = torch.tensor(means_np[:3], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1) # Take first 3
            self.stds_tensor = torch.tensor(stds_np[:3], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1) # Take first 3
        else: # Assumed to be 12 bands (e.g. original Sentinel-2 without B10) or other counts
            means_np, stds_np = get_means_and_stds()
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    def _load_ocean_features_and_map(self, all_file_paths: List[Path]):
        """
        Loads oceanographic features (bathy, chlorophyll, secchi) from the specified CSV 
        and creates a mapping from TIF file path to these features.
        Matching is done using KDTree for nearest neighbor search based on the TIF's
        geographic center (derived from rasterio.bounds) and the CSV's lat/lon points.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Ocean features CSV not found at '{self.csv_path}'. "
                "Please ensure the CSV file is in the correct location and name."
            )
        try:
            # Load the CSV containing bathy, chlorophyll, secchi
            csv_df = pd.read_csv(self.csv_path)
            
            # Prepare CSV points for KDTree (using lat, lon columns)
            csv_coords = csv_df[['lat', 'lon']].values 
            csv_kdtree = KDTree(csv_coords)

            # Prepare features for quick lookup by original DataFrame index
            # Store bathy, chlorophyll, secchi as a torch tensor
            csv_features_by_index = {
                i: torch.tensor([row['bathy'], row['chlorophyll'], row['secchi']], dtype=torch.float32)
                for i, row in csv_df.iterrows()
            }

            successful_matches = []
            for file_path in all_file_paths:
                try:
                    with rasterio.open(file_path) as src:
                        # Get the bounding box of the TIF file from its metadata
                        bounds = src.bounds # Returns (left, bottom, right, top) in CRS units

                        # Calculate the center point of the TIF's bounding box
                        # This assumes the TIF's CRS is compatible with the CSV's lat/lon (e.g., WGS84)
                        tif_center_lon = (bounds.left + bounds.right) / 2
                        tif_center_lat = (bounds.bottom + bounds.top) / 2
                        query_point = np.array([tif_center_lat, tif_center_lon])

                        # Query KDTree for the nearest neighbor within max_match_distance
                        distance, index = csv_kdtree.query(query_point, k=1, distance_upper_bound=self.max_match_distance)

                        # If a match is found within the threshold
                        # index will be KDTree.n if no point is found within distance_upper_bound
                        if distance <= self.max_match_distance and index != csv_kdtree.n: 
                            self.file_path_to_features_map[file_path] = csv_features_by_index[index]
                            successful_matches.append(file_path)
                        else:
                            # Optionally print for debugging which files are skipped
                            # print(f"No CSV match found within {self.max_match_distance} for TIF center ({tif_center_lat:.4f}, {tif_center_lon:.4f}) from {file_path.name}. Skipping.")
                            pass

                except rasterio.errors.RasterioIOError as e:
                    print(f"Error reading TIF file {file_path}: {e}. Skipping this file.")
                except Exception as e: # Catch other potential errors during processing
                    print(f"An unexpected error occurred while processing {file_path}: {e}. Skipping this file.")
            
            self.file_paths = successful_matches 
            print(f"Successfully mapped {len(self.file_paths)} TIF files to ocean features from '{self.csv_path}'.")

        except Exception as e:
            raise RuntimeError(
                f"Error loading or processing ocean features from '{self.csv_path}': {e}. "
                "Please ensure the CSV format (lat,lon,bathy,chlorophyll,secchi) is correct and 'scipy' is installed."
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
                        # Ensure means are correctly shaped for broadcasting during imputation
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
        
        # Check if the number of channels in the image tensor matches the normalization parameters
        if img_tensor.shape[0] != self.means_tensor.shape[0]:
            print(f"Warning: Image tensor has {img_tensor.shape[0]} channels, but normalization tensors have {self.means_tensor.shape[0]} channels. This might lead to incorrect normalization for the full image before slicing.")
            # Depending on expected behavior, you might want to adjust self.means_tensor/stds_tensor
            # dynamically or raise a more severe error. Proceeding for now with potential misalignment.

        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        
        # Select channels from 1 to 3 (inclusive), which is 1:4 in Python slicing
        # This means the *output* of the dataset (the image fed to the model)
        # will always have 3 channels (Band 2, Band 3, Band 4 assuming 0-indexed).
        if normalized_tensor.shape[0] < 4: # Ensure there are enough channels to slice [1:4]
            raise ValueError(f"Cannot slice channels [1:4]. Normalized tensor only has {normalized_tensor.shape[0]} channels. Check `self.bands` configuration or `_normalize_tensor` logic.")
            
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        file_path = self.file_paths[idx]
        sample = self._read_and_process_image(file_path)

        if sample is None:
            # If image reading failed, return None. DataLoader's collate_fn should handle this.
            return None 

        # Apply data augmentations/transformations if provided
        if self.transforms is not None:
            sample = self.transforms(sample)

        # Retrieve the oceanographic features for this file_path
        ocean_features = self.file_path_to_features_map.get(file_path)
        if ocean_features is None:
            # This should ideally not happen if self.file_paths is filtered correctly in __init__
            print(f"WARNING: Ocean features not found for '{file_path}'. This sample might be problematic.")
            return None 

        # Handle different model types and transformation outputs
        if self.model_name == "moco-ocean-features": # Adapted to the new model name
            # Expecting a list of two tensors from TwoCropsTransform for MoCo
            if isinstance(sample, list) and len(sample) == 2:
                q_raw, k_raw = sample
                
                # Normalize query and key crops using the dataset's _normalize_tensor method
                # This method will also perform the [1:4, :, :] slicing
                q_normalized = self._normalize_tensor(q_raw)
                k_normalized = self._normalize_tensor(k_raw) 
                
                # Return as a tuple: (query_image, key_image, ocean_features_tensor)
                return (q_normalized, k_normalized, ocean_features)
            
            # Fallback for if transforms somehow returned a single tensor for moco-ocean-features
            elif isinstance(sample, torch.Tensor):
                sample_normalized = self._normalize_tensor(sample)
                # If only one crop, return (image, features)
                return sample_normalized.float(), ocean_features
            else:
                raise TypeError(
                    f"Unexpected type returned by transforms for '{self.model_name}' model: {type(sample)}. "
                    "Expected a list of two Tensors or a single Tensor."
                )
        else: # For other models (e.g., 'mae'), typically only the image is expected
            sample_normalized = self._normalize_tensor(sample)
            # For models like MAE (pre-training), you might only need the image itself
            # The features would be used in a downstream fine-tuning task.
            return sample_normalized.float()