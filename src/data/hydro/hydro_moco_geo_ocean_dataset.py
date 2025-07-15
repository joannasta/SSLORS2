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
from scipy.spatial import KDTree
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
        model_name: str = "moco-ocean-features",
        csv_features_path: str = "/home/joanna/SSLORS/src/utils/train_ocean_labels_3_clusters.csv",
        max_match_distance: float = 0.001
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        print(f"DEBUG: Found {len(all_file_paths)} TIF files initially.")
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        self.transforms = transforms
        self.model_name = model_name
        self.location = location
        self.csv_features_path = Path(csv_features_path)
        self.max_match_distance = max_match_distance

        self._load_normalization_params()

        self.file_path_to_csv_index_map = {}
        self.file_paths = []

        self.csv_df = None
        self.csv_kdtree = None

        self._load_ocean_features_and_map(all_file_paths)

        if not self.file_paths and self.csv_features_path.exists():
            raise ValueError(
                "No TIF files could be matched to ocean features in the CSV. "
                "Check TIF file validity, CSV data, `max_match_distance`, or if there are TIF files at all."
            )
        print(f"DEBUG: Dataset initialization complete. Total matched files: {len(self.file_paths)}")


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
        print(f"DEBUG: Starting _load_ocean_features_and_map. CSV path: {self.csv_features_path}")
        if not self.csv_features_path.exists():
            raise FileNotFoundError(
                f"Ocean features CSV not found at '{self.csv_features_path}'. "
                "Please ensure the CSV file is in the correct location and name."
            )
        try:
            self.csv_df = pd.read_csv(self.csv_features_path)
            print(f"DEBUG: Loaded CSV with {len(self.csv_df)} rows.")
            print(f"DEBUG: Memory usage of self.csv_df: {self.csv_df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

            csv_coords = self.csv_df[['lat', 'lon']].values.astype(np.float32)
            self.csv_kdtree = KDTree(csv_coords)
            print(f"DEBUG: Built KDTree from CSV coordinates.")
            print(f"DEBUG: Memory usage of KDTree data (csv_coords): {csv_coords.nbytes / (1024**3):.2f} GB")

            successful_matches = []
            print(f"DEBUG: Starting matching process for {len(all_file_paths)} TIF files.")
            
            for i, file_path in enumerate(all_file_paths):
                if i < 10 or i % 1000 == 0: 
                    print(f"DEBUG: Processing TIF file {i}/{len(all_file_paths)}: {file_path.name}")
                try:
                    with rasterio.open(file_path) as src:
                        if i < 10 or i % 1000 == 0:
                            print(f"  DEBUG: TIF CRS for {file_path.name}: {src.crs}")

                        left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                        
                        native_center_lon = (left + right) / 2
                        native_center_lat = (bottom + top) / 2

                        transformed_lon, transformed_lat = rasterio_transform(
                            src.crs,
                            'EPSG:4326',
                            [native_center_lon],
                            [native_center_lat]
                        )
                        tif_center_lon = transformed_lon[0]
                        tif_center_lat = transformed_lat[0]

                        query_point = np.array([tif_center_lat, tif_center_lon], dtype=np.float32)

                        if i < 10 or i % 1000 == 0:
                            print(f"  DEBUG: Transformed TIF center (lat, lon): ({tif_center_lat:.6f}, {tif_center_lon:.6f})")

                        distance, index = self.csv_kdtree.query(query_point, k=1, distance_upper_bound=self.max_match_distance)

                        if distance <= self.max_match_distance and index != self.csv_kdtree.n:
                            self.file_path_to_csv_index_map[file_path] = index
                            successful_matches.append(file_path)
                            if i < 10 or i % 1000 == 0:
                                print(f"  DEBUG: MATCH FOUND! Distance: {distance:.6f}, CSV Index: {index}")
                        else:
                            if i < 10 or i % 1000 == 0:
                                print(f"  DEBUG: NO MATCH! Distance: {distance:.6f} (vs max {self.max_match_distance}), Index: {index} (Index 'n' means no match within distance)")
                                if index != self.csv_kdtree.n:
                                    nearest_csv_lat = self.csv_df.iloc[index]['lat']
                                    nearest_csv_lon = self.csv_df.iloc[index]['lon']
                                    print(f"    DEBUG: Nearest CSV point found (but too far): ({nearest_csv_lat:.6f}, {nearest_csv_lon:.6f})")
                                else:
                                    print(f"    DEBUG: No nearest CSV point found within KDTree search space (likely means beyond max distance entirely).")

                except rasterio.errors.RasterioIOError as e:
                    print(f"ERROR: Error reading TIF file {file_path}: {e}. Skipping this file.")
                except Exception as e:
                    print(f"ERROR: An unexpected error occurred while processing {file_path}: {e}. Skipping this file.")
            
            self.file_paths = successful_matches
            print(f"DEBUG: Finished matching and mapping. Successfully matched {len(self.file_paths)} TIF files.")
            print(f"DEBUG: Memory usage of file_path_to_csv_index_map (approx): {sys.getsizeof(self.file_path_to_csv_index_map) / (1024**2):.2f} MB")

        except Exception as e:
            raise RuntimeError(
                f"Error loading or processing ocean features from '{self.csv_features_path}': {e}. "
                "Please ensure the CSV format (lat,lon,bathy,chlorophyll,secchi) is correct and 'scipy' is installed."
            ) from e

    def __len__(self) -> int:
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> Optional[torch.Tensor]:
        try:
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
        except rasterio.errors.RasterioIOError as e:
            return None
        except Exception as e:
            return None

    def _normalize_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor for normalization, but got {type(img_tensor)}")
        if img_tensor.shape[0] != self.means_tensor.shape[0]:
            pass

        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        if normalized_tensor.shape[0] < 4:
            raise ValueError(f"Cannot slice channels [1:4]. Normalized tensor only has {normalized_tensor.shape[0]} channels. Check `self.bands` configuration or `_normalize_tensor` logic.")
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        file_path = self.file_paths[idx]
        sample = self._read_and_process_image(file_path)

        if sample is None:
            return None

        if self.transforms is not None:
            sample = self.transforms(sample)

        csv_row_index = self.file_path_to_csv_index_map.get(file_path)
        if csv_row_index is None:
            return None

        try:
            features_row = self.csv_df.iloc[csv_row_index]
            ocean_features = torch.tensor([
                features_row['bathy'],
                features_row['chlorophyll'],
                features_row['secchi']
            ], dtype=torch.float32)
        except KeyError as e:
            raise KeyError(f"Missing expected ocean feature column in CSV at index {csv_row_index} for file {file_path}: {e}")
        except Exception as e:
            return None

        # Simplified logic for __getitem__ based on model_name
        if self.model_name == "moco-geo-ocean":
            # For MoCo, 'sample' MUST be a list of two Tensors after transforms.
            # If not, the transforms are configured incorrectly.
            if not (isinstance(sample, list) and len(sample) == 2 and 
                    isinstance(sample[0], torch.Tensor) and isinstance(sample[1], torch.Tensor)):
                raise TypeError(
                    f"For 'moco-ocean-features' model, transforms must return a list of two Tensors, "
                    f"but got type {type(sample)} with content {sample}"
                )
            
            q_raw, k_raw = sample
            q_normalized = self._normalize_tensor(q_raw)
            k_normalized = self._normalize_tensor(k_raw)
            return (q_normalized, k_normalized, ocean_features)
        else:
            # For other models, 'sample' should be a single Tensor after transforms.
            if not isinstance(sample, torch.Tensor):
                raise TypeError(
                    f"For '{self.model_name}' model, transforms must return a single Tensor, "
                    f"but got type {type(sample)}"
                )
            sample_normalized = self._normalize_tensor(sample)
            return sample_normalized.float(), ocean_features