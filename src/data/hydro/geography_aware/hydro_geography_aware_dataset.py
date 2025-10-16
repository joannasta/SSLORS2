import torch
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Callable

from torch.utils.data import Dataset
from torchvision import transforms as T

from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_Hydro_means_and_stds, get_marida_means_and_stds

class HydroGeographyAwareDataset(Dataset):
    """Geography-aware SSL dataset: returns two normalized views (q, k) and a pseudo geo label."""
    def __init__(
            self,
            path_dataset: Path,
            bands: Optional[List[str]] = None,
            transforms: Optional[Callable] = None,
            location: str = "agia_napa",
            csv_file_path: str = "/home/joanna/SSLORS2/src/utils/train_geo_labels10.csv",
            num_geo_clusters: int = 10,
            limit_files=False
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.num_geo_clusters = num_geo_clusters
        self.limit_files=limit_files
        
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        
        self.transforms = transforms
        self.location = location 
        self.csv_file_path = Path(csv_file_path)

        self._load_normalization_params()
        self.geo_to_label = {}
        
        
        self.file_path_to_csv_row_map = {}
        
        # Map TIFFs to CSV rows
        initial_file_paths = [] 
        if self.limit_files:
            csv_df = pd.read_csv(self.csv_file_path)
            csv_file_dir_map = {Path(p).resolve(): row for p, row in csv_df.set_index('file_dir').iterrows()}
            
            for file_path in all_file_paths:
                resolved_file_path = file_path.resolve()
                if resolved_file_path in csv_file_dir_map:
                    self.file_path_to_csv_row_map[file_path] = csv_file_dir_map[resolved_file_path]
                    initial_file_paths.append(file_path)
        else:
            initial_file_paths = all_file_paths

        # Filter by availability of geo labels 
        self._load_geo_labels()
        self.file_paths = [
                fp for fp in initial_file_paths if fp.resolve() in self.geo_to_label
            ]

    def _load_normalization_params(self):
        """Load per-band mean/std for normalization."""
        if len(self.bands) == 11:
            means_np, stds_np, _ = get_marida_means_and_stds()
            self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).clone().detach()
            self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).clone().detach()
        elif len(self.bands) == 3:
            means_np, stds_np = np.load(NORM_PARAM_PATHS[self.location])[0], np.load(NORM_PARAM_PATHS[self.location])[1]
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).clone().detach()
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).clone().detach()
        else:
            means_np, stds_np = get_Hydro_means_and_stds()
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).clone().detach()
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).clone().detach()

    def _load_geo_labels(self):
        """Load pseudo geo labels from CSV."""
        df = pd.read_csv(self.csv_file_path)
        self.geo_to_label = {Path(row['file_dir']).resolve(): row['label'] for _, row in df.iterrows()}
            
    def __len__(self) -> int:
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> torch.Tensor:
        """Read TIFF bands and impute NaNs."""
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
        """Per-band normalization; return RGB (B02,B03,B04) if available, else all bands."""
        normalized_tensor = (img_tensor - self.means_tensor) / self.stds_tensor
        return normalized_tensor[1:4, :, :]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return two normalized views and a pseudo label."""
        file_path = self.file_paths[idx]
        sample = self._read_and_process_image(file_path)

        # Make two views
        if self.transforms is not None:
            sample = self.transforms(sample)

        q_raw, k_raw = sample

        q_normalized = self._normalize_tensor(q_raw)
        k_normalized = self._normalize_tensor(k_raw)
        
        # Geo label
        resolved_file_path = file_path.resolve()
        
        pseudo_label = self.geo_to_label.get(resolved_file_path) 
        
        return q_normalized, k_normalized, torch.tensor(pseudo_label, dtype=torch.long)