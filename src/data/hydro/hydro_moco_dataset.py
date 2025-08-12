import torch
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

from typing import Optional, List, Callable, Tuple
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import get_means_and_stds, get_marida_means_and_stds, NORM_PARAM_DEPTH, NORM_PARAM_PATHS

class HydroMoCoDataset(Dataset):
    def __init__(
            self, path_dataset: Path, bands: List[str] = None,
            location="agia_napa", transform: Optional[Callable] = None,
            csv_features_path: str = "/home/joanna/SSLORS2/src/utils/train_ocean_labels_3_clusters_correct.csv"):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands
        self.location = location
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]

        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        self.transform = transform
        
        self.csv_features_path = Path(csv_features_path)
        
        self.file_path_to_csv_row_map = {}
        self.file_paths = []
        self._load_ocean_features_and_map(all_file_paths)
            
    def _load_ocean_features_and_map(self, all_file_paths: List[Path]):
        csv_df = pd.read_csv(self.csv_features_path)
        csv_file_dir_map = {Path(p).resolve(): row for p, row in csv_df.set_index('file_dir').iterrows()}
        
        for file_path in all_file_paths:
            resolved_file_path = file_path.resolve()
            if resolved_file_path in csv_file_dir_map:
                self.file_path_to_csv_row_map[file_path] = csv_file_dir_map[resolved_file_path]
                self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def _load_and_preprocess(self, file_path: Path) -> torch.Tensor:
        with rasterio.open(file_path) as src:
            band_indices = list(range(1, len(self.bands) + 1))
            sample_data = src.read(band_indices)
            sample = torch.from_numpy(sample_data.astype(np.float32))

            if len(self.bands) == 11:
                nan_mask = torch.isnan(sample)
                means, _, _ = get_marida_means_and_stds()
                impute_nan = np.tile(means.numpy(), (sample.shape[1], sample.shape[2], 1)).transpose(2, 0, 1)
                sample[nan_mask] = torch.from_numpy(impute_nan)[nan_mask]
            
            return sample.float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        img = self._load_and_preprocess(file_path)

        img_q, img_k = self.transform(img) 

        if len(self.bands) == 11:
            means, stds, _ = get_marida_means_and_stds()
            means_tensor = means[:11, None, None]
            stds_tensor = stds[:11, None, None]
            img_q = (img_q - means_tensor) / stds_tensor
            img_k = (img_k - means_tensor) / stds_tensor
        elif len(self.bands) == 3:
            means, stds = self.norm_param[0][:, None, None], self.norm_param[1][:, None, None]
            means_tensor = torch.tensor(means, dtype=torch.float32)
            stds_tensor = torch.tensor(stds, dtype=torch.float32)
            img_q = (img_q - means_tensor) / stds_tensor
            img_k = (img_k - means_tensor) / stds_tensor
        else:
            means, stds = get_means_and_stds()
            means_tensor = means[:, None, None]
            stds_tensor = stds[:, None, None]
            img_q = (img_q - means_tensor) / stds_tensor
            img_k = (img_k - means_tensor) / stds_tensor
            img_q = img_q[1:4, :, :]
            img_k = img_k[1:4, :, :]

        return img_q, img_k