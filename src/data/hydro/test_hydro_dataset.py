
import torch
import rasterio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from pathlib import Path

NORM_PARAM_DEPTH = {
    "agia_napa": -30.443,
    "puck_lagoon": -11.0
}

# Normalization File Paths MagicBathyNet
NORM_PARAM_PATHS = {
    "agia_napa": "/mnt/storagecube/joanna/MagicBathyNet/agia_napa/norm_param_s2_an.npy",
    "puck_lagoon": "/mnt/storagecube/joanna/MagicBathyNet/puck_lagoon/norm_param_s2_pl.npy"
}

#MARIDA Normalization Parameter
def get_marida_means_and_stds():
    # Pixel-level number of negative/number of positive per class
    pos_weight = torch.Tensor([ 2.65263158, 27.91666667, 11.39285714, 18.82857143,  6.79775281,
            6.46236559,  0.60648148, 27.91666667, 22.13333333,  5.03478261,
        17.26315789, 29.17391304, 16.79487179, 12.88      ,  9.05797101])

    bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
    0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

    bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
    0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')
    return bands_mean, bands_std, pos_weight

def get_Hydro_means_and_stds():
    means = torch.tensor([
        340.76769064, 429.9430203, 614.21682446, 590.23569706,
        950.68368468, 1792.46290469, 2075.46795189, 2218.94553375,
        2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131
    ], dtype=torch.float)

    stds = torch.tensor([
        554.81258967, 572.41639287, 582.87945694, 675.88746967,
        729.89827633, 1096.01480586, 1273.45393088, 1365.45589904,
        1356.13789355, 1302.3292881, 1079.19066363, 818.86747235
    ], dtype=torch.float)
    
    return means, stds

class HydroDataset(Dataset):
    """Hydro dataset with optional ocean-feature filtering and band-wise normalization."""
    def __init__(
        self,
        path_dataset: Path,
        bands=None,
        transforms=None,
        compute_stats: bool = False,
        location: str = "agia_napa",
        limit_files: bool = False,
        csv_file_path: str = "/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv",
    ):
        self.path_dataset = Path(path_dataset)
        all_file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.bands = bands if bands is not None else ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
        self.location = location

        self.band_means, self.band_stds = get_Hydro_means_and_stds()
        self.band_means_marida, self.band_stds_marida, _ = get_marida_means_and_stds()
        self.transforms = transforms
        self.impute_nan = np.tile(self.band_means_marida, (256, 256, 1))
        self.limit_files = limit_files

        # Normalization params MagicBathyNet
        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        self.csv_file_path = Path(csv_file_path)

        # Use Ocean Features and Cluster Label
        if self.limit_files:
            self.file_path_to_csv_row_map = {}
            self.file_paths = []
            self.csv_df = None
            self._load_ocean_features_and_map(all_file_paths)
        else:
            self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))

        print("len self.file_paths", len(self.file_paths))

    def _load_ocean_features_and_map(self, all_file_paths):
        """Keep only dataset TIFFs listed in ocean CSV."""
        self.csv_df = pd.read_csv(self.csv_file_path)
        csv_file_dir_map = {Path(p).resolve(): row for p, row in self.csv_df.set_index('file_dir').iterrows()}
        successful_matches = []

        for file_path in all_file_paths:
            resolved_file_path = file_path.resolve()
            if resolved_file_path in csv_file_dir_map:
                matched_row = csv_file_dir_map[resolved_file_path]
                self.file_path_to_csv_row_map[file_path] = matched_row
                successful_matches.append(file_path)

        self.file_paths = successful_matches
        print(f"Finished direct mapping. Successfully matched {len(self.file_paths)} TIF files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        """Load image, impute NaNs, normalize per setting."""
        file_path = self.file_paths[idx]

        with rasterio.open(file_path) as src:
            # Read requested bands
            sample_data = src.read(list(range(1, len(self.bands) + 1)))
            sample = torch.from_numpy(sample_data.astype(np.float32))

            if len(self.bands) == 11:
                # Impute NaNs with per-band means
                nan_mask = torch.isnan(sample)
                sample[nan_mask] = torch.from_numpy(self.impute_nan.transpose(2, 1, 0))[nan_mask]
                means, stds, _ = get_marida_means_and_stds()
                sample = (sample - means[:11, None, None]) / stds[:11, None, None]
            elif len(self.bands) == 3:
                # Use per-location mean/std
                means, stds = self.norm_param[0][:, np.newaxis, np.newaxis], self.norm_param[1][:, np.newaxis, np.newaxis]
                sample = (sample - means) / stds
            else:
                means, stds = get_Hydro_means_and_stds()
                sample = (sample - means[:, None, None]) / stds[:, None, None]

            if self.transforms is not None:
                sample = self.transforms(sample)

            if len(self.bands) == 12:
                sample = sample[0:12, :, :]
            elif len(self.bands) == 11:
                sample = sample[0:11, :, :]

            return sample.float()