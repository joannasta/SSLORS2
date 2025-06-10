
# --- HydroMoCoGeoDataset - Modified to load both Image and Tabular Data ---
class HydroMoCoGeoDataset(Dataset):
    def __init__(
        self,
        path_dataset: Path, # Path to folder containing .tif files
        bands: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        location: str = "agia_napa",
        model_name: str = "moco-geo", # Ensure this is "moco-geo" for this setup
        csv_path: str = "/home/joanna/SSLORS/src/data/hydro/train_geo_labels10_projected.csv",
        num_geo_clusters: int = 100 # This parameter is now used by MoCoGeo
    ):
        self.path_dataset = Path(path_dataset)
        self.file_paths = sorted(list(self.path_dataset.glob("*.tif")))
        self.num_geo_clusters = num_geo_clusters
        # Default bands if not provided (assuming Sentinel-2 bands for example)
        self.bands = bands if bands is not None else [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
        ]
        self.transforms = transforms # Expects a TwoCropsTransform for MoCo
        self.model_name = model_name 
        self.location = location 
        self.csv_path = Path(csv_path)

        # Load normalization parameters upfront
        self._load_normalization_params()

        # Dictionary to store geo-labels and ocean features linked by file_path
        self.data_lookup = {}
        if self.model_name == "moco-geo":
            self._load_geo_labels_and_ocean_features()

    def _load_normalization_params(self):
        """Pre-computes and stores normalization tensors."""
        # This part depends heavily on your specific `config.py` and band structure.
        # Ensure 'channels' in `src_channels` for MoCoGeo matches the output of `_normalize_tensor`.
        if len(self.bands) == 11: # Example for MARIDA
            means_np, stds_np, _ = get_marida_means_and_stds()
            self.means_tensor = torch.tensor(means_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np[:11], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        elif len(self.bands) == 3: # Example for RGB
            means_np, stds_np = np.load(NORM_PARAM_PATHS[self.location])[0], np.load(NORM_PARAM_PATHS[self.location])[1]
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        else: # Generic case, e.g., for 12 bands from Sentinel-2
            means_np, stds_np = get_means_and_stds()
            self.means_tensor = torch.tensor(means_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.stds_tensor = torch.tensor(stds_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    def _load_geo_labels_and_ocean_features(self):
        """Loads geo-labels and ocean features from CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Required geo-labels and ocean features CSV not found at '{self.csv_path}'. "
                "Please ensure the CSV file is in the correct location and contains 'file_dir', 'label', "
                "'lat', 'lon', 'bathy', 'chlorophyll', 'secchi' columns."
            )
        try:
            df = pd.read_csv(self.csv_path)
            # Ensure required columns are present
            required_cols = ['file_dir', 'label', 'lat', 'lon', 'bathy', 'chlorophyll', 'secchi']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"CSV must contain all of {required_cols}. Found: {df.columns.tolist()}"
                )
            
            for _, row in df.iterrows():
                file_dir_key = str(Path(row['file_dir'])) # Ensure consistent path format
                self.data_lookup[file_dir_key] = {
                    'label': row['label'],
                    'ocean_features': torch.tensor([
                        float(row['lat']), float(row['lon']), 
                        float(row['bathy']), float(row['chlorophyll']), float(row['secchi'])
                    ], dtype=torch.float32)
                }
            print(f"Successfully loaded {len(self.data_lookup)} geo-labels and ocean features from '{self.csv_path}'.")
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from '{self.csv_path}': {e}. "
                "Please ensure the CSV format is correct (label,file_dir,lat,lon,bathy,chlorophyll,secchi)."
            ) from e

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def _read_and_process_image(self, file_path: Path) -> Optional[torch.Tensor]:
        """Reads a TIFF image and performs initial processing."""
        try:
            with rasterio.open(file_path) as src:
                bands_data = []
                # Read only the bands expected by src_channels in MoCoGeo
                # For this example, we assume 3 bands will be used after normalization/selection
                # If your TIFF has 12 bands and you want to use 3, `_normalize_tensor` handles the selection.
                # If src_channels in MoCoGeo is 3, make sure src.count is at least 3.
                for i in range(1, src.count + 1): # Read all available bands first
                    band_data = src.read(i)
                    band_tensor = torch.from_numpy(band_data.astype(np.float32)).unsqueeze(0)
                    bands_data.append(band_tensor)
                sample = torch.cat(bands_data, dim=0).contiguous()

                # NaN handling for 11-band MARIDA data (as per your original code)
                if len(self.bands) == 11: # This check might need adjustment based on src.count vs self.bands
                    nan_mask = torch.isnan(sample)
                    if torch.any(nan_mask):
                        means, _, _ = get_marida_means_and_stds()
                        # Impute per channel using its mean
                        impute_mean_tensor = torch.tensor(means.astype(np.float32), dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
                        # Expand mean to match shape of current band if necessary
                        for b_idx in range(sample.shape[0]):
                             sample[b_idx][nan_mask[b_idx]] = impute_mean_tensor[b_idx].item()
            return sample
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening image file '{file_path}': {e}. This sample will be skipped.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during image reading for '{file_path}': {e}. This sample will be skipped.")
            return None

    def _normalize_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies normalization (mean/std) to a tensor and selects channels 1:4 (B02, B03, B04).
        This assumes input `img_tensor` has at least 4 channels.
        The output will have 3 channels, matching `src_channels=3` in MoCoGeo.
        """
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor for normalization, but got {type(img_tensor)}")
        
        # Ensure means_tensor and stds_tensor match the number of channels of img_tensor
        # If your means/stds are for 12 bands, and img_tensor has 12 bands, normalize all 12 first.
        # Then select the desired channels.
        if img_tensor.shape[0] != self.means_tensor.shape[0]:
            # This is a common point of error if means_tensor isn't broad enough or matches.
            # For simplicity, if means/stds are for 12 bands, assume img_tensor also has 12.
            # If not, you'd need more sophisticated handling or specific means/stds per band selection.
            print(f"Warning: img_tensor channels ({img_tensor.shape[0]}) mismatch normalization params ({self.means_tensor.shape[0]}). Normalizing all available and then selecting.")
        
        normalized_tensor = (img_tensor - self.means_tensor[:img_tensor.shape[0]]) / self.stds_tensor[:img_tensor.shape[0]]
        
        # Select channels from 1 to 3 (inclusive), which is 1:4 in Python slicing
        # This typically corresponds to Blue, Green, Red for Sentinel-2 (B02, B03, B04)
        # Ensure that these channels exist in your actual TIFF files after reading.
        if normalized_tensor.shape[0] < 4:
            raise ValueError(f"Not enough channels in normalized image to select channels 1:4. Has {normalized_tensor.shape[0]} channels.")
        return normalized_tensor[1:4, :, :] # This makes the output 3 channels

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        sample_image = self._read_and_process_image(file_path)

        if sample_image is None:
            # Handle problematic samples by returning a dummy or filtering in DataLoader
            # For this example, we'll return None and handle it in collate_fn
            return None 

        # Retrieve geo-label and ocean features
        data_info = self.data_lookup.get(str(file_path))
        if data_info is None:
            # If no data found, return None (or raise error)
            print(f"WARNING: No geo-label or ocean features found for '{file_path}' in CSV. Skipping this sample.")
            return None
        
        pseudo_label = data_info['label']
        ocean_features = data_info['ocean_features']

        # Apply data augmentations/transformations if provided (for MoCo, TwoCropsTransform)
        if self.transforms is not None:
            # self.transforms should return a list of two augmented image tensors
            augmented_images = self.transforms(sample_image)
            q_raw_image, k_raw_image = augmented_images
        else:
            # If no transforms, use the same raw image for query and key (no augmentation)
            q_raw_image, k_raw_image = sample_image, sample_image

        # Normalize query and key crops
        q_normalized_image = self._normalize_tensor(q_raw_image)
        k_normalized_image = self._normalize_tensor(k_raw_image)

        # Return as a tuple: (query_image, key_image, tabular_features, geo_label)
        # Ensure all tensors are float and labels are long
        return (q_normalized_image.float(), 
                k_normalized_image.float(), 
                ocean_features.float(), 
                torch.tensor(pseudo_label, dtype=torch.long))

# --- Custom Collate Function for DataLoader ---
# This is crucial to handle `None` values returned by __getitem__
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# --- Example TwoCropsTransform for MoCo ---
class TwoCropsTransform:
    """
    Takes an image, applies a set of transformations twice, returning two augmented crops.
    Used for self-supervised learning methods like MoCo.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return [q, k]
