import lmdb
import torch
from torch.utils.data import Dataset
from safetensors.torch import load as safetensor_load

class HydroDataset(Dataset):
    def __init__(self, data_dir: str, bands: list = None, transforms=None):
        """
        Args:
            data_dir: Path to the LMDB database.
            bands: List of band names to use (default is all bands).
            transforms: Optional transformations to apply to the data.
        """
        self.data_dir = data_dir
        self.bands = bands  # List of bands, e.g., ["B01", "B02"] or None for all
        self.transforms = transforms

        # Open LMDB environment
        self.lmdb_env = lmdb.open(self.data_dir, readonly=True, lock=False)

        # Get all keys during initialization
        with self.lmdb_env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Get the key
        key = self.keys[idx]

        # Retrieve the sample from the LMDB
        with self.lmdb_env.begin() as txn:
            data = txn.get(key)

        if data is None:
            raise IndexError(f"Data for key {key} not found in LMDB.")

        # Load safetensor data
        safetensor_dict = safetensor_load(data)

        # Select specific bands if specified
        if self.bands:
            safetensor_dict = {band: safetensor_dict[band] for band in self.bands}

        # Stack bands into a tensor
        tensor = torch.stack([torch.tensor(safetensor_dict[band]) for band in safetensor_dict])

        # Apply transformations if provided
        if self.transforms:
            tensor = self.transforms(tensor)

        return tensor

    def close(self):
        self.lmdb_env.close()
