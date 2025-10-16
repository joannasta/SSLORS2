import torch
from pathlib import Path
from typing import List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

# Adjust this import to your structure, e.g.:
# from .hydro_dataset import HydroDataset
from test_hydro_dataset import HydroDataset

def collate_fn(batch):
    """Filter out None samples and collate the remaining batch."""
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        raise ValueError("All samples in the batch failed to load.")
    return torch.utils.data.default_collate(batch)

class HydroDataModule(LightningDataModule):
    """Hydro DataModule with 80/20 train/val split."""
    def __init__(
        self,
        data_dir: str,
        bands: List[str] = None,
        transform=None,
        num_workers: int = 8,
        batch_size: int = 64,
        csv_file_path: str = "/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv",
        limit_files: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bands = bands
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_file_path = csv_file_path
        self.limit_files = limit_files
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Build one base dataset (limit_files will filter via your CSV if True)
        base = HydroDataset(
            path_dataset=self.data_dir,
            bands=self.bands,
            compute_stats=False,
            transforms=self.transform,
            csv_file_path=self.csv_file_path,
            limit_files=self.limit_files,
        )

        # Create 80/20 split for fit
        if stage == "fit":
            n = len(base)
            if n < 2:
                raise ValueError(f"Not enough samples to split: {n}")
            n_train = int(0.8 * n)
            n_val = n - n_train
            gen = torch.Generator()
            self.train_dataset, self.val_dataset = random_split(base, [n_train, n_val], generator=gen)
            
            print("train dataset len",len(self.train_dataset))
            print("val dataset len",len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

if __name__ == "__main__":
    # Example usage: update data_dir and any args you need
    dm = HydroDataModule(
        data_dir="/mnt/storagecube/joanna/Hydro/",
        bands=None,            # or a list like ["B02","B03","B04"]
        transform=None,
        num_workers=8,
        batch_size=64,
        csv_file_path="/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv",
        limit_files=False,     # set True to filter files using the CSV
        seed=42,
    )

    dm.setup(stage="fit")

    # Inspect which files are in train vs val
    train_paths = [dm.train_dataset.dataset.file_paths[i] for i in dm.train_dataset.indices]
    val_paths   = [dm.val_dataset.dataset.file_paths[i]   for i in dm.val_dataset.indices]

    print("Train count:", len(train_paths))
    print("Val count:", len(val_paths))
    print("/n")
    print("First few train files:", train_paths[:5])
    print("First few val files:", val_paths[:5])
    print("/n")

    # Try loading a batch to confirm shapes
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("Train batch shape:", batch.shape)