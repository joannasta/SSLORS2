import torch
from torch.utils.data import Dataset, DataLoader, default_collate,random_split # Import default_collate
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from src.data.hydro.geography_aware.hydro_geography_aware_dataset import HydroGeographyAwareDataset
from pytorch_lightning import LightningDataModule

class HydroGeographyAwareDataModule(LightningDataModule):
    """LightningDataModule for geography-aware SSL on Hydro data."""
    def __init__(
        self, data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        num_geo_clusters=10,
        csv_file_path="/home/joanna/SSLORS2/src/utils/train_geo_labels10.csv",
        limit_files=False,
        seed=42
        
        ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.num_geo_clusters = num_geo_clusters
        self.csv_file_path=csv_file_path
        self.limit_files=limit_files
        self.seed=seed
        

    def custom_collate_fn(self, batch):
        """Filter out None samples and use default_collate; returns None if batch is empty."""
        batch = [item for item in batch if item is not None]
        if not batch:
            return None 
        return default_collate(batch)


    def setup(self, stage: Optional[str] = None):
        """Instantiate datasets for train/val/test/predict depending on stage."""
        base = HydroGeographyAwareDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                num_geo_clusters=self.num_geo_clusters,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
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
            pin_memory=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn 
        )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn 
            )

    def test_dataloader(self):
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn 
            )
