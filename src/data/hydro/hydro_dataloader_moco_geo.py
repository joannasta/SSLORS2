import torch
from torch.utils.data import Dataset, DataLoader, default_collate # Import default_collate
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List
from src.data.hydro.hydro_moco_geo_dataset import HydroMoCoGeoDataset
from pytorch_lightning import LightningDataModule

class HydroMoCoGeoDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, transform=None, model_name="mae", num_geo_clusters=100):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.model_name = model_name
        # Ensure num_geo_clusters is not a tuple
        self.num_geo_clusters = num_geo_clusters
        
    # --- Custom collate_fn to filter out None values ---
    def custom_collate_fn(self, batch):
        # Filter out None values from the batch
        batch = [item for item in batch if item is not None]
        if not batch: # If the batch is empty after filtering, return None or an empty list
            return None # Or raise an error if an empty batch is not desired
        return default_collate(batch)
    # --- End custom collate_fn ---

    def setup(self, stage: Optional[str] = None):
        # Use stage to load data depending on the task
        # For pretraining without splits, all these datasets will point to the same full data.
        if stage == 'fit' or stage is None:
            self.train_dataset = HydroMoCoGeoDataset(
                path_dataset=self.data_dir,
                # Assuming bands and compute_stats are handled by HydroMoCoGeoDataset's __init__
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters
            )
            self.val_dataset = HydroMoCoGeoDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters
            )

        if stage == 'test' or stage is None:
            # Initialize test dataset for evaluation
            self.test_dataset = HydroMoCoGeoDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters
            )

        if stage == 'predict':
            # You can set up a different dataset for prediction if needed
            self.predict_dataset = HydroMoCoGeoDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn # Apply the custom collate_fn
        )

    def val_dataloader(self):
        # Ensure val_dataset is initialized before trying to create DataLoader
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn # Apply the custom collate_fn
            )
        else:
            print("WARNING: Validation dataloader requested but val_dataset not initialized or is None.")
            return None # Return None or raise error depending on your strictness

    def test_dataloader(self):
        # Ensure test_dataset is initialized before trying to create DataLoader
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn # Apply the custom collate_fn
            )
        else:
            print("WARNING: Test dataloader requested but test_dataset not initialized or is None.")
            return None # Return None or raise error depending on your strictness