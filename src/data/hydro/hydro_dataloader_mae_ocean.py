import torch
from torch.utils.data import Dataset, DataLoader, default_collate # Import default_collate
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

from src.data.hydro.hydro_dataset_mae_ocean import HydroMaeOceanFeaturesDataset
from pytorch_lightning import LightningDataModule

class HydroMaeOceanFeaturesDataModule(LightningDataModule): 
    """LightningDataModule for MAE + ocean features: builds train/val/test dataloaders."""
    def __init__(
            self, 
            data_dir: str, 
            batch_size: int = 32, 
            num_workers: int = 4, 
            transform: Optional[Callable] = None, 
            model_name: str = "mae_ocean", 
            csv_file_path="/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv",
            limit_files=False
            
        ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.model_name = model_name
        self.csv_file_path=csv_file_path
        self.limit_files=limit_files
        
    def custom_collate_fn(self, batch: List[Any]): 
        """Collate function that filters out None samples"""
        batch = [item for item in batch if item is not None]
        if not batch: 
            print("Warning: Batch is empty after filtering None values. This batch will be skipped.")
            return None 
        return default_collate(batch)


    def setup(self, stage: Optional[str] = None):
        """Instantiate datasets for the given stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = HydroMaeOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

            self.val_dataset = HydroMaeOceanFeaturesDataset(
                path_dataset=self.data_dir, 
                transforms=self.transform,
                model_name=self.model_name,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

        if stage == 'test' or stage is None:
            self.test_dataset = HydroMaeOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

        if stage == 'predict':
            self.predict_dataset = HydroMaeOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

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
        else:
            print("WARNING: Validation dataloader requested but val_dataset not initialized or is None. Returning None.")
            return None 

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
        else:
            print("WARNING: Test dataloader requested but test_dataset not initialized or is None. Returning None.")
            return None