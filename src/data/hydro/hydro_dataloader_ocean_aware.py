import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

from src.data.hydro.hydro_moco_geo_ocean_dataset import HydroOceanAwareDataset 
from pytorch_lightning import LightningDataModule

class HydroOceanAwareDataModule(LightningDataModule): 
    """LightningDataModule for ocean-aware Hydro datasets: prepares train/val/test loaders."""
    def __init__(
            self, 
            data_dir: str, 
            batch_size: int = 32, 
            num_workers: int = 4, 
            transform: Optional[Callable] = None, 
            model_name: str = "ocean_aware", 
            csv_features_path: str = "/home/joanna/SSLORS2/src/utils/ocean_features/train_ocean_labels_3_clusters.csv",
            num_geo_clusters=3,
            ocean_flag = True
            
        ):
        """Store configuration and paths for datasets and dataloaders."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.model_name = model_name
        self.num_geo_clusters = num_geo_clusters
        self.csv_features_path = csv_features_path
        self.ocean_flag=ocean_flag
        
    def custom_collate_fn(self, batch: List[Any]): 
        """Filter out None samples and collate, raise error if batch ends up empty."""
        batch = [item for item in batch if item is not None]
        if not batch: 
            return None 
        return default_collate(batch)


    def setup(self, stage: Optional[str] = None):
        """Instantiate datasets for train/val/test/predict depending on stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters,
                csv_features_path=self.csv_features_path,
                ocean_flag=self.ocean_flag
            )

            self.val_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir, 
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters,
                csv_features_path=self.csv_features_path,
                ocean_flag=self.ocean_flag
            )

        if stage == 'test' or stage is None:
            self.test_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters,
                csv_features_path=self.csv_features_path,
                ocean_flag=self.ocean_flag
            )

        if stage == 'predict':
            self.predict_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                num_geo_clusters=self.num_geo_clusters,
                csv_features_path=self.csv_features_path,
                ocean_flag=self.ocean_flag
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
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn 
            )