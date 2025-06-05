
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List
from src.data.hydro.hydro_moco_geo_dataset import HydroDataset
from pytorch_lightning import LightningDataModule

class HydroDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, transform=None, model_name="mae", num_geo_clusters=100):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.model_name = model_name
        self.num_geo_clusters = num_geo_clusters
        self.train_dataset: Optional[HydroDataset] = None
        self.val_dataset: Optional[HydroDataset] = None
        self.test_dataset: Optional[HydroDataset] = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = HydroDataset(
            path_dataset=self.data_dir / "train",
            transforms=self.transform,
            model_name=self.model_name,
            num_geo_clusters=self.num_geo_clusters
        )
        self.val_dataset = HydroDataset(
            path_dataset=self.data_dir / "val",
            transforms=None,
            model_name=self.model_name,
            num_geo_clusters=self.num_geo_clusters
        )
        self.test_dataset = HydroDataset(
            path_dataset=self.data_dir / "test",
            transforms=None,
            model_name=self.model_name,
            num_geo_clusters=self.num_geo_clusters
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True) # Added drop_last=True

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)