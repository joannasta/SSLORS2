import torch

from pathlib import Path
from typing import List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

from .hydro_dataset import HydroDataset 

def collate_fn(batch):
    '''Filter out None samples and collate the remaining batch'''
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        raise ValueError("All samples in the batch failed to load.")
    return torch.utils.data.default_collate(batch)

class HydroDataModule(LightningDataModule):
    '''LightningDataModule for Hydro data, creates train/val/test DataLoaders with optional ocean filtering.'''
    def __init__(
        self,
        data_dir: str,
        bands: List[str] = None,
        transform = None,
        model_name="mae",
        num_workers: int = 8, 
        batch_size = 64,
        csv_file_path="/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv",
        limit_files=False
    ):
        '''Store configuration for datasets and loaders.'''
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bands = bands
        self.transform = transform
        self.model_name=model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_file_path=csv_file_path
        self.limit_files=limit_files

    def setup(self, stage=None):
        '''Instantiate HydroDataset objects for train/val/test/predict depending on stage.'''
        if stage == 'fit' or stage is None:
            self.train_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )
            
            self.val_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

        if stage == 'test' or stage is None:
            self.test_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

        if stage == 'predict':
            self.predict_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform,
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers, 
            collate_fn=collate_fn
        )

    def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, 
                collate_fn=collate_fn
            )

    def test_dataloader(self):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, 
                collate_fn=collate_fn
            )
        
 