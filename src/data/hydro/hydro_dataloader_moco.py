import torch 

from pathlib import Path 
from typing import List 
from pytorch_lightning import LightningDataModule 
from torch.utils.data import DataLoader, random_split 
from torchvision import transforms as T 

from .hydro_moco_dataset import HydroMoCoDataset 

def collate_fn(batch): 
    """Puts each data field into a tensor with outer dimension batch size""" 
    img1_batch = [] 
    img2_batch = [] 
    for img1, img2 in batch: 
        img1_batch.append(img1) 
        img2_batch.append(img2) 
    img1_batch = torch.stack(img1_batch) 
    img2_batch = torch.stack(img2_batch) 
    return (img1_batch, img2_batch) 


class HydroMoCoDataModule(LightningDataModule):
    """LightningDataModule for MoCo-style SSL on Hydro data."""
    def __init__( 
        self, 
        data_dir: str, 
        bands: List[str] = None, 
        transform = None, 
        batch_size = 64, 
        model_name = "moco",
        num_workers: int = 8, 
        csv_file_path="/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv",
        limit_files=False
    ): 
        super().__init__() 
        self.data_dir = Path(data_dir) 
        self.bands = bands 
        self.transform = transform 
        self.batch_size = batch_size 
        self.model_name = model_name 
        self.num_workers = num_workers
        self.csv_file_path=csv_file_path
        self.limit_files=limit_files

    def setup(self, stage=None): 
        """Instantiate train/val/test/predict datasets depending on stage."""
        if stage == 'fit' or stage is None: 
            self.train_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                transform=self.transform, 
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            ) 
            self.val_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                transform=self.transform, 
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            ) 

        if stage == 'test' or stage is None: 
            self.test_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                transform=self.transform, 
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            ) 

        if stage == 'predict': 
            self.predict_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                transform=self.transform, 
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            ) 

    def train_dataloader(self): 
        return DataLoader( 
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=collate_fn, 
            pin_memory=True, drop_last=True) 

    def val_dataloader(self): 
        if self.val_dataset is not None: 
            return DataLoader( 
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers, 
                collate_fn=collate_fn, 
                pin_memory=True, drop_last=False
            ) 

    def test_dataloader(self): 
            return DataLoader( 
                self.test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers, 
                collate_fn=collate_fn, 
                pin_memory=True, drop_last=False) 