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
    def __init__( 
        self, 
        data_dir: str, 
        bands: List[str] = None, 
        transform = None, 
        batch_size = 64, 
        model_name = "moco",
        num_workers: int = 8, 
    ): 
        super().__init__() 
        self.data_dir = Path(data_dir) 
        self.bands = bands 
        self.transform = transform 
        self.batch_size = batch_size 
        self.model_name = model_name 
        self.num_workers = num_workers

    def setup(self, stage=None): 
        # Use stage to load data depending on the task 
        if stage == 'fit' or stage is None: 
            self.train_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                compute_stats=False, 
                transform=self.transform, 
                model_name = self.model_name 
            ) 
            self.val_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                compute_stats=False, 
                transform=self.transform, 
                model_name = self.model_name 
            ) 

        if stage == 'test' or stage is None: 
            # Initialize test dataset for evaluation 
            self.test_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                compute_stats=False, 
                transform=self.transform, 
                model_name = self.model_name 
            ) 

        if stage == 'predict': 
            # You can set up a different dataset for prediction if needed 
            self.predict_dataset = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                compute_stats=False, 
                transform=self.transform, 
                model_name = self.model_name 
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
                pin_memory=True, drop_last=True 
            ) 
        else: 
            raise ValueError("Validation dataset is None.") 

    def test_dataloader(self): 
        if self.test_dataset is not None: 
            return DataLoader( 
                self.test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers, 
                collate_fn=collate_fn, 
                pin_memory=True, drop_last=True) 
        else: 
            raise ValueError("Test dataset is None.")