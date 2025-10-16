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
        limit_files=False,
        seed: int = 42,
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
        self.seed=seed

    def setup(self, stage=None): 
        """Instantiate train/val/test/predict datasets depending on stage."""
        if stage == 'fit' or stage is None: 
            base = HydroMoCoDataset( 
                path_dataset=self.data_dir, 
                bands=self.bands, 
                transform=self.transform, 
                csv_file_path=self.csv_file_path,
                limit_files=self.limit_files
            ) 
            n = len(base)
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
