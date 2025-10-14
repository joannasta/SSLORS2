import torch
import pytorch_lightning as pl
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from config import get_marida_means_and_stds

from .marida_dataset import MaridaDataset 


class MaridaDataModule(pl.LightningDataModule):
    """LightningDataModule for MARIDA: creates train/val/test datasets and dataloaders."""
    def __init__(self, batch_size=32, root_dir="/home/jovyan/SSLORS/marida", transform=None, standardization=None,
                 full_finetune=True, random=False, ssl=False, pretrained_model=None,model_type='mae'):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.path ="/mnt/storagecube/joanna/MARIDA"
        self.transform = transform
        self.standardization = standardization
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        
    def setup(self, stage=None):
        """Instantiate datasets for train/val/test splits."""
        self.train_dataset = MaridaDataset(root_dir=self.root_dir,mode='train', transform=self.transform, standardization=self.standardization, path=self.path,full_finetune=self.full_finetune, random=self.random, ssl=self.ssl, pretrained_model=self.pretrained_model,model_type=self.model_type)
        self.val_dataset = MaridaDataset(root_dir=self.root_dir, mode='val', transform=self.transform, standardization=self.standardization, path=self.path,full_finetune=self.full_finetune, random=self.random, ssl=self.ssl,pretrained_model=self.pretrained_model,model_type=self.model_type)
        self.test_dataset = MaridaDataset(root_dir=self.root_dir, mode='test', transform=self.transform, standardization=self.standardization, path=self.path,full_finetune=self.full_finetune, random=self.random, ssl=self.ssl, pretrained_model=self.pretrained_model,model_type=self.model_type)

    def train_dataloader(self):
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4)

    def val_dataloader(self):
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4)

    def test_dataloader(self):
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4)
