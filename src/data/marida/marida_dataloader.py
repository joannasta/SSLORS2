import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

import matplotlib.pyplot as plt
# Import your dataset class (ensure this is correctly defined in your project)
from .marida_dataset import MaridaDataset # Update the path as per your project structure
import numpy as np

class MaridaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, root_dir="/faststorage/joanna/marida/MARIDA", transform=None, standardization=None,pretrained_model=None):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform
        self.standardization = standardization
        self.pretrained_model = pretrained_model

    def setup(self, stage=None):
        """Setup dataset for training, validation, or test based on the stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = MaridaDataset(
                root_dir=self.root_dir, mode='train', transform=self.transform,
                standardization=self.standardization, path=self.root_dir,
                pretrained_model=self.pretrained_model
            )
            self.val_dataset = MaridaDataset(
                root_dir=self.root_dir, mode='val', transform=self.transform,
                standardization=self.standardization, path=self.root_dir,
                pretrained_model=self.pretrained_model
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = MaridaDataset(
                root_dir=self.root_dir, mode='test', transform=self.transform,
                standardization=self.standardization, path=self.root_dir,
                pretrained_model=self.pretrained_model
            )
    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

