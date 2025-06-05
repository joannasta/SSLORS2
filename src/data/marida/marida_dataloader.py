import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

import matplotlib.pyplot as plt
# Import your dataset class (ensure this is correctly defined in your project)
from .marida_dataset import MaridaDataset # Update the path as per your project structure
import numpy as np

class MaridaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, root_dir="/faststorage/joanna/marida/MARIDA", transform=None, standardization=None):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform
        self.standardization = standardization

    def setup(self, stage=None):
        """Setup dataset for training, validation, or test."""
        self.train_dataset = MaridaDataset(mode='train', transform=self.transform, standardization=self.standardization, path=self.root_dir)
        self.val_dataset = MaridaDataset(mode='val', transform=self.transform, standardization=self.standardization, path=self.root_dir)
        self.test_dataset = MaridaDataset(mode='test', transform=self.transform, standardization=self.standardization, path=self.root_dir)

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

