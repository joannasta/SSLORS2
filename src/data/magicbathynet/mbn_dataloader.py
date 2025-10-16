import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .mbn_dataset import MagicBathyNetDataset
import os
import random
import torch
import numpy as np

class MagicBathyNetDataModule(pl.LightningDataModule):
    """LightningDataModule for MagicBathyNet, creates train/val/test datasets and dataloaders."""
    def __init__(self, root_dir, batch_size=1, transform=None, cache=True,model_name="mae", pretrained_model=None, location="agia_napa",
                 full_finetune=False, random=False, ssl=True):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.cache = cache
        self.model_name=model_name
        self.pretrained_model = pretrained_model
        self.location = location
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl
        print("Dataloader location:",location)

    def setup(self, stage=None):
        """Instantiate datasets for train/val/test splits."""
        if stage == 'fit' or stage is None:
            self.train_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='train',
                cache=self.cache,
                model_name=self.model_name,
                pretrained_model=self.pretrained_model,
                location=self.location
            )
            
            self.val_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='val',
                cache=self.cache,
                model_name=self.model_name,
                pretrained_model=self.pretrained_model,
                location=self.location
            )

        if stage == 'test' or stage is None: #'test'
            self.test_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='test',
                cache=self.cache,
                model_name=self.model_name,
                pretrained_model=self.pretrained_model,
                location=self.location
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4)