import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.transforms as T

from torchvision.utils import make_grid # Import make_grid
import os
from config import get_marida_means_and_stds

import matplotlib.pyplot as plt
# Import your dataset class (ensure this is correctly defined in your project)
from .marida_dataset import MaridaDataset # Update the path as per your project structure
import numpy as np

class MaridaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, root_dir="/home/jovyan/SSLORS/marida", transform=None, standardization=None,
                 full_finetune=True, random=False, ssl=False, pretrained_model=None,model_type='mae'):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.path ="/mnt/storagecube/joanna/MARIDA"
        self.transform = transform
        self.standardization = standardization
        self.img_only_dir = "/mnt/storagecube/joanna/MARIDA/roi_data"
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        
    def setup(self, stage=None):
        """Setup dataset for training, validation, or test."""
        print("self.path", self.path)
        self.train_dataset = MaridaDataset(root_dir=self.root_dir,mode='train', transform=self.transform, standardization=self.standardization, path=self.path,full_finetune=self.full_finetune, random=self.random, ssl=self.ssl, pretrained_model=self.pretrained_model,model_type=self.model_type)
        self.val_dataset = MaridaDataset(root_dir=self.root_dir, mode='val', transform=self.transform, standardization=self.standardization, path=self.path,full_finetune=self.full_finetune, random=self.random, ssl=self.ssl,pretrained_model=self.pretrained_model,model_type=self.model_type)
        self.test_dataset = MaridaDataset(root_dir=self.root_dir, mode='test', transform=self.transform, standardization=self.standardization, path=self.path,full_finetune=self.full_finetune, random=self.random, ssl=self.ssl, pretrained_model=self.pretrained_model,model_type=self.model_type)

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
