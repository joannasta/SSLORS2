import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .mbn_dataset import MagicBathyNetDataset
import os
import random
import torch
import numpy as np
from config import train_images, test_images # Import these here

class MagicBathyNetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=1, transform=None, cache=True, pretrained_model=None, location="agia_napa",
                 full_finetune=True, random=False, ssl=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.cache = cache
        self.pretrained_model = pretrained_model
        self.location = location
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl

        # Removed: self.save_hyperparameters(...) - No hyperparameters saved by the DataModule

    def setup(self, stage=None):
        # Use the actual list of image IDs from config.py directly
        all_train_ids = train_images
        all_test_ids = test_images

        print(f"DataModule: Training images: {len(all_train_ids)}")
        print(f"DataModule: Test/Validation images: {len(all_test_ids)}")

        if stage == 'fit' or stage is None:
            # Initialize train dataset using all_train_ids
            self.train_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='train', 
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl,
                image_ids_for_this_split=all_train_ids # Pass all training IDs
            )
            
            # Initialize validation dataset using all_test_ids (as per your request)
            self.val_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='val', 
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl,
                image_ids_for_this_split=all_train_ids # Pass all testing IDs for validation
            )

        if stage == 'test' or stage is None:
            # Initialize test dataset using all_test_ids
            self.test_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='test', # Label this as 'test'
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl,
                image_ids_for_this_split=all_test_ids # Pass all testing IDs
            )


    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
