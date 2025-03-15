import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

import matplotlib.pyplot as plt
# Import your dataset class (ensure this is correctly defined in your project)
from .marida_dataset import MaridaDataset # Update the path as per your project structure
import numpy as np
from config import get_marida_means_and_stds

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
"""
# Initialize the DataModule
marida_data = MaridaDataModule(
    root_dir="/faststorage/joanna/marida/MARIDA",  # Replace with your path
    batch_size=32,
    # If you have transforms or standardization, include them here.
)
marida_data.setup()  # Set up the datasets

# Visualize a few samples
num_samples = 10  # Number of samples to visualize
modes = ['train', 'val', 'test']
means, stds,_ = get_marida_means_and_stds()
for mode in modes:
    print(f"Visualizing samples from {mode} set:")
    for i in range(num_samples):
        sample = marida_data.train_dataset[i]

        image, target, embedding = sample
        # Visualize the image and depth
        print("image shape", image.shape)
        print("target shape", target.shape)
        img = (image *stds[None,None,:]) + ( means[None,None,:]) 
        img = img[:,:,1:4]  
        img = img[:,:,[2, 1, 0]]  # Swap BGR to RGB

        img = np.clip(img, 0, np.percentile(img, 99))
        image = (img / img.max() * 255).astype('uint8')
        
        target = target.squeeze(0)
        print("image shape", image.shape)
        print("target shape", target.shape)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"{mode.capitalize()} Image {i}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(target, cmap='viridis')  # Use a colormap for depth
        plt.title(f"{mode.capitalize()} Depth {i}")
        plt.axis('off')
        plt.savefig(f"marida_sample__{i}.png")
        plt.show()"""