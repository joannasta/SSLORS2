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


# Example transformation (e.g., resize, normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Instantiate the DataModule
data_module = MaridaDataModule(batch_size=32, transform=transform)
# Setup the data (loading datasets into memory)
data_module.setup()

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')
# Test train dataloader loading
for batch in data_module.train_dataloader():
    img, target = batch
    print(f"Image batch shape: {img.shape}")
    print(f"Target batch shape: {target.shape}")
    
    # Convert the first image in the batch to a NumPy array
    img = img[0].cpu().numpy()

    img = (img - bands_mean[:, None, None]) / bands_std[:, None, None]
    
    # Select the relevant channels (if the image is in BGR format)
    img = img[1:4, :, :]  # Assuming channels are BGR (index 1:4 for Green, Blue, Red)

    # Convert BGR to RGB by swapping channels
    img = img[[2, 1, 0], :, :]  # Swap BGR to RGB

    # If the image has 3 channels (RGB), transpose it to (H, W, C) format for visualization
    if img.shape[0] == 3:  # Assuming it's a 3-channel image (RGB)
        img = np.transpose(img, (1, 2, 0))

    img = np.clip(img, 0, np.percentile(img, 99))
    img = (img / img.max() * 255).astype('uint8')
    
    # Save the image using matplotlib
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.savefig('image_sample.png', bbox_inches='tight')  # Save the image as a PNG file
    plt.close()  # Close the plot to free memory
    break  # Only load one batch for testing
