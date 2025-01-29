import torch
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from hydro_dataset import HydroDataset 

def collate_fn(batch):
    """
    Collate function to filter out None samples from the batch.
    """
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        raise ValueError("All samples in the batch failed to load.")
    return torch.utils.data.default_collate(batch)

class HydroDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        bands: List[str] = None,
        batch_size: int = 4,
        shuffle: bool = False,
        num_workers: int = 2,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bands = bands
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: str = None):
        """
        Prepare dataset for pretraining (no need for validation/test splits).
        """
        # Load the entire dataset
        full_dataset = HydroDataset(
            path_dataset=self.data_dir,
            bands=self.bands,
            compute_stats=False
        )

        # Create transforms for pretraining
        train_transform = T.Compose([
            T.RandomResizedCrop(
                    256,
                    scale=(0.67, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                #T.Lambda(lambda x: x / 10000.0),
                #T.Normalize(mean=self.band_means, std=self.band_stds)
            ]
        ) 

        # Apply the transformation to the entire dataset for pretraining
        full_dataset.transforms = train_transform

        # Split dataset if needed
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        test_size = int(self.test_split * total_size)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn
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
                collate_fn=collate_fn
            )
        else:
            raise ValueError("Test dataset is None.")
        
 

# Initialize the HydroDataModule with appropriate parameters
data_dir = "/faststorage/joanna/Hydro/raw_data"  # Change this to your actual dataset path
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]  # Modify according to your dataset
batch_size = 4
data_module = HydroDataModule(data_dir=data_dir, bands=bands, batch_size=batch_size, shuffle=True)
data_module.setup()  # Set up the datasets

data = []

means=  np.array([340.76769064, 429.9430203, 614.21682446, 590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375, 2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131])
stds= np.array([554.81258967, 572.41639287, 582.87945694, 675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904, 1356.13789355, 1302.3292881, 1079.19066363, 818.86747235])
  
# Loop through a few images for debugging
for i in range(50):  # Inspect first 5 images
    # Step 1: Get an image from the dataset
    image = data_module.train_dataset[i]  # shape: (C, H, W)
    
    image = (image - means[:,None,None]) /stds[:,None,None]

    print("After Normalization:")
    print(f"Min: {image.min().item()}, Max: {image.max().item()}, Mean: {image.mean().item()}")

    # Step 3: Denormalize the image
    image_denormalized =  (image * stds[:,None,None]) + means[:,None,None]#(image * (max_value - min_value+ 1e-7)) + min_value
    print("After Denormalization:")
    print(f"Min: {image_denormalized.min().item()}, Max: {image_denormalized.max().item()}, Mean: {image_denormalized.mean().item()}")

    # Step 4: Append normalized data for histogram
    data.append(image.flatten())

    # Step 5 (optional): Visualize RGB image (Bands 2, 3, 4 as RGB)
    # Uncomment if needed
    image_denormalized  = image_denormalized.squeeze(0)
    bgr_image = image_denormalized [1:4, :, :]
    rgb_image = bgr_image[[2, 1, 0], :, :].permute(1, 2, 0).numpy()
    rgb_image = np.clip(rgb_image, 0, np.percentile(rgb_image, 99))
    rgb_image = rgb_image / rgb_image.max()

    print("After Clipping:")
    print(f"Min: {rgb_image.min().item()}, Max: {rgb_image.max().item()}, Mean: {rgb_image.mean().item()}")

    plt.imshow(rgb_image)
    plt.title(f"RGB Visualization - Image {i}")
    plt.axis("off")
    plt.savefig(f"rgb_visualization_image_{i}.png")
    plt.show()

# Step 6: Plot histogram of normalized pixel values
#plt.figure(figsize=(10, 6))
#data = np.concatenate(data)
#data = data[np.isfinite(data)]
#plt.hist(data, bins=10, color='blue', alpha=0.7)
#plt.title("Pixel Value Distribution (Normalized Images)")
#plt.xlabel("Pixel Value")
#plt.ylabel("Frequency")
#plt.grid()
#plt.savefig("pixel_distribution_normalized.png")
#plt.show()
