import pytorch_lightning as pl
import numpy as np
import scipy
from torch.utils.data import DataLoader
from .mbn_dataset import MagicBathyNetDataset
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG

class MagicBathyNetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, modality, batch_size=32, transform=None, cache=False):
        super().__init__()
        self.root_dir = root_dir
        self.modality = modality
        self.batch_size = batch_size
        self.transform = transform
        self.cache = cache
        self.norm_param_depth = NORM_PARAM_DEPTH["agia_napa"]
        self.norm_param = np.load(NORM_PARAM_PATHS["agia_napa"])

        # Load common model parameters
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]

    def setup(self, stage=None):
        # Initialize train and test datasets
        self.train_dataset = MagicBathyNetDataset(
            root_dir=self.root_dir,
            modality=self.modality,
            transform=self.transform,
            split_type='train',
            cache=self.cache
        )
        
        self.test_dataset = MagicBathyNetDataset(
            root_dir=self.root_dir,
            modality=self.modality,
            transform=self.transform,
            split_type='test',
            cache=self.cache
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # Ensure scipy is imported for zoom operations

data_module = MagicBathyNetDataModule(
    root_dir='/faststorage/joanna/magicbathynet/MagicBathyNet',
    modality="s2",
    batch_size=4,
)
def test_dataset(data_module):
    # Set up the data
    data_module.setup()
    crop_size = 256
    WINDOW_SIZE = (256, 256)
    norm_param_depth = -30.443
    ratio = crop_size / WINDOW_SIZE[0]

    # Load normalization parameters
    norm_param = np.load('/faststorage/joanna/magicbathynet/MagicBathyNet/agia_napa/norm_param_s2_an.npy')

    # Load the train dataloader
    train_loader = data_module.train_dataloader()

    # Limit to 5 images
    image_count = 0
    max_images = 10

    for batch_idx, (images, depths) in enumerate(train_loader):
        for img, depth in zip(images, depths):
            if image_count >= max_images:
                break

            # Normalize and process the image
            img = torch.clamp(img, min=0, max=1)  # Clip to [0, 1]
            processed_img = np.transpose(img.numpy(), (1, 2, 0))  # Convert to (H, W, C)

            # Depth denormalization and resizing
            depth = depth.numpy() * norm_param_depth
            depth_resized = zoom(depth, (1 / ratio, 1 / ratio), order=1)

            # Plot and save depth histogram
            plt.figure(figsize=(10, 6))
            plt.hist(depth_resized.flatten(), bins=50, color='blue', alpha=0.7)
            plt.title("Depth Pixel Denormalized Value Distribution")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            histogram_path = f"depth_histogram_image_{image_count}.png"
            #plt.savefig(histogram_path)
            plt.close()
            print(f"Saved histogram to {histogram_path}")

            # Plot and save image and depth map
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Display the RGB image (assumes channels are in RGB order)
            axes[0].imshow(processed_img[:, :, [2, 1, 0]])  # Switch to BGR for visualization
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Display the depth map
            axes[1].imshow(depth_resized, cmap='viridis')  # Add colormap for better visualization
            axes[1].set_title('Depth Map')
            axes[1].axis('off')

            # Save the figure
            save_path = f"image_and_depth_{image_count}.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved image and depth map to {save_path}")

            image_count += 1

        if image_count >= max_images:
            break

if __name__ == '__main__':
    # Assume data_module is already defined and initialized
    test_dataset(data_module)