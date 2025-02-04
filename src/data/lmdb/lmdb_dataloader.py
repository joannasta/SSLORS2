from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import torch
from hydro import HydroDataset  # Your provided HydroDataset class

class HydroDataModule(LightningDataModule):
    def __init__(self, data_dir: str, bands: list = None, batch_size: int = 4, 
                 shuffle: bool = True, num_workers: int = 2):
        """
        Initialize the HydroDataModule.
        
        Args:
            data_dir: Path to the LMDB database.
            bands: List of band names to use (default is all bands).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Define transformations
        self.transforms = T.Compose([
            T.Resize((256, 256)),               # Resize tensor to 256x256
            T.Normalize(mean=[0.5]*len(bands),  # Normalize per band
                        std=[0.5]*len(bands)) if bands else None
        ])

    def setup(self, stage: str = None):
        """
        Setup datasets for training, validation, and testing.
        """
        if stage == 'fit' or stage is None:
            full_dataset = HydroDataset(self.data_dir, bands=self.bands, transforms=self.transforms)
            train_size = int(0.7 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = HydroDataset(self.data_dir, bands=self.bands, transforms=self.transforms)

    def train_dataloader(self):
        """
        Returns DataLoader for training data.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns DataLoader for validation data.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns DataLoader for test data.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)

    def teardown(self, stage: str = None):
        """
        Close LMDB environments when done.
        """
        if hasattr(self, "train_dataset"):
            self.train_dataset.dataset.close()
        if hasattr(self, "val_dataset"):
            self.val_dataset.dataset.close()
        if hasattr(self, "test_dataset"):
            self.test_dataset.close()


if __name__ == "__main__":
    data_dir = "/faststorage/joanna/lmdb/Encoded-Hydro"  # Path to LMDB database
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]  # Example band names
    batch_size = 16

    # Initialize DataModule
    data_module = HydroDataModule(data_dir=data_dir, bands=bands, batch_size=batch_size, num_workers=4)
    data_module.setup("fit")

    # Get training DataLoader
    train_dataloader = data_module.train_dataloader()
    
    # Iterate over batches
    for batch in train_dataloader:
        print(f"Batch shape: {batch.shape}")  # Check shape, e.g., [batch_size, num_bands, 256, 256]
        break
