from pathlib import Path
from typing import List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import torch
from .hydro_dataset import HydroDataset 

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
        
 