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
        transform = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bands = bands
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform


    def setup(self, stage=None):
        # Use stage to load data depending on the task
        if stage == 'fit' or stage is None:
            self.train_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform
            )
            
            self.val_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform
            )

        if stage == 'test' or stage is None:
            # Initialize test dataset for evaluation
            self.test_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform
            )

        if stage == 'predict':
            # You can set up a different dataset for prediction if needed
            self.predict_dataset = HydroDataset(
                path_dataset=self.data_dir,
                bands=self.bands,
                compute_stats=False,
                transforms=self.transform
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
            
    