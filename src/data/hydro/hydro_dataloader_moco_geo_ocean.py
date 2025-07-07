import torch
from torch.utils.data import Dataset, DataLoader, default_collate # Import default_collate
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
# IMPORTANT: Update the import path to your new dataset class
from src.data.hydro.hydro_moco_geo_ocean_dataset import HydroMocoGeoOceanFeaturesDataset 
from pytorch_lightning import LightningDataModule

class HydroOceanFeaturesDataModule(LightningDataModule): # Renamed
    def __init__(
            self, 
            data_dir: str, 
            batch_size: int = 32, 
            num_workers: int = 4, 
            transform: Optional[Callable] = None, 
            model_name: str = "moco-ocean-features", 
            csv_features_path: str = "/mnt/storagecube/joanna/ocean_features_filtered.csv", 
            max_match_distance: float = 0.001 
        ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.model_name = model_name
        self.csv_features_path = csv_features_path
        self.max_match_distance = max_match_distance
        
    # --- Custom collate_fn to filter out None values ---
    def custom_collate_fn(self, batch: List[Any]): # Added type hint for batch
        # Filter out None values from the batch
        batch = [item for item in batch if item is not None]
        if not batch: # If the batch is empty after filtering, return None or an empty list
            # Return an empty list or raise an error if an empty batch is not desired.
            # Returning None can cause issues with DataLoader. Best to return empty tensors or skip.
            # For PyTorch Lightning, returning an empty list might cause issues in training_step/validation_step.
            # Consider filtering files more strictly in the dataset's __init__ instead of returning None from __getitem__.
            print("Warning: Batch is empty after filtering None values. This batch will be skipped.")
            return None # Or raise ValueError("Empty batch after filtering None values.")
        return default_collate(batch)
    # --- End custom collate_fn ---

    def setup(self, stage: Optional[str] = None):
        # Instantiate HydroOceanFeaturesDataset instead of HydroMoCoGeoDataset
        # Pass the new parameters: csv_features_path and max_match_distance
        if stage == 'fit' or stage is None:
            self.train_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                csv_features_path=self.csv_features_path,
                max_match_distance=self.max_match_distance
            )
            # You might want separate validation data. For simplicity, reusing data_dir here.
            # In a real scenario, you'd have train_data_dir and val_data_dir.
            self.val_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir, 
                transforms=self.transform,
                model_name=self.model_name,
                csv_features_path=self.csv_features_path,
                max_match_distance=self.max_match_distance
            )

        if stage == 'test' or stage is None:
            self.test_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                csv_features_path=self.csv_features_path,
                max_match_distance=self.max_match_distance
            )

        if stage == 'predict':
            self.predict_dataset = HydroMocoGeoOceanFeaturesDataset(
                path_dataset=self.data_dir,
                transforms=self.transform,
                model_name=self.model_name,
                csv_features_path=self.csv_features_path,
                max_match_distance=self.max_match_distance
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn 
        )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn 
            )
        else:
            print("WARNING: Validation dataloader requested but val_dataset not initialized or is None. Returning None.")
            return None 

    def test_dataloader(self):
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.custom_collate_fn 
            )
        else:
            print("WARNING: Test dataloader requested but test_dataset not initialized or is None. Returning None.")
            return None