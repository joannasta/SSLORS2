import numpy as np
import random
import os
import torch
import glob
import re
import torch.nn.functional as F  # Import for interpolation

from skimage import io # For image resizing
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from utils.finetuning_utils import get_random_pos
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG
from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing import DatasetProcessor
from src.models.mae import MAE
from pathlib import Path

random.seed(1)


# MagicBathyNetDataset class (modified)
class MagicBathyNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split_type='train', cache=False, augmentation=True, pretrained_model=None):
        print("Initializing MagicBathyNetDataset...")
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform
        self.cache = cache
        self.augmentation = augmentation
        self.pretrained_model = pretrained_model

        # Use DatasetProcessor to handle file organization
        self.processor = DatasetProcessor(
            img_dir=Path(self.root_dir) / 'agia_napa' / 'img' / 's2',
            depth_dir=Path(self.root_dir) / 'agia_napa' / 'depth' / 's2',
            output_dir=Path(self.root_dir) / 'processed_data' ,
            img_only_dir=Path(self.root_dir) / 'processed_img',
            depth_only_dir=Path(self.root_dir) / 'processed_depth' 
        )

        self.paired_files = self.processor.paired_files  # Use the paired files from the processor
        self.data_files = [pair[0] for pair in self.paired_files]
        self.label_files = [pair[1] for pair in self.paired_files]
        print(f"Found {len(self.data_files)} data files and {len(self.label_files)} label files.")

        self.hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=[ "B02", "B03", "B04"])
        self.embeddings = []
        self._create_embeddings()
        print(f"Created embeddings. Shape: {self.embeddings.shape}")
        # Normalisierungsparameter laden
        self.norm_param_depth = NORM_PARAM_DEPTH["agia_napa"]
        self.norm_param = np.load(NORM_PARAM_PATHS["agia_napa"])
        print("Loaded normalization parameters.")

        # Modellparameter laden
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]
        print("Loaded model parameters.")

        # Cache initialisieren
        self.data_cache_ = {}
        self.label_cache_ = {}
        print("Initialized cache.")
        
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    def _create_embeddings(self):
        print("Creating embeddings...")
        hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=["B02", "B03", "B04"])
        self.embeddings = []

        for idx in range(len(hydro_dataset)):  
            img = hydro_dataset[idx]
            print(f"Embedding creation: Processing image {idx+1}/{len(hydro_dataset)}. Image shape before interpolation: {img.shape}")

            img = img.unsqueeze(0).to(self.pretrained_model.device) 
            img = F.interpolate(img, size=(256,256), mode='nearest')
            print(f"Image shape after interpolation: {img.shape}")
            embedding = self.pretrained_model.forward_encoder(img) 
            self.embeddings.append(embedding.cpu())  

        self.embeddings = torch.stack(self.embeddings).cpu() 
        print("Finished creating embeddings.")
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        results = []
        for array in arrays:
            original_shape = array.shape
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            print(f"Data augmentation: Original shape: {original_shape}, Augmented shape: {results[-1].shape}") # Print shape after augmentation
            
        return tuple(results)
    
    def __getitem__(self, idx):
        print(f"Getting item at index {idx}...")
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        print(f"Random index chosen: {random_idx}")
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
            print("Data loaded from cache.")
        else:
            # Data is normalized in [0, 1]
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            print(f"Original data shape: {data.shape}")
            data = (data - self.norm_param[0][:, np.newaxis, np.newaxis]) / (self.norm_param[1][:, np.newaxis, np.newaxis] - self.norm_param[0][:, np.newaxis, np.newaxis]) 
            print(f"Normalized data shape: {data.shape}")
            
            if self.cache:
                self.data_cache_[random_idx] = data
            print("Data loaded and potentially cached.")
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
            print("Label loaded from cache.")
        else: 
            # Labels are converted from RGB to their numeric values
            label =np.asarray(io.imread(self.label_files[random_idx]), dtype='float32')
            print(f"Original label shape: {label.shape}")
            label = 1/self.norm_param_depth * label
            print(f"Processed label shape: {label.shape}")

            if self.cache:
                self.label_cache_[random_idx] = label
            print("Label loaded and potentially cached.")

        embedding = self.embeddings[random_idx].cpu()
        print(f"Embedding shape: {embedding.shape}")
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        print(f"Random crop coordinates: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        print(f"Cropped data shape: {data_p.shape}, Cropped label shape: {label_p.shape}")


        data_p, label_p = self.data_augmentation(data_p, label_p)

        print(f"Final data shape: {data_p.shape}, Final label shape: {label_p.shape}")
        return (torch.from_numpy(data_p).detach(),
                torch.from_numpy(label_p).detach(),embedding.detach())