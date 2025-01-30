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

        self.hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=[ "B02", "B03", "B04"])
        self.embeddings = []
        self._create_embeddings()
        # Normalisierungsparameter laden
        self.norm_param_depth = NORM_PARAM_DEPTH["agia_napa"]
        self.norm_param = np.load(NORM_PARAM_PATHS["agia_napa"])

        # Modellparameter laden
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]

        # Cache initialisieren
        self.data_cache_ = {}
        self.label_cache_ = {}
        
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    def _create_embeddings(self):  # This method is now called only once
        hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=["B02", "B03", "B04"])
        self.embeddings = []
        for idx in range(0,len(hydro_dataset)):
            img = hydro_dataset[idx]
            img = img.unsqueeze(0).to(self.pretrained_model.device)
            img = F.interpolate(img, size=(256,256), mode='nearest')
            self.embeddings.append(self.pretrained_model.forward_encoder(img))

        self.embeddings = torch.stack(self.embeddings)
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
                # Create Vision Transformer backbone
        results = []
        for array in arrays:
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
            
        return tuple(results)
    
    def __getitem__(self, idx):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            data = (data - self.norm_param[0][:, np.newaxis, np.newaxis]) / (self.norm_param[1][:, np.newaxis, np.newaxis] - self.norm_param[0][:, np.newaxis, np.newaxis]) 
            
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = 1/self.norm_param_depth * np.asarray(io.imread(self.label_files[random_idx]), dtype='float32')
            if self.cache:
                self.label_cache_[random_idx] = label
        
        embedding = np.asarray(self.embeddings[random_idx], dtype='float32')
        img = img.unsqueeze(0)
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]


        data_p, label_p = self.data_augmentation(data_p, label_p)

        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p),torch.from_numpy(embedding))
