import numpy as np
import random
import os
import torch
import glob
import re
import torch.nn.functional as F  # Import for interpolation
import torch.nn as nn

from pathlib import Path
from skimage import io # For image resizing
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

from utils.finetuning_utils import get_random_pos
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images
from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing import DatasetProcessor
from src.models.mae import MAE

random.seed(1)


# MagicBathyNetDataset class (modified)
class MagicBathyNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split_type='train', cache=True, augmentation=True, pretrained_model=None,location="agia_napa"):
        print("Initializing MagicBathyNetDataset...")
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform
        self.cache = cache
        self.augmentation = augmentation
        self.pretrained_model = pretrained_model
        self.train_images = train_images
        self.test_images = test_images
        self.random = False
        
        print("split_type:", split_type)
        print("Initializing DatasetProcessor...")
        self.processor = DatasetProcessor(
            img_dir=Path(self.root_dir) / location/ 'img' / 's2',
            depth_dir=Path(self.root_dir) /location / 'depth' / 's2',
            output_dir=Path(self.root_dir) / 'processed_data' ,
            img_only_dir=Path(self.root_dir) / 'processed_img',
            depth_only_dir=Path(self.root_dir) / 'processed_depth',
            split_type=self.split_type
        )
        print("DatasetProcessor initialized.")
        self.paired_files = self.processor.paired_files  # Use the paired files from the processor
        self.data_files = [pair[0] for pair in self.paired_files]
        self.label_files = [pair[1] for pair in self.paired_files]

        indices_to_use = self.train_images if split_type == 'train' else self.test_images

        filtered_data_files = []
        filtered_label_files = []
        for data_file, label_file in zip(self.data_files, self.label_files):
            file_number = self.processor.extract_index((data_file))  # Use the extract_last_number function
            if file_number is not None and str(file_number) in indices_to_use:
                filtered_data_files.append(data_file)
                filtered_label_files.append(label_file)
        print("filtered_data_files:", len(filtered_data_files))

        self.data_files = filtered_data_files
        self.label_files = filtered_label_files

        self.hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=[ "B02", "B03", "B04"])
        self.embeddings = []
        print("creating embeddings...")
        self._create_embeddings()
        print("embeddings created.")
        self.norm_param_depth = NORM_PARAM_DEPTH[location]
        self.norm_param = np.load(NORM_PARAM_PATHS[location])
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]

        self.data_cache_ = {}
        self.label_cache_ = {}
        
    def __len__(self):
        return 10000
    

    def _create_embeddings(self):
        hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=["B02", "B03", "B04"])
        self.embeddings = []

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.random:
            self.pretrained_model.cpu() # ensure the model is on the cpu.
            self.pretrained_model.apply(weights_init)


        for idx in range(len(hydro_dataset)):
            img = hydro_dataset[idx]
            img = img.unsqueeze(0).to(self.pretrained_model.device)
            img = F.interpolate(img, size=(256,256), mode='nearest')
            #self.embeddings.append(img)

            with torch.no_grad():
                embedding = self.pretrained_model.forward_encoder(img)
                print("embedding", embedding) # Uncomment this line.
            self.embeddings.append(embedding.cpu())

        self.embeddings = torch.stack(self.embeddings).cpu()

    

    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
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
        random_idx = random.randint(0, len(self.data_files) - 1)
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            data = (data - self.norm_param[0][:, np.newaxis, np.newaxis]) / (self.norm_param[1][:, np.newaxis, np.newaxis] - self.norm_param[0][:, np.newaxis, np.newaxis]) 
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            label = 1/self.norm_param_depth * np.asarray(io.imread(self.label_files[random_idx]), dtype='float32')
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]

        data_p, label_p = self.data_augmentation(data_p, label_p)
        
        data_p_tensor = torch.from_numpy(data_p)
        label_p_tensor = torch.from_numpy(label_p)
        embedding = self.embeddings[random_idx].cpu()

        return (data_p_tensor,
                label_p_tensor,embedding)