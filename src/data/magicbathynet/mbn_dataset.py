# src/data/magicbathynet/mbn_dataset.py

import numpy as np
import random
import os
import torch
import glob
import re
import torch.nn.functional as F
import torch.nn as nn

from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from src.utils.finetuning_utils import get_random_pos
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images
from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing import DatasetProcessor
from src.models.mae import MAE
from pathlib import Path

random.seed(1)


class MagicBathyNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split_type='train', cache=True, augmentation=True, pretrained_model=None,location="agia_napa",
                 full_finetune=True, random=False, ssl=False,
                 image_ids_for_this_split=None):
        print(f"Initializing MagicBathyNetDataset for split: {split_type}")
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform
        self.cache = cache
        self.augmentation = augmentation
        
        self.pretrained_model = pretrained_model
        self.train_images = train_images
        self.test_images = test_images
        self.location = location

        print("split_type:", split_type)
        print("Initializing DatasetProcessor...")
             
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl
        
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
        
        self.embeddings = None
        print("Starting embedding creation...")
        if self.pretrained_model is not None:
            self._create_embeddings() 
            print("Embedding creation complete.")

        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])
        print("Normalization parameters loaded.")

        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]
        print(f"Model config: crop_size={self.crop_size}, window_size={self.window_size}, stride={self.stride}")


        self.data_cache_ = {}
        self.label_cache_ = {}
        print("Data and label caches initialized.")
        
    def __len__(self):
        # Keep 10000 for your "epoch length"
        return 10000

    def _create_embeddings(self):
        hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=["B02", "B03", "B04"],location=self.location)
        print(f"Length of hydro_dataset: {len(hydro_dataset)}")
        self.embeddings = []
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(model_device)
        self.pretrained_model.eval()
        
        if self.random:
            self.pretrained_model.cpu() 
            self.pretrained_model.apply(weights_init)

        for idx in range(len(hydro_dataset)):
            img = hydro_dataset[idx]
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(256,256), mode='nearest')
            
            if self.full_finetune:
                self.embeddings.append(img.cpu())
            elif self.random or self.ssl:
                with torch.no_grad():
                    img = img.cuda()
                    self.pretrained_model.to(img.device)
                    if self.pretrained_model.__class__.__name__ == "MAE":
                        embedding = self.pretrained_model.forward_encoder(img)
                    elif self.pretrained_model.__class__.__name__ in ["MoCo", "MoCoGeo"]:
                        embedding = self.pretrained_model.backbone(img).flatten(start_dim=1)
  
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
        embedding = self.embeddings[random_idx]
        
        data_p_tensor = torch.from_numpy(data_p)
        label_p_tensor = torch.from_numpy(label_p)

        return (data_p_tensor,
                label_p_tensor,embedding)
