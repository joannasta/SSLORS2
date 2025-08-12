import numpy as np
import random
import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from skimage import io
from torch.utils.data import Dataset
from src.utils.finetuning_utils import get_random_pos
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG, train_images, test_images
from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing import DatasetProcessor
from pathlib import Path

random.seed(1)

class MagicBathyNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split_type='train', cache=True, augmentation=True, pretrained_model=None, location="agia_napa",
                 full_finetune=False, random_init=False, ssl=True, image_ids_for_this_split=None):
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform
        self.cache = cache
        self.augmentation = augmentation
        
        self.pretrained_model = pretrained_model
        self.train_images = train_images
        self.test_images = test_images
        self.location = location
             
        self.full_finetune = full_finetune
        self.random_init = random_init
        self.ssl = ssl
        
        self.processor = DatasetProcessor(
            img_dir=Path(self.root_dir) / location / 'img' / 's2',
            depth_dir=Path(self.root_dir) / location / 'depth' / 's2',
            output_dir=Path(self.root_dir) / 'processed_data',
            img_only_dir=Path(self.root_dir) / 'processed_img',
            depth_only_dir=Path(self.root_dir) / 'processed_depth',
            split_type=self.split_type
        )
        self.paired_files = self.processor.paired_files
        self.data_files = [pair[0] for pair in self.paired_files]
        self.label_files = [pair[1] for pair in self.paired_files]

        indices_to_use = self.train_images if split_type == 'train' else self.test_images

        filtered_data_files = []
        filtered_label_files = []
        for data_file, label_file in zip(self.data_files, self.label_files):
            file_number = self.processor.extract_index(data_file)
            if str(file_number) in indices_to_use:
                filtered_data_files.append(data_file)
                filtered_label_files.append(label_file)

        self.data_files = filtered_data_files
        self.label_files = filtered_label_files
        
        self.embeddings = None
        if self.pretrained_model is not None:
            self._create_embeddings() 

        self.norm_param_depth = NORM_PARAM_DEPTH[self.location]
        self.norm_param = np.load(NORM_PARAM_PATHS[self.location])

        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]

        self.data_cache_ = {}
        self.label_cache_ = {}
        
    def __len__(self):
        return 10000

    def _create_embeddings(self):
        hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=["B02", "B03", "B04"], location=self.location, ocean_flag=False)
        self.embeddings = []
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(model_device)
        self.pretrained_model.eval()
        
        if self.random_init:
            self.pretrained_model.cpu() 
            self.pretrained_model.apply(weights_init)

        for idx in range(len(hydro_dataset)):
            img = hydro_dataset[idx]
            
            if self.full_finetune:
                img = img.unsqueeze(0)
                img = F.interpolate(img, size=(256,256), mode='nearest')
                self.embeddings.append(img.cpu())
            elif self.random_init or self.ssl:
                with torch.no_grad():
                    img = img.cuda()
                    img = img.unsqueeze(0)
                    img = F.interpolate(img, size=(224,224), mode='nearest')
                    self.pretrained_model.to(img.device)
                    if self.pretrained_model.__class__.__name__ == "MAE" or self.pretrained_model.__class__.__name__  == 'MAE_Ocean':
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
        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            data = (data - self.norm_param[0][:, np.newaxis, np.newaxis]) / (self.norm_param[1][:, np.newaxis, np.newaxis] - self.norm_param[0][:, np.newaxis, np.newaxis]) 
            self.data_cache_[random_idx] = data

            
        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else: 
            label = 1/self.norm_param_depth * np.asarray(io.imread(self.label_files[random_idx]), dtype='float32')
            self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]

        data_p, label_p = self.data_augmentation(data_p, label_p)
        embedding = self.embeddings[random_idx]
        
        data_p_tensor = torch.from_numpy(data_p)
        label_p_tensor = torch.from_numpy(label_p)

        return data_p_tensor, label_p_tensor, embedding