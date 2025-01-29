import numpy as np
import random
import os
import torch
import glob
import re
from skimage import io # For image resizing
import torch.nn.functional as F  # Import for interpolation
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from utils.finetuning_utils import get_random_pos
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG


random.seed(1)

class MagicBathyNetDataset(Dataset):
    def __init__(self, root_dir, modality, transform=None, split_type='train', cache=False, augmentation=True, target_size=(3, 256, 256), split_ratio=0.8):
        self.root_dir = root_dir
        self.modality = modality
        self.split_type = split_type
        self.transform = transform
        self.cache = cache
        self.augmentation = augmentation
        self.target_size = target_size  # Zielgröße für die Größenanpassung
        self.split_ratio = split_ratio  # Train/Test-Split-Verhältnis
        self.train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
        self.test_images = ['411', '387', '410', '398', '370', '369', '397']

        # Verzeichnisse definieren
        self.img_dir = os.path.join(self.root_dir, 'agia_napa', 'img', self.modality)
        self.depth_dir = os.path.join(self.root_dir, 'agia_napa', 'depth', self.modality)

        # Alle Dateien sammeln
        self.all_img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.tif')))
        self.all_depth_files = sorted(glob.glob(os.path.join(self.depth_dir, '*.tif')))

        # Paare sicherstellen (Bild- und Tiefendateien)
        #self.paired_files = [(img, depth) for img, depth in zip(self.all_img_files, self.all_depth_files) if os.path.isfile(img) and os.path.isfile(depth)]

        # Regex pattern to extract the numeric index from the filenames (e.g., "img_1062.tif" -> "1062")

        # Pairing logic with index matching
        self.paired_files = []
        for img_path in self.all_img_files:
            img_idx = self.extract_index(os.path.basename(img_path))
            
            for depth_path in self.all_depth_files:
                depth_idx = self.extract_index(os.path.basename(depth_path))
                
                # Ensure indices match and files exist
                if img_idx == depth_idx and os.path.isfile(img_path) and os.path.isfile(depth_path):
                    self.paired_files.append((img_path, depth_path))
                    break  # Stop searching once a match is found

                
        # Dateien mischen und aufteilen
        random.shuffle(self.paired_files)
        split_index = int(len(self.paired_files) * self.split_ratio)
        train_files = self.paired_files[:split_index]
        test_files = self.paired_files[split_index:]

        # Train/Test-Daten basierend auf split_type zuweisen
        if self.split_type == "train":
            self.data_files, self.label_files = zip(*train_files) if train_files else ([], [])
        elif self.split_type == "test":
            self.data_files, self.label_files = zip(*test_files) if test_files else ([], [])

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
        self.dataset_len_true = len(self.all_depth_files)

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    def extract_index(self,filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else None
    
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
        
  
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]


        data_p, label_p = self.data_augmentation(data_p, label_p)

        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))
