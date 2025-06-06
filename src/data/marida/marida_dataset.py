import os
import json
import torch
import numpy as np
import random
import rasterio
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing_marida import DatasetProcessor
from config import get_marida_means_and_stds


dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)

class RandomRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

class MaridaDataset(Dataset):
    def __init__(self, root_dir, mode='train', pretrained_model=None, transform=None, standardization=None, path=dataset_path, img_only_dir=None, agg_to_water=True, save_data=False
                 , full_finetune=True, random=False, ssl=False, model_type='mae'):
        
        self.mode = mode
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl
        self.root_dir = Path(root_dir)
        self.X = []
        self.y = []
        self.means, self.stds, self.pos_weight = get_marida_means_and_stds()
        self.ROIs = np.genfromtxt(os.path.join(path, 'splits', f'{mode}_X.txt'), dtype='str')  # Dynamically generate split file path

        self.save_data = save_data
        
        with open(os.path.join(path, 'labels_mapping.txt'), 'r') as inputfile:
            self.labels = json.load(inputfile)

        self._load_data()

        self.embeddings = []
        self.pretrained_model = pretrained_model
        self.mode = mode
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            RandomRotationTransform([-90, 0, 90, 180]),
                                            transforms.RandomHorizontalFlip()
                                        ]) if transform else None
        self.standardization = transforms.Normalize(self.means, self.stds) if standardization else None
        self.length = len(self.y)
        self.path = Path(path)
        self.agg_to_water = agg_to_water

        if self.save_data:
            self._save_data_to_tiff()
        if pretrained_model:
            self._create_embeddings()
    
    def _load_data(self):
        for roi in tqdm(self.ROIs, desc=f'Load {self.mode} set to memory'):
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            roi_file = os.path.join(self.path, 'patches', roi_folder, roi_name + '.tif')
            roi_file_cl = os.path.join(self.path, 'patches', roi_folder, roi_name + '_cl.tif')

            try:
                with rasterio.open(roi_file_cl) as ds_y:
                    temp_y = np.copy(ds_y.read().astype(np.int64))
                    if self.agg_to_water:
                        temp_y[temp_y == 15] = 7
                        temp_y[temp_y == 14] = 7
                        temp_y[temp_y == 13] = 7
                        temp_y[temp_y == 12] = 7
                    temp_y = np.copy(temp_y - 1)
                    ds_y = None
                    self.y.append(temp_y)

                with rasterio.open(roi_file) as ds_x:
                    temp_x = np.copy(ds_x.read())
                    ds_x = None
                    self.X.append(temp_x)

            except rasterio.errors.RasterioIOError as e:
                print(f"Error opening file for ROI {roi}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for ROI {roi}: {e}")
        self.impute_nan = np.tile(self.means, (temp_x.shape[1],temp_x.shape[2],1))

        self.length = len(self.y)




    def _create_embeddings(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print("Initializing HydroDataset for embedding creation...")
        print("Using img_only_dir:", self.img_only_dir)
        hydro_dataset = HydroDataset(path_dataset=Path(self.img_only_dir), bands=["B02", "B03", "B04"])
        print(f"HydroDataset for embeddings has {len(hydro_dataset)} images.")
        self.embeddings_list = []
        
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(model_device)
        self.pretrained_model.eval() # Ensure model is in eval mode for embeddings

        if self.random:
            print("Applying random weights initialization to pretrained model.")
            self.pretrained_model.apply(weights_init)

        for idx in range(len(hydro_dataset)):
            if (idx + 1) % 100 == 0 or idx == len(hydro_dataset) - 1:
                print(f"  Processing image {idx + 1}/{len(hydro_dataset)} for embeddings...")
            
            img = hydro_dataset[idx]
            img = img.unsqueeze(0).to(model_device)
            img = F.interpolate(img, size=(256,256), mode='bilinear', align_corners=False)
            print("image shape", img.shape)
            
            if self.full_finetune:
                self.embeddings_list.append(img)
            elif self.random or self.ssl:
                with torch.no_grad():
                    if self.model_type == 'mae':
                        embedding = self.pretrained_model.forward_encoder(img)
                    elif self.model_type == "moco" or self.model_type == "mocogeo":
                        embedding = self.pretrained_model.backbone(img).flatten(start_dim=1)
                    else:
                        raise ValueError(f"Model {self.pretrained_model.__class__.__name__} not configured for embedding creation.")
                    self.embeddings_list.append(embedding)
            
        self.embeddings = torch.cat(self.embeddings_list, dim=0)
        print(f"Embeddings concatenated. Shape: {self.embeddings.shape}")  
        self.embeddings = self.embeddings.cpu()

    def __getitem__(self, index):
        img = self.X[index]
        target = self.y[index]
        embedding = self.embeddings[index].cpu()
        print("beginning")
        print(f"Image shape: {img.shape}, Target shape: {target.shape}")

        img = np.transpose(img, (1, 2, 0)).astype('float32')
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]
        
        if target.ndim == 3 and target.shape[0] == 1:
            target = np.squeeze(target, axis=0)

        if self.transform is not None:
            if target.ndim == 2:
                target_expanded = np.expand_dims(target, axis=-1)
            else:
                target_expanded = target

            print("During")
            print(f"Image shape: {img.shape}, Target shape: {target_expanded.shape}")
            stack = np.concatenate([img, target_expanded], axis=-1).astype('float32')
            stack = self.transform(stack)

            img = stack[:-1,:,:]
            target = stack[-1,:,:].long()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
            target = torch.from_numpy(target).long()

        if self.standardization is not None:
            img = self.standardization(img)
        print("end of get item")
        print(f"Image shape: {img.shape}, Target shape: {target.shape}")
        return img, target, embedding
