import os
import json
import torch
import numpy as np
import random
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import rasterio
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
# Handle potential import errors gracefully

from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing_marida import DatasetProcessor
from config import get_marida_means_and_stds

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

class MaridaDataset(Dataset):
    def __init__(self, root_dir, mode='train', pretrained_model=None, transform=None, standardization=None, path=dataset_path, img_only_dir=None, agg_to_water=True, save_data=False):
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = []
        self.y = []
        self.transform = transform
        self.random = False
        self.fully_finetuning = False
        self.means,self.stds, self.pos_weight = get_marida_means_and_stds()

        if mode == 'train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'), dtype='str')
        elif mode == 'test':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'), dtype='str')
            self.transform = transforms.Compose([transforms.ToTensor()])
        elif mode == 'val':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'), dtype='str')
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.path = Path(path)
        self.embeddings = []
        self.pretrained_model = pretrained_model
        self.mode = mode
        self.standardization = transforms.Normalize(self.means[:11], self.stds[:11]) if standardization else None
        self.length = len(self.y)
        self.agg_to_water = agg_to_water
        self.save_data = save_data
        
        with open(os.path.join(path, 'labels_mapping.txt'), 'r') as inputfile:
            self.labels = json.load(inputfile)

        self._load_data()
        if self.save_data:
            self._save_data_to_tiff()
        if not self.fully_finetuning:
            self._create_embeddings()


    def getnames(self):
        return self.ROIs

    def _load_data(self):
        for roi in tqdm(self.ROIs, desc=f'Load {self.mode} set to memory'):
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            cwd = os.getcwd()
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
                    self.y.append(temp_y)

                with rasterio.open(roi_file) as ds_x:
                    temp_x = np.copy(ds_x.read())
                    self.X.append(temp_x)
                    temp = temp_x

            except rasterio.errors.RasterioIOError as e:
                print(f"Error opening file for ROI {roi}: {e}")
        print("impute nan tempshape",  (temp.shape[1],temp.shape[2],1))
        self.impute_nan = np.tile(self.means, (temp.shape[1],temp.shape[2],1))

        self.length = len(self.y)

    def _save_data_to_tiff(self):
        print("Entering _save_data_to_tiff function...")  # Debug print

        output_folder = os.path.join(self.path, "roi_data", self.mode)
        os.makedirs(output_folder, exist_ok=True)
        output_img_folder = os.path.join(output_folder, "_images")
        os.makedirs(output_img_folder, exist_ok=True)
        output_target_folder = os.path.join(output_folder, "_target")
        os.makedirs(output_target_folder, exist_ok=True)
        output_paired_folder = os.path.join(output_folder, "_paired")
        os.makedirs(output_paired_folder, exist_ok=True)

        for i, roi in enumerate(tqdm(self.ROIs, desc=f'Saving {self.mode} data to TIFFs')):
            try:
                img = self.X[i]
                target = self.y[i]
                target = target.reshape(256, 256)  
                roi_name = '_'.join(['S2'] + roi.split('_'))

                img_filename = os.path.join(output_img_folder, f"X_{i:04d}_{roi_name}.tif")
                with rasterio.open(img_filename, 'w', driver='GTiff', width=img.shape[2], height=img.shape[1], count=img.shape[0], dtype=img.dtype, crs=None, transform=None) as dst:
                    dst.write(img)
                print(f"Image saved successfully: {os.path.exists(img_filename)}")  # Debug print

                target_filename = os.path.join(output_target_folder, f"y__{i:04d}_{roi_name}.tif") # Fixed filename
                with rasterio.open(target_filename, 'w', driver='GTiff', width=target.shape[1], height=target.shape[0], count=1, dtype=target.dtype, crs=None, transform=None) as dst:
                    dst.write(target, 1)
                print(f"Target saved successfully: {os.path.exists(target_filename)}")  # Debug print

                # Save paired data
                paired_filename = os.path.join(output_paired_folder, f"paired_{i:04d}_{roi_name}.tif")
                paired_data = np.concatenate([img, target[np.newaxis, :, :]], axis=0)
                with rasterio.open(paired_filename, 'w', driver='GTiff', width=img.shape[2], height=img.shape[1], count=paired_data.shape[0], dtype=paired_data.dtype, crs=None, transform=None) as dst:
                    dst.write(paired_data)
                print(f"Paired data saved successfully: {os.path.exists(paired_filename)}")  # Debug print

            except Exception as e:
                print(f"Error saving ROI {i}: {e}")  # Catch and print any exceptions

        print("Finished _save_data_to_tiff function.")  # Debug print


    def __len__(self):
        return self.length

    def getnames(self):
        return self.ROIs
    
    def _create_embeddings(self):
        hydro_dataset = HydroDataset(path_dataset=self.path / "roi_data" / self.mode / "_images", bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11"])
        self.embeddings = []

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.random:
            print("Random weights initialization")
            self.pretrained_model.cpu() # ensure the model is on the cpu.
            self.pretrained_model.apply(weights_init)

        for idx in range(len(hydro_dataset)):
            img = hydro_dataset[idx]
            img = img.unsqueeze(0)

            # Get the device of the model
            device = next(self.pretrained_model.parameters()).device

            # Move the input image to the same device as the model
            img = img.to(device)

            if not self.fully_finetuning:
                with torch.no_grad():
                    embedding = self.pretrained_model.forward_encoder(img)
                self.embeddings.append(embedding.cpu())

        self.embeddings = torch.stack(self.embeddings).cpu()

    def _finetune_embeddings(self,idx):
        hydro_dataset = HydroDataset(path_dataset=self.path / "roi_data" / self.mode / "_images", bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11"])

        #print(f"Processing image index: {idx}") # print the index
        img = hydro_dataset[idx]
        img = img.unsqueeze(0)
        return img


    def __getitem__(self, index):
        img = self.X[index]
        target = self.y[index]

        if self.pretrained_model and not self.fully_finetuning:
            embedding = self.embeddings[index]
        else:
            embedding = self._finetune_embeddings(index)

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')  # CxWxH to WxHxC
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]

        #print(f"Image shape after imputation: {img.shape}, dtype: {img.dtype}")  # Debug print
        if self.transform is not None:
            target = target.transpose(1, 2, 0)
            stack = np.concatenate([img, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image

            stack = self.transform(stack)
            img = stack[:-1,:,:]
            target = stack[-1,:,:]                                    # Recast target values back to int64 or torch long dtype

        if self.standardization is not None:
            img = self.standardization(img)
        return img, target, embedding



###############################################################
# Transformations                                             #
###############################################################
class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)
    
###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)