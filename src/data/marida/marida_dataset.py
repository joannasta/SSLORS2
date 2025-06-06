import os
import json
import torch
import numpy as np
import random

from tqdm import tqdm
import rasterio
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import torch.nn.functional as F

from src.data.hydro.hydro_dataset import HydroDataset
from src.utils.data_processing_marida import DatasetProcessor
from config import get_marida_means_and_stds

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

class MaridaDataset(Dataset):
    def __init__(self, root_dir, mode='train', pretrained_model=None, transform=None, standardization=None, path=dataset_path, img_only_dir=None, agg_to_water=True, save_data=False
                 , full_finetune=True, random=False, ssl=False,model_type="mae"):
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.X = []
        self.y = []
        self.transform = transform
        self.means, self.stds, self.pos_weight = get_marida_means_and_stds()
        self.full_finetune = True
        self.random = False
        self.ssl = False
        self.model_type = model_type
        
        if mode == 'train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'), dtype='str')
            print("train ROIS are used")
        elif mode == 'test': 
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'), dtype='str')
            self.transform = transforms.Compose([transforms.ToTensor()]) # This transform will apply only to the test set if no other transform is passed
        elif mode == 'val': # Changed to elif for correct logic flow
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'), dtype='str')
            print("val ROIS are used")
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.path = Path(path)
        self.embeddings = []
        self.pretrained_model = pretrained_model
        # self.mode = mode # This line is redundant as 'mode' is set at the very beginning of __init__
        self.standardization = transforms.Normalize(self.means, self.stds) if standardization else None
        
        # self.length is initialized here but will be updated after _load_data()
        self.length = 0 
        
        self.agg_to_water = agg_to_water
        self.save_data = save_data
        
        with open(os.path.join(path, 'labels_mapping.txt'), 'r') as inputfile:
            self.labels = json.load(inputfile)

        self._load_data()
        # Update length AFTER data is loaded
        self.length = len(self.y)

        if self.save_data:
            self._save_data_to_tiff()
        if pretrained_model:
            self._create_embeddings()


    def getnames(self):
        return self.ROIs

    def _load_data(self):
        for roi in tqdm(self.ROIs, desc=f'Load {self.mode} set to memory'):
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            # cwd = os.getcwd() # This line is not used and can be removed
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
                    temp_y = np.copy(temp_y - 1) # This operation will apply to all modes
                    self.y.append(temp_y)

                with rasterio.open(roi_file) as ds_x:
                    temp_x = np.copy(ds_x.read())
                    self.X.append(temp_x)
                temp = temp_x # Keep track of the last successfully loaded image for impute_nan
            except rasterio.errors.RasterioIOError as e:
                print(f"Error opening file for ROI {roi}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for ROI {roi}: {e}")
        
        # Initialize self.impute_nan after loading data, or with a robust default
        if self.X: # Only set if X has data
            # Assumes all images have the same (H, W) dimensions
            self.impute_nan = np.tile(self.means, (self.X[0].shape[1], self.X[0].shape[2], 1))
        else:
            # Fallback if no images were loaded (e.g., ROIs list was empty or all failed)
            print("Warning: No ROIs loaded, `impute_nan` might not be correctly initialized. Assuming default (256, 256, 3) shape for safety.")
            # Provide a reasonable default shape based on expected image dimensions and number of channels in means
            self.impute_nan = np.tile(self.means, (256, 256, 1)) 


    def _save_data_to_tiff(self):
        print("Entering _save_data_to_tiff function...") # Debug print

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
                if target.ndim == 3 and target.shape[0] == 1: # If (1, H, W)
                    target = target.squeeze(0)
                elif target.ndim != 2:
                    raise ValueError(f"Target has unexpected dimensions: {target.shape}. Expected (H, W) or (1, H, W).")
                
                roi_name = '_'.join(['S2'] + roi.split('_'))

                img_filename = os.path.join(output_img_folder, f"X_{i:04d}_{roi_name}.tif")

                if img.ndim == 3 and img.shape[0] > 4: 
                    pass #
                elif img.ndim == 3 and img.shape[2] > 4: 
                    img = np.moveaxis(img, -1, 0)
                elif img.ndim == 2: 
                    img = np.expand_dims(img, axis=0)
                else:
                    raise ValueError(f"Image has unexpected dimensions: {img.shape}. Expected (C, H, W) or (H, W, C).")

                with rasterio.open(img_filename, 'w', driver='GTiff', width=img.shape[2], height=img.shape[1], count=img.shape[0], dtype=img.dtype, crs=None, transform=None) as dst:
                    dst.write(img)
  
                target_filename = os.path.join(output_target_folder, f"y_{i:04d}_{roi_name}.tif") # Fixed the double underscore
                with rasterio.open(target_filename, 'w', driver='GTiff', width=target.shape[1], height=target.shape[0], count=1, dtype=target.dtype, crs=None, transform=None) as dst:
                    dst.write(target, 1) # Write as band 1

                paired_filename = os.path.join(output_paired_folder, f"paired_{i:04d}_{roi_name}.tif")
                if target.ndim == 2:
                    target_for_concat = target[np.newaxis, :, :] # Make it (1, H, W)
                else:
                    target_for_concat = target # If it's already (C', H, W)

                paired_data = np.concatenate([img, target_for_concat], axis=0) # Concatenate along channels
                with rasterio.open(paired_filename, 'w', driver='GTiff', width=img.shape[2], height=img.shape[1], count=paired_data.shape[0], dtype=paired_data.dtype, crs=None, transform=None) as dst:
                    dst.write(paired_data)

            except Exception as e:
                print(f"Error saving ROI {i}: {e}") # Catch and print any exceptions

        print("Finished _save_data_to_tiff function.") # Debug print


    def __len__(self):
        return self.length

    def _create_embeddings(self):
        # Using self.path for HydroDataset to point to the saved image data
        hydro_dataset = HydroDataset(path_dataset=self.path / "roi_data" / self.mode / "_images", bands=["B01","B02", "B03", "B04","B05","B06","B07","B08","B8A","B09","B11"])
        self.embeddings = []
        
        # Ensure pretrained_model is on the correct device for inference
        if self.pretrained_model is not None:
            self.pretrained_model.eval() # Set to evaluation mode for inference
            device = next(self.pretrained_model.parameters()).device # Get model's device
        else:
            # Handle case where pretrained_model is None, or raise an error
            raise ValueError("Pretrained model is not provided for embedding creation.")

        for idx in tqdm(range(len(hydro_dataset)), desc=f'Creating embeddings for {self.mode} set'):
            img = hydro_dataset[idx]
            
            # Ensure img has the correct number of dimensions (e.g., [C, H, W]) before unsqueeze
            if img.dim() == 3: # Assuming C, H, W
                img = img.unsqueeze(0).to(device) # Add batch dimension and move to device
            elif img.dim() == 2: # Assuming H, W, need to add channel and batch (e.g., for grayscale)
                img = img.unsqueeze(0).unsqueeze(0).to(device) # Add channel and batch dimension
            else:
                raise ValueError(f"Unexpected image dimensions from HydroDataset: {img.dim()}. Expected 2 or 3.")

            # Interpolate to 256x256 (as mentioned in previous discussions)
            img = F.interpolate(img, size=(256,256), mode='nearest')
            
            with torch.no_grad():
                img = img[:,1:4, :, :]  
                embedding = self.pretrained_model.forward_encoder(img)
            self.embeddings.append(embedding.cpu())
        self.embeddings = torch.stack(self.embeddings).cpu()

    def __getitem__(self, index):
        img = self.X[index]
        target = self.y[index]

        if self.pretrained_model:
            embedding = self.embeddings[index]
        else:
            embedding = torch.zeros(32, 1, 256, 256)

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')

        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]

        if self.transform is not None:
            target = target.transpose(1, 2, 0)
            stack = np.concatenate([img, target], axis=-1).astype('float32')
            stack = self.transform(stack)
            img = stack[:-1, :, :]
            target = stack[-1, :, :]

        if self.standardization is not None:
            img = self.standardization(img)

        return img, target, embedding



class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


## Weighting Function for Semantic Segmentation

def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)