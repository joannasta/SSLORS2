import os
import json
import torch
import numpy as np
from tqdm import tqdm
#import gdal
import random
import rasterio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


# Pixel-level number of negative/number of positive per class
pos_weight = torch.Tensor([ 2.65263158, 27.91666667, 11.39285714, 18.82857143,  6.79775281,
        6.46236559,  0.60648148, 27.91666667, 22.13333333,  5.03478261,
       17.26315789, 29.17391304, 16.79487179, 12.88      ,  9.05797101])

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)

class RandomRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

class MaridaDataset(Dataset): 
    def __init__(self, mode='train', transform=None, standardization=None, path=dataset_path, agg_to_water=True):
        if mode == 'train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'), dtype='str')
        elif mode == 'test':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'), dtype='str')
        elif mode == 'val':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'), dtype='str')
        else:
            raise ValueError(f"Unknown mode {mode}")
        
        self.X = []  # Loaded Images
        self.y = []  # Loaded Output masks
        self.path = path
                
        with open(os.path.join(path, 'labels_mapping.txt'), 'r') as inputfile:
            labels = json.load(inputfile)

        '''if agg_to_water: 
            for k in labels.keys():
                if labels[k][14] == 1 or labels[k][13] == 1 or labels[k][12] == 1 or labels[k][11] == 1:
                    labels[k][6] = 1        # Aggregate to Water
                labels[k] = labels[k][:-4]  # Drop Mixed Water, Wakes, Cloud Shadows, Waves labels'''

        for roi in tqdm(self.ROIs, desc='Load ' + mode + ' set to memory'):
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            roi_file = os.path.join(path, 'patches', roi_folder, roi_name + '.tif')
            roi_file_cl = os.path.join(path, 'patches', roi_folder,roi_name + '_cl.tif') # Get Class Mask
            
            # Open the file using Rasterio instead of GDAL
            with rasterio.open(roi_file_cl) as ds:
                # Read the data (this is an example; depending on your needs, you may want to adjust this)
                temp = np.copy(ds.read())
                
                # Aggregation
                if agg_to_water:
                    temp[temp==15]=7          # Mixed Water to Marine Water Class
                    temp[temp==14]=7          # Wakes to Marine Water Class
                    temp[temp==13]=7          # Cloud Shadows to Marine Water Class
                    temp[temp==12]=7          # Waves to Marine Water Class
                
                temp = np.copy(temp -1)
                ds = None
                self.y.append(temp)
             # Open the file using Rasterio instead of GDAL
            with rasterio.open(roi_file) as ds:
                # Read the data (this is an example; depending on your needs, you may want to adjust this)
                temp = np.copy(ds.read())
                ds = None
                self.X.append(temp)
                
        transform = transforms.Compose([
            transforms.ToTensor(),
            RandomRotationTransform([-90, 0, 90, 180]),
            transforms.RandomHorizontalFlip()
        ])

        self.impute_nan = np.tile(bands_mean, (temp.shape[1], temp.shape[2], 1))
        self.mode = mode
        self.transform = transform
        self.standardization = transforms.Normalize(bands_mean, bands_std)
        self.length = len(self.y)
        self.path = path
        self.agg_to_water = agg_to_water
        
    def __len__(self):
        return self.length
    
    def getnames(self):
        return self.ROIs
    
    def __getitem__(self, index):
        img = self.X[index]
        target = self.y[index]
        print("beginning")
        print(f"Image shape: {img.shape}, Target shape: {target.shape}")
        #target = torch.tensor(target).float()

        # Convert the image to the proper format: WxHxC -> CxWxH (for PyTorch)
        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')  # CxWxH to WxHxC
        
        # Handle NaN values by replacing them with the imputation mean
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]
        
        
        if self.transform is not None:
            target = np.transpose(target,(2,1,0))
            print("During")
            print(f"Image shape: {img.shape}, Target shape: {target.shape}")
            stack = np.concatenate([img, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image
        
            stack = self.transform(stack)

            img = stack[:-1,:,:]
            target = stack[-1,:,:].long()  

        # Standardize (normalize) the image using the global bands_mean and bands_std
        if self.standardization is not None:
            img = self.standardization(img)
            
        print("end of get item")
        print(f"Image shape: {img.shape}, Target shape: {target.shape}")
            
        return img, target
