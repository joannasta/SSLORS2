import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
import rasterio
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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
    def __init__(self,root_dir, mode='train',pretrained_model=None, transform=None, standardization=None, path=dataset_path,img_only_dir=None, agg_to_water=True):
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.X = []
        self.y = []
        self.means, self.stds,self.pos_weight = get_marida_means_and_stds()
        if mode == 'train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'), dtype='str')
        elif mode == 'test':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'), dtype='str')
        elif mode == 'val':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'), dtype='str')
        else:
            raise ValueError(f"Unknown mode {mode}")
        print("ROIS",self.ROIs)
        self.embeddings = []
        self.transform = transform
        self.standardization = transforms.Normalize(self.means, self.stds)
        self.length = len(self.y)
        self.path = Path(path)
        self.agg_to_water = agg_to_water
        self.pretrained_model = pretrained_model
        with open(os.path.join(path, 'labels_mapping.txt'), 'r') as inputfile:
            labels = json.load(inputfile)

        '''if agg_to_water:
            for k in labels.keys():
                if labels[k][14] == 1 or labels[k][13] == 1 or labels[k][12] == 1 or labels[k][11] == 1:
                    labels[k][6] = 1 # Aggregate to Water
                    labels[k] = labels[k][:-4] # Drop Mixed Water, Wakes, Cloud Shadows, Waves labels'''
        for roi in tqdm(self.ROIs, desc='Load ' + mode + ' set to memory'):
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            roi_file = os.path.join(path, 'patches', roi_folder, roi_name + '.tif')
            roi_file_cl = os.path.join(path, 'patches', roi_folder,roi_name + '_cl.tif')
            with rasterio.open(roi_file_cl) as ds:
                temp = np.copy(ds.read())
                if agg_to_water:
                    temp[temp==15]=7
                    temp[temp==14]=7
                    temp[temp==13]=7
                    temp[temp==12]=7
                temp = np.copy(temp -1)
                ds = None
                self.y.append(temp)
            with rasterio.open(roi_file) as ds:
                temp = np.copy(ds.read())
                ds = None
                self.X.append(temp)
        
        if self.X:
            self.impute_nan = np.tile(self.means, (self.X[0].shape[1], self.X[0].shape[2], 1))
        
        self.length = len(self.X)

        self.processor = DatasetProcessor(
            img_dir = self.path / "patches",
            depth_dir = self.path / "patches",
            output_dir = self.root_dir / "marida" / f"processed_{mode}",
            img_only_dir = str(img_only_dir),
            depth_only_dir = str(self.root_dir / "marida"/ f"depth_only_{mode}"),
            split_type = mode
        )
        self.img_only_dir = img_only_dir
        self._create_embeddings()
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RandomRotationTransform([-90, 0, 90, 180]),
                transforms.RandomHorizontalFlip()
            ])

    def __len__(self):
        return self.length

    def getnames(self):
        return self.ROIs

    def _create_embeddings(self):
        hydro_dataset = HydroDataset(path_dataset=self.processor.img_only_dir, bands=["B02", "B03", "B04"])
        self.embeddings = []
        for idx in range(len(hydro_dataset)):
            img = hydro_dataset[idx]
            img = img.unsqueeze(0).to(self.pretrained_model.device)
            img = F.interpolate(img, size=(256,256), mode='nearest')

            with torch.no_grad():
                embedding = self.pretrained_model.forward_encoder(img)
                self.embeddings.append(embedding.cpu())
        self.embeddings = torch.stack(self.embeddings).cpu()


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
