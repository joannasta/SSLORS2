# Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

# Libraries for models and utilities
import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from torch.autograd import Variable
from torchvision.transforms import RandomCrop, Resize

# Project-specific imports
from .magicbathynet_unet import UNet_bathy
from src.utils.finetuning_utils import calculate_metrics
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG

# ... (rest of the imports)

class MAEFineTuning(pl.LightningModule):
    def __init__(self, src_channels=3, mask_ratio=0.5):
        super().__init__()
        print("Initializing MAEFineTuning...")
        self.writer = SummaryWriter()
        self.train_step_losses = []
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_hyperparameters()
        self.run_dir = None  
        self.base_dir = "bathymetry_results"

        self.src_channels = 3
        self.mask_ratio = mask_ratio
        self.norm_param_depth = NORM_PARAM_DEPTH["agia_napa"]
        self.norm_param = np.load(NORM_PARAM_PATHS["agia_napa"])
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]

        self.adapter_layer = nn.Conv2d(3, 12, kernel_size=1)
        self.projection_head = UNet_bathy(in_channels=3, out_channels=1)
        self.cache = True
        self.criterion = CustomLoss()
        
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print("MAEFineTuning initialized.")


    def _generate_mask(self, tensor, batch_size, device, average_channels=False):
        """
        Generates a mask for non-annotated pixels in the given tensor.
        """
        print("Generating mask...")
        mask = (tensor.cpu().numpy() != 0).astype(np.float32)
        if average_channels:
            mask = np.mean(mask, axis=1) 
        mask = torch.from_numpy(mask).to(device)
        mask = mask.view(batch_size, 1, self.crop_size, self.crop_size)
        print(f"Mask shape: {mask.shape}")
        return mask

    def forward(self, images, embedding):
        print("Forward pass...")
        print(f"Images shape: {images.shape}, Embedding shape: {embedding.shape}")
        output = self.projection_head(embedding, images)
        print(f"Output shape: {output.shape}")
        return output
    
    def training_step(self, batch, batch_idx):
        print(f"Training step - Batch {batch_idx}")
        train_dir = "training_results"
        data, target, embedding = batch
        print(f"Data shape: {data.shape}, Target shape: {target.shape}, Embedding shape: {embedding.shape}")

        data, target, embedding = Variable(data.to(self.device)), Variable(target.to(self.device)), Variable(embedding.to(self.device))
        print(f"Data shape (on device): {data.shape}, Target shape (on device): {target.shape}, Embedding shape (on device): {embedding.shape}")

        size = (256, 256)
        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(1), size=size, mode='nearest')  # Added unsqueeze for target
        print(f"Data shape (after interpolation): {data.shape}, Target shape (after interpolation): {target.shape}")

        data_size = data.size()[2:]
        print(f"Data size: {data_size}")

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
            data_transform = RandomCrop(size=self.crop_size)
            target_transform = RandomCrop(size=self.crop_size)

            data = data_transform(data)
            target = target_transform(target)
            print(f"Data shape (after cropping): {data.shape}, Target shape (after cropping): {target.shape}")

        target_mask = (target.cpu().numpy() != 0).astype(np.float32)
        target_mask = torch.from_numpy(target_mask).squeeze(1).to(self.device)
        print(f"Target mask shape: {target_mask.shape}")

        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(self.device)
        print(f"Data mask shape: {data_mask.shape}")

        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float().to(self.device)  # Ensure combined_mask is on the correct device
        print(f"Combined mask shape: {combined_mask.shape}")

        data = torch.clamp(data, min=0, max=1).to(self.device)
        print(f"Data shape (clamped): {data.shape}")
        output = self(data.float(), embedding.float())
        print(f"Output shape: {output.shape}")

        loss = self.criterion(output, target, combined_mask)
        print(f"Loss: {loss.item()}")


        pred = output.data.cpu().numpy()  # Get all predictions
        gt = target.data.cpu().numpy()    # Get all ground truth values
        combined_mask_cpu = combined_mask.cpu().numpy() # Get the mask on the CPU

        print(f"Predictions shape: {pred.shape}, Ground truth shape: {gt.shape}, Combined mask shape (CPU): {combined_mask_cpu.shape}")

        masked_pred = pred * combined_mask_cpu  # Apply the mask on the CPU
        masked_gt = gt * combined_mask_cpu


        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),   
                masked_pred[0],    
                gt[0],
                train_dir   
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred[0].ravel(), masked_gt[0].ravel())
        rmse = -self.norm_param_depth * rmse

        self.log('train_loss', loss)
        self.log('train_rmse', rmse, on_step=True, on_epoch=True, prog_bar=True)

        self.total_train_loss += loss.item()
        self.train_batch_count += 1

        return loss
    
    def validation_step(self, batch, batch_idx):
        print(f"Validation step - Batch {batch_idx}")
        val_dir = "validation_results"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data, target, embedding = batch
        print(f"Data shape: {data.shape}, Target shape: {target.shape}, Embedding shape: {embedding.shape}")

        data, target = Variable(data.to(device)), Variable(target.to(device))
        print(f"Data shape (on device): {data.shape}, Target shape (on device): {target.shape}")

        size = (256, 256)
        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(1), size=size, mode='nearest')  # Added unsqueeze for target