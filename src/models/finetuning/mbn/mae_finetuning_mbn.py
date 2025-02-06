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

# TODOs and planned experiments
# - Perform ablation studies to test input combinations.
# - Add auxiliary losses for decoded data.
# - Implement attention mechanisms.
# - Experiment with freezing and unfreezing encoder/decoder.
# - Preprocess and verify decoded data quality.
# - Experiment with learnable contributions for input balance.




class MAEFineTuning(pl.LightningModule):
    def __init__(self, src_channels=3, mask_ratio=0.5):
        super().__init__()
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


    def _generate_mask(self, tensor, batch_size, device, average_channels=False):
        """
        Generates a mask for non-annotated pixels in the given tensor.
        """
        mask = (tensor.cpu().numpy() != 0).astype(np.float32)
        if average_channels:
            mask = np.mean(mask, axis=1) 
        mask = torch.from_numpy(mask)
        mask = mask.view(batch_size, 1, self.crop_size, self.crop_size)
        return mask.to(device)

    def forward(self, images,embedding):
        #batch_size = images.shape[0]
        #idx_keep, idx_mask = utils.random_token_mask(size=(batch_size, self.sequence_length), mask_ratio=self.mask_ratio, device=images.device)
        return self.projection_head(embedding,images)
    
    
    #TODO sein code mit richtiger datensatzlänge
    #TODO alle files statt 21 ??
    #TODO oVerfitten an einem batch
    #TODO gradients printen
    #TODO Unet variieren
    def training_step(self, batch,batch_idx):
        print("TRAINING STEP START")
        train_dir = "training_results"
        data,target,embedding =batch
        data, target,embedding = Variable(data.to(self.device)), Variable(target.to(self.device)), Variable(embedding.to(self.device))
        size=(256, 256)
        # Resizing data_p and label_p
        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(1), size=size, mode='nearest')
            
        data_size = data.size()[2:]  # Get the original data size

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
                data_transform = RandomCrop(size=self.crop_size)
                target_transform = RandomCrop(size=self.crop_size)
                data = data_transform(data)
                target = target_transform(target)

        print("Shape of data after RandomCrop:", data.shape)
        print("Shape of target after RandomCrop:", target.shape)

        target_mask = (target.cpu().numpy() != 0).astype(np.float32)  
        target_mask = torch.from_numpy(target_mask).to(self.device)  
        print("target mask shape before reshape",target_mask.shape)
        print("self crop size",self.crop_size)
        for i in range(target_mask.shape[0]):
            target_mask[i] = target_mask[i].reshape(self.crop_size, self.crop_size)
        print("target mask shape after reshape",target_mask.shape)
        target_mask = target_mask.squeeze(1)
            
        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(self.device)
        print("Shape of data_mask:", data_mask.shape)
        # Combine the masks
        combined_mask = target_mask * data_mask
        print("combined mask shape before threshold",combined_mask.shape)
        combined_mask = (combined_mask >= 0.5).float()
        if torch.sum(combined_mask) == 0:
            return None

        print("Shape of combined_mask:", combined_mask.shape)
        data = torch.clamp(data, min=0, max=1)
        output = self(data.float(),embedding.float())
        output = output.to(self.device)

        loss = self.criterion(output, target, combined_mask)

        pred = output.data.cpu().numpy()[0]
        print("pred shape",pred.shape)
        gt = target.data.cpu().numpy()[0]
        print("combined mask shape",combined_mask.shape)
        # Apply the mask to the predictions and ground truth
        masked_pred = pred * combined_mask.cpu().numpy()
        print("masked_pred shape",masked_pred.shape)
        masked_gt = gt * combined_mask.cpu().numpy()
        print("before image logging")
        print("masked_pred shape",masked_pred.shape)
        print("data shape",data.shape)
        print("gt shape",gt.shape)
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

        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1
        print("TRAINING STEP END")
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("VALIDATION STEP START")
        val_dir = "validation_results"
        data,target,embedding =batch
        data, target,embedding = Variable(data.to(self.device)), Variable(target.to(self.device)), Variable(embedding.to(self.device))
        size=(256, 256)
        # Resizing data_p and label_p
        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(1), size=size, mode='nearest')
            
        data_size = data.size()[2:]  # Get the original data size

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
                data_transform = RandomCrop(size=self.crop_size)
                target_transform = RandomCrop(size=self.crop_size)
                data = data_transform(data)
                target = target_transform(target)

        print("Shape of data after RandomCrop:", data.shape)
        print("Shape of target after RandomCrop:", target.shape)

        target_mask = (target.cpu().numpy() != 0).astype(np.float32)  
        target_mask = torch.from_numpy(target_mask).to(self.device)  
        print("target mask shape before reshape",target_mask.shape)
        print("self crop size",self.crop_size)
        for i in range(target_mask.shape[0]):
            target_mask[i] = target_mask[i].reshape(self.crop_size, self.crop_size)
        print("target mask shape after reshape",target_mask.shape)
        target_mask = target_mask.squeeze(1)
            
        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(self.device)
        print("Shape of data_mask:", data_mask.shape)
        # Combine the masks

        combined_mask = target_mask * data_mask
        print("combined mask shape before threshold",combined_mask.shape)
        combined_mask = (combined_mask >= 0.5).float()
        if torch.sum(combined_mask) == 0:
            return None

        print("Shape of combined_mask:", combined_mask.shape)
        data = torch.clamp(data, min=0, max=1)
        output = self(data.float(),embedding.float())
        output = output.to(self.device)
        val_loss = self.criterion(output, target, combined_mask)

        pred = output.data.cpu().numpy()[0]
        gt = target.data.cpu().numpy()[0]
            
        # Apply the mask to the predictions and ground truth
        masked_pred = pred * combined_mask.cpu().numpy()
        masked_gt = gt * combined_mask.cpu().numpy()

        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),  
                masked_pred[0], 
                gt[0],
                val_dir 
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred[0].ravel(), masked_gt[0].ravel())
        rmse = -self.norm_param_depth * rmse

        self.log('val_loss', val_loss)
        self.log('val_rmse', rmse)
        self.log('val_mae', mae * -self.norm_param_depth)
        self.log('val_std_dev', std_dev * -self.norm_param_depth)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1
        print("VALIDATION STEP END")
        return val_loss
    
    def on_train_start(self):
        self.log_results()

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch - Current learning rate: {current_lr}")
    
    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss_epoch', avg_train_loss)
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

    def on_train_end(self):
        self.writer.close()
        

    def log_images(self, data: torch.Tensor, reconstructed_images: torch.Tensor, depth: torch.Tensor,dir) -> None:
        self.log_results()
        print("logging images")
        #print(f"Data shape: {data.shape}, Reconstructed images shape: {reconstructed_images.shape}, Depth shape: {depth.shape}")

        bgr = np.asarray(np.transpose(data.cpu().numpy(),(1,2,0)), dtype='float32')
        #orig = (orig * (self.norm_param[1][:, np.newaxis, np.newaxis] - self.norm_param[0][:, np.newaxis, np.newaxis])) + self.norm_param[0][:, np.newaxis, np.newaxis]
        rgb = bgr[:, :, [2, 1, 0]] 
        #print("RGB shape",rgb.shape)
        depth_denorm = depth * self.norm_param_depth
        #print("depth shape",depth.shape)
        ratio = self.crop_size / self.window_size[0]
        pred_normalized = reconstructed_images#.squeeze(0)
        pred_denormalized = pred_normalized * self.norm_param_depth 
        #print("prediction shape",pred_denormalized.shape)
        pred_processed = scipy.ndimage.zoom(pred_normalized, (1/ratio, 1/ratio), order=1)
        #print("prediction shape",pred_processed.shape)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(rgb)
        plt.title("Original RGB")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(depth)#, cmap="viridis",vmin=0, vmax=1)
        plt.title("Ground Truth Depth")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(pred_processed)#, cmap="viridis", vmin=0, vmax=1)
        plt.title("Predicted Depth")
        plt.colorbar()
        plt.axis("off")

        dir_rel = os.path.join(self.run_dir, dir)  # Relative path
        dir_abs = os.path.abspath(dir_rel)  # Absolute path

        os.makedirs(dir_abs, exist_ok=True)  # Create (or do nothing) using absolute path

        filename = os.path.join(dir_abs, f"depth_comparison_epoch_{self.current_epoch}.png")
        #print("filename",filename)
        plt.savefig(filename)  # Save using absolute path
        #print(f"Saving to: {filename}") # Print to check where you are saving

        plt.close()

    def log_results(self):
        if self.run_dir is None:  # Only create the directory if it doesn't exist
            # Find the next available folder for the training run
            run_index = 0
            while os.path.exists(os.path.join(self.base_dir, f"run_{run_index}")):
                run_index += 1
            
            # Create the directory for the training run
            self.run_dir = os.path.join(self.base_dir, f"run_{run_index}")
            os.makedirs(self.run_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def configure_optimizers(self):
        # 1. Parameter-specific learning rates (if needed):
        params_dict = dict(self.projection_head.named_parameters())
        params = []
        for key, value in params_dict.items():
            lr = 0.0001 # Default learning rate
            if '_D' in key: # Example condition to apply different LR to some parameters
                lr = 0.0001 # Or some other value if needed
            params.append({'params': [value], 'lr': lr})

        optimizer = optim.Adam(params, lr=0.0001) # Pass the list of parameter dicts
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}  # Return a dictionary

            

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        #print("Initializing CustomLoss...")  # Print initialization message

    def forward(self, output, depth, mask):
        mask.unsqueeze(1)
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(output, depth)
        loss = (loss * mask.float()).sum() # gives \sigma_euclidean over unmasked elements
        non_zero_elements = mask.sum()
        rmse_loss_val = torch.sqrt(loss / non_zero_elements)
        return rmse_loss_val