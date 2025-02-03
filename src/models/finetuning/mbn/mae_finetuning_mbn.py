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
        target_mask = torch.from_numpy(target_mask).squeeze(1).to(device)
        print(f"Target mask shape: {target_mask.shape}")

        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(device)
        print(f"Data mask shape: {data_mask.shape}")

        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float().to(device)
        print(f"Combined mask shape: {combined_mask.shape}")

        data = torch.clamp(data, min=0, max=1).to(self.device)  # Use self.device for consistency
        target = target.to(self.device)  # Use self.device for consistency

        print(f"Data shape (clamped): {data.shape}")
        output = self(data.float(), embedding.float())  # Pass embedding
        print(f"Output shape: {output.shape}")
        combined_mask = combined_mask.to(self.device)  # Ensure mask is on the correct device

        val_loss = self.criterion(output, target, combined_mask)
        print(f"Validation loss: {val_loss.item()}")

        pred = output.data.cpu().numpy()  # Get all predictions
        gt = target.data.cpu().numpy()    # Get all ground truth
        combined_mask_cpu = combined_mask.cpu().numpy()  # Get the mask on the CPU

        print(f"Predictions shape: {pred.shape}, Ground truth shape: {gt.shape}, Combined mask shape (CPU): {combined_mask_cpu.shape}")

        masked_pred = pred * combined_mask_cpu  # Apply the mask on the CPU
        masked_gt = gt * combined_mask_cpu

        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),
                masked_pred[0],
                gt[0],
                val_dir
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel())
        rmse = -self.norm_param_depth * rmse

        self.log('val_loss', val_loss)
        self.log('val_rmse', rmse)
        self.log('val_mae', mae * -self.norm_param_depth)
        self.log('val_std_dev', std_dev * -self.norm_param_depth)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1

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
        
    def log_images(self, data, reconstructed_images, depth, dir):
        print("Logging images...")

        print(f"Data shape: {data.shape}, Reconstructed images shape: {reconstructed_images.shape}, Depth shape: {depth.shape}")

        bgr = np.asarray(np.transpose(data.cpu().numpy(), (1, 2, 0)), dtype='float32')
        rgb = bgr[:, :, [2, 1, 0]]

        depth_denorm = depth * self.norm_param_depth

        ratio = self.crop_size / self.window_size[0]
        pred_normalized = reconstructed_images
        pred_denormalized = pred_normalized * self.norm_param_depth
        pred_processed = scipy.ndimage.zoom(pred_denormalized, (1 / ratio, 1 / ratio), order=1)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(rgb)
        plt.title("Original RGB")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(depth_denorm)  # Use denormalized depth
        plt.title("Ground Truth Depth")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(pred_processed)  # Use pred_processed
        plt.title("Predicted Depth")
        plt.colorbar()
        plt.axis("off")

        dir_rel = os.path.join(self.run_dir, dir)  # Relative path
        dir_abs = os.path.abspath(dir_rel)  # Absolute path

        os.makedirs(dir_abs, exist_ok=True)  # Create (or do nothing) using absolute path

        filename = os.path.join(dir_abs, f"depth_comparison_epoch_{self.current_epoch}.png")
        print("filename", filename)
        plt.savefig(filename)  # Save using absolute path
        print(f"Saving to: {filename}")  # Print to check where you are saving

        plt.close()
        
    def log_results(self):
        if self.run_dir is None: 
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
        
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        print("Initializing CustomLoss...")  # Print initialization message

    def forward(self, output, depth, mask):
        print("CustomLoss forward pass...")
        print(f"Output shape: {output.shape}, Depth shape: {depth.shape}, Mask shape: {mask.shape}")

        mask = mask.unsqueeze(1)  # Add channel dimension to mask
        print(f"Mask shape (after unsqueeze): {mask.shape}")

        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(output, depth)
        print(f"MSE Loss shape: {loss.shape}")

        loss = (loss * mask.float()).sum()  # Apply mask and sum
        print(f"Masked Loss (summed): {loss.item()}")  # Print the scalar loss value

        non_zero_elements = mask.sum()
        print(f"Number of non-zero elements in mask: {non_zero_elements.item()}") # Print the number of unmasked elements

        rmse_loss_val = torch.sqrt(loss / non_zero_elements)
        print(f"RMSE Loss Value: {rmse_loss_val.item()}") # Print the final loss value

        return rmse_loss_val