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
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

# Libraries for models and utilities
import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from torchvision.transforms import RandomCrop

# Project-specific imports
from .magicbathynet_unet import UNet_bathy
from src.utils.finetuning_utils import calculate_metrics
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, depth, mask):
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(output, depth)
        loss = (loss * mask.float()).sum()
        non_zero_elements = mask.sum()
        rmse_loss_val = torch.sqrt(loss / non_zero_elements)
        return rmse_loss_val

class MAEFineTuning(pl.LightningModule):
    def __init__(self, src_channels=3, mask_ratio=0.5,pretrained_model=None,location="agia_napa",
                 full_finetune=False, random=False, ssl=False, model_type="mae"):

        super().__init__()
        self.writer = SummaryWriter()
        self.save_hyperparameters()

        self.run_dir = None
        self.base_dir = "bathymetry_results"
        self.pretrained_model = pretrained_model
        self.model_type = model_type

        self.src_channels = src_channels
        self.mask_ratio = mask_ratio
        self.location = location
        self.norm_param_depth = NORM_PARAM_DEPTH[location]
        self.norm_param = np.load(NORM_PARAM_PATHS[location])
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]
        self.full_finetune = full_finetune

        self.projection_head = UNet_bathy(in_channels=3, out_channels=1, model_type=self.model_type, full_finetune=self.full_finetune) 
        self.criterion = CustomLoss()

        # Metrics and Loss Tracking
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []
        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []
        self.test_image_count = 0 # Counter for logging test images uniquely

        # Enable gradient tracking for all parameters if full_finetune is True
        if self.full_finetune:
            for param in self.parameters():
                param.requires_grad = True
 
    def forward(self, images, embedding):
        """
        Forward pass for the MAEFineTuning module.
        Processes the embedding through the pretrained model if full_finetune is enabled,
        then passes the processed embedding and input images to the UNet (projection_head).

        Args:
            images (torch.Tensor): The input image tensor for the UNet encoder.
            embedding (torch.Tensor): The embedding tensor from the dataset.
                                      This can be an image (full finetune) or a feature vector (SSL).
        Returns:
            torch.Tensor: The output depth prediction from the UNet.
        """
        processed_embedding = embedding

        if self.full_finetune:
            if self.model_type == "mae":
                processed_embedding = self.pretrained_model.forward_encoder(embedding)
            elif self.model_type in ["moco", "mocogeo"]:
                print("embedding shape:", embedding.shape)
                processed_embedding = self.pretrained_model.backbone(embedding).flatten(start_dim=1) 
        
        return self.projection_head(processed_embedding, images)

    def training_step(self, batch,batch_idx):
        train_dir = "training_results"
        size=(256, 256)
        data, target, embedding = batch
        data, target,embedding = data.to(self.device), target.to(self.device), embedding.to(self.device)
        
        # Standardize resize for all images to self.crop_size (256x256)
        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(0), size=size, mode='nearest')
        
        data_size = data.size()[2:] 

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
            data_transform = RandomCrop(size=self.crop_size)
            target_transform = RandomCrop(size=self.crop_size)
            data = data_transform(data)
            target = target_transform(target)
        
        # Create masks
        target_mask = (target.cpu().numpy() != 0).astype(np.float32)
        target_mask = torch.from_numpy(target_mask).to(self.device)
        target_mask = target_mask.reshape(self.crop_size, self.crop_size)
        #target_mask = target_mask.to(device)  


        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1) # Mean over channels to get [B, H, W]
        data_mask = torch.from_numpy(data_mask).to(self.device)

        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float()
        if torch.sum(combined_mask) == 0:
            return None

        data = torch.clamp(data, min=0, max=1)
        
        output = self(data.float(),embedding.float()) 
        output = output.to(self.device)

        loss = self.criterion(output, target, combined_mask)

        pred = output.data.cpu().numpy()[0]
        gt = target.data.cpu().numpy()[0]
        
        masked_pred = pred * combined_mask.cpu().numpy()[0]
        masked_gt = gt * combined_mask.cpu().numpy()[0]

        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),
                masked_pred,
                gt,
                train_dir
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel())

        self.log('train_rmse_step', (rmse * -self.norm_param_depth), on_step=True)
        self.log('train_mae_step', (mae * -self.norm_param_depth), on_step=True)
        self.log('train_std_dev_step', (std_dev * -self.norm_param_depth), on_step=True)

        self.epoch_rmse_list.append(rmse * -self.norm_param_depth)
        self.epoch_mae_list.append(mae * -self.norm_param_depth)
        self.epoch_std_dev_list.append(std_dev * -self.norm_param_depth)

        print('Mean RMSE (per image):', rmse * -self.norm_param_depth)
        print('Mean MAE (per image):', mae * -self.norm_param_depth)
        print('Mean Std Dev (per image):', std_dev * -self.norm_param_depth)

        self.total_train_loss += loss.item()
        self.train_batch_count += 1

        return loss

    def validation_step(self, batch, batch_idx):
        val_dir = "validation_results"
        data, target, embedding = batch
        data, target,embedding = data.to(self.device), target.to(self.device), embedding.to(self.device)
        
        size = (256,256)
        # Standardize resize to self.crop_size (256x256)
        print("data shape before resize:", data.shape)
        print("target shape before resize:", target.shape)
        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(0), size=size, mode='nearest')
        
        data_size = data.size()[2:]

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
            data_transform = RandomCrop(size=self.crop_size)
            target_transform = RandomCrop(size=self.crop_size)
            data = data_transform(data)
            target = target_transform(target)
        
        # Create masks
        target_mask = (target.cpu().numpy() != 0).astype(np.float32)
        target_mask = torch.from_numpy(target_mask).to(self.device)
        #target_mask = torch.from_numpy(target_mask)  
        target_mask = target_mask.reshape(self.crop_size, self.crop_size)
        target_mask = target_mask.to(self.device)  
            
        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(self.device)
        #data_mask = data_mask.reshape(self.crop_size, self.crop_size)
        
        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float()
        if torch.sum(combined_mask) == 0:
            return None

        data = torch.clamp(data, min=0, max=1)

        output = self(data.float(),embedding.float())
        output = output.to(self.device)

        val_loss = self.criterion(output, target, combined_mask)

        pred = output.data.cpu().numpy()[0]
        gt = target.data.cpu().numpy()[0]
        masked_pred = pred * combined_mask.cpu().numpy()[0]
        masked_gt = gt * combined_mask.cpu().numpy()[0]

        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),
                masked_pred,
                gt,
                val_dir
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel())

        self.log('val_rmse', (rmse * -self.norm_param_depth), on_step=True)
        self.log('val_mae', (mae * -self.norm_param_depth), on_step=True)
        self.log('val_std_dev', (std_dev * -self.norm_param_depth), on_step=True)

        self.val_rmse_list.append(rmse * -self.norm_param_depth)
        self.val_mae_list.append(mae * -self.norm_param_depth)
        self.val_std_dev_list.append(std_dev * -self.norm_param_depth)

        print('Mean RMSE (per image):', rmse * -self.norm_param_depth)
        print('Mean MAE (per image):', mae * -self.norm_param_depth)
        print('Mean Std Dev (per image):', std_dev * -self.norm_param_depth)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1
        return val_loss

    def test_step(self, batch, batch_idx):
        test_dir = "test_results"
        test_data_batch, targets_batch, embeddings_batch = batch # Batch contains lists of tensors
        
        self.crop_size = 256
        pad_size = 32
        ratio = self.crop_size / self.window_size[0]
        # Process each individual sample in the test batch
        for img, gt,gt_e, embedding in zip(test_data_batch, targets_batch,targets_batch, embeddings_batch):
            img = img.cpu()
            gt = gt.cpu()
            gt_e = gt_e.cpu()
            embedding = embedding.unsqueeze(0)
            print("img shape before processing:", img.shape)
            print("gt shape before processing:", gt.shape)
            print("gt_e shape before processing:", gt_e.shape)
            print("embedding shape before processing:", embedding.shape)
            
            img = scipy.ndimage.zoom(img, (1,ratio, ratio), order=1)
            gt = scipy.ndimage.zoom(gt, (ratio, ratio), order=1)
            gt_e = scipy.ndimage.zoom(gt_e, (ratio, ratio), order=1)

            print("img shape after zoom:", img.shape)
            print("gt shape after zoom:", gt.shape)
            print("gt_e shape after zoom:", gt_e.shape)
            
            # Pad the image, ground truth, and eroded ground truth with reflection
            #img = np.pad(img, ((0, 0),(pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            #gt = np.pad(gt, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            #gt_e = np.pad(gt_e, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            
            
            #print("img shape after padding:", img.shape)
            #print("gt shape after padding:", gt.shape)
            #print("gt_e shape after padding:", gt_e.shape)
            
            #img = torch.clamp(img, min=0, max=1)
            
            gt = torch.from_numpy(gt).float()
            #gt = gt.unsqueeze(0) # Add batch dim

            gt_e = torch.from_numpy(gt_e).float()
            #gt_e = gt_e.unsqueeze(0)

            with torch.no_grad():#
                img = torch.from_numpy(img).float()
                img = img.unsqueeze(0)  # Add batch dimension and move to device
                print("self.full_finetune", self.full_finetune)
                outs = self(img, embedding)
                pred = outs.data.cpu().numpy().squeeze()

            #pred = pred[pad_size:-pad_size, pad_size:-pad_size]
            #img = img[pad_size:-pad_size, pad_size:-pad_size]
            #gt = gt[pad_size:-pad_size, pad_size:-pad_size]
            #gt_e = gt_e[pad_size:-pad_size, pad_size:-pad_size]

            # Generate mask for non-annotated pixels in depth data 

            gt_mask = (gt_e != 0)
            gt_mask = gt_mask.unsqueeze(0)
            gt_mask = gt_mask.reshape(self.crop_size, self.crop_size)
            gt_mask = gt_mask.to(self.device) 
            
            img_mask = (img != 0).float()
            #img_mask = torch.mean(img_mask, dim=2)
            #img_mask = img_mask.reshape(crop_size, crop_size)
            img_mask = img_mask.to(self.device) 

            self.test_image_count += 1 # Increment for unique logging per image
            print("img_mask shape:", img_mask.shape)
            print("gt_mask shape:", gt_mask.shape)
            combined_mask = img_mask*gt_mask
            
            print("combined_mask shape:", combined_mask.shape)
      
            masked_pred = pred * combined_mask.cpu().numpy()
            print("masked_pred shape:", masked_pred.shape)
            masked_gt_e = gt_e * combined_mask.cpu().numpy()
            print("masked_gt_e shape:", masked_gt_e.shape)
            
            pred = torch.from_numpy(pred).unsqueeze(0)  # Add batch dimension for logging
            gt_e = gt_e.unsqueeze(0)  # Add batch dimension for logging
            # Log images for visualization
            if batch_idx % 100 == 0:
                self.log_images(
                    img[0].cpu(),
                    pred,
                    gt_e,
                    test_dir
                )
                
            print("masked_pred type:", type(masked_pred))
            print("masked_gt_e type:", type(masked_gt_e))
            rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt_e.numpy().ravel())
            
            # Log metrics for each image in the test batch
            self.log(f'test_rmse_step_image_{self.test_image_count}', (rmse * -self.norm_param_depth), on_step=True)
            self.log(f'test_mae_step_image_{self.test_image_count}', (mae * -self.norm_param_depth), on_step=True)
            self.log(f'test_std_dev_step_image_{self.test_image_count}', (std_dev * -self.norm_param_depth), on_step=True)

            # Log epoch-level metrics (will be averaged over the epoch by PyTorch Lightning)
            self.log('test_rmse_epoch', (rmse * -self.norm_param_depth), on_step=False, on_epoch=True)
            self.log('test_mae_epoch', (mae * -self.norm_param_depth), on_step=False, on_epoch=True)
            self.log('test_std_dev_epoch', (std_dev * -self.norm_param_depth), on_step=False, on_epoch=True)

            self.test_rmse_list.append(rmse * -self.norm_param_depth)
            self.test_mae_list.append(mae * -self.norm_param_depth)
            self.test_std_dev_list.append(std_dev * -self.norm_param_depth)

            print(f'Mean RMSE for image {self.test_image_count} :', rmse * -self.norm_param_depth)
            print(f'Mean MAE for image {self.test_image_count} :', mae * -self.norm_param_depth)
            print(f'Mean Std Dev for image {self.test_image_count} :', std_dev * -self.norm_param_depth)

    def on_train_start(self):
        self.log_results()

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch {self.current_epoch} - Current learning rate: {current_lr}")

        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []

    def on_validation_epoch_start(self):
        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []

    def on_test_start(self):
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []
        self.test_image_count = 0

    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss_epoch', avg_train_loss)
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

        avg_rmse = torch.tensor(self.epoch_rmse_list).mean() if self.epoch_rmse_list else torch.tensor(0.0)
        avg_mae = torch.tensor(self.epoch_mae_list).mean() if self.epoch_mae_list else torch.tensor(0.0)
        avg_std_dev = torch.tensor(self.epoch_std_dev_list).mean() if self.epoch_std_dev_list else torch.tensor(0.0)

        self.log('avg_train_rmse', avg_rmse)
        self.log('avg_train_mae', avg_mae)
        self.log('avg_train_std_dev', avg_std_dev)

        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []

    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

        avg_rmse = torch.tensor(self.val_rmse_list).mean() if self.val_rmse_list else torch.tensor(0.0)
        avg_mae = torch.tensor(self.val_mae_list).mean() if self.val_mae_list else torch.tensor(0.0)
        avg_std_dev = torch.tensor(self.val_std_dev_list).mean() if self.val_std_dev_list else torch.tensor(0.0)

        self.log('avg_val_rmse', avg_rmse)
        self.log('avg_val_mae', avg_mae)
        self.log('avg_val_std_dev', avg_std_dev)

        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []

    def on_test_epoch_end(self):
        avg_rmse = torch.tensor(self.test_rmse_list).mean() if self.test_rmse_list else torch.tensor(0.0)
        avg_mae = torch.tensor(self.test_mae_list).mean() if self.test_mae_list else torch.tensor(0.0)
        avg_std_dev = torch.tensor(self.test_std_dev_list).mean() if self.test_std_dev_list else torch.tensor(0.0)

        self.log('avg_test_rmse', avg_rmse)
        self.log('avg_test_mae', avg_mae)
        self.log('avg_test_std_dev', avg_std_dev)

        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []

    def on_train_end(self):
        self.writer.close()

    def log_images(self, data: torch.Tensor, reconstructed_images: np.ndarray, depth: np.ndarray, dir: str) -> None:
        self.log_results()

        img_for_display = torch.clamp(data.cpu(), min=0, max=1)
        rgb_display_np = np.transpose(img_for_display.numpy(), (1, 2, 0)) # First, convert C,H,W to H,W,C
        rgb_display_np = rgb_display_np[:, :, [2, 1, 0]]
        reconstructed_images_display = reconstructed_images.astype(np.float32) * -self.norm_param_depth
        depth_display = depth.astype(np.float32) * -self.norm_param_depth

        if reconstructed_images_display.ndim == 3 and reconstructed_images_display.shape[0] == 1:
            reconstructed_images_display = reconstructed_images_display.squeeze(0)
        if depth_display.ndim == 3 and depth_display.shape[0] == 1:
            depth_display = depth_display.squeeze(0)
        
        print(f"img shape for imshow: {rgb_display_np.shape}")
        print(f"reconstructed_images shape for imshow: {reconstructed_images_display.shape}")
        print(f"depth shape for imshow: {depth_display.shape}")
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(rgb_display_np)
        plt.title("Original RGB")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(depth_display, cmap="viridis")
        plt.title("Ground Truth Depth")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(reconstructed_images_display, cmap="viridis")
        plt.title("Predicted Depth")
        plt.colorbar()
        plt.axis("off")

        dir_rel = os.path.join(self.run_dir, dir)
        dir_abs = os.path.abspath(dir_rel)

        os.makedirs(dir_abs, exist_ok=True)
        
        filename = ""
        if dir == "test_results":
            filename = os.path.join(dir_abs, f"depth_comparison_epoch_{self.current_epoch}_image_{self.test_image_count}.png")
        else:
            filename = os.path.join(dir_abs, f"depth_comparison_epoch_{self.current_epoch}_batch_{self.global_step}.png")

        plt.savefig(filename)
        print(f"Saving to: {filename}")

        plt.close()

    def log_results(self):
        if self.run_dir is None:
            run_index = 0
            while os.path.exists(os.path.join(self.base_dir, f"run_{run_index}")):
                run_index += 1
            self.run_dir = os.path.join(self.base_dir, f"run_{run_index}")
            os.makedirs(self.run_dir, exist_ok=True)

    def configure_optimizers(self):
        params_dict = dict(self.projection_head.named_parameters())
        params = []
        lr = 0.0001
        weight_decay = 0.0001
        for key, value in params_dict.items():
            if '_D' in key: # Assuming _D refers to a specific part of your model
                params+= [{'params': [value], 'lr': lr}]
            else:
                params += [{'params':[value],'lr': lr}]

        optimizer = optim.Adam(params, lr=lr)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}