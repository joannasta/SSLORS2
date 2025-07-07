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
from torch.autograd import Variable
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
        loss = (loss * mask.float()).sum() # gives \sigma_euclidean over unmasked elements
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
        self.base_lr = 0.0001

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
                embedding = embedding.squeeze(0)
                embedding = self.pretrained_model.forward_encoder(embedding)
                embedding = embedding.unsqueeze(0)
            elif self.model_type in ["moco", "mocogeo"]:
                print("embedding shape:", embedding.shape)
                embedding = self.pretrained_model.backbone(embedding).flatten(start_dim=1) 
        
        return self.projection_head(embedding, images)

    def training_step(self, batch,batch_idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dir = "training_results"
        data, target, embedding = batch
        data, target,embedding  = Variable(data.to(self.device)), Variable(target.to(self.device)), Variable(embedding.to(self.device))
        size = (256, 256)
        batch_size = data.size(0)
        # Standardize resize for all images to self.crop_size (256x256)
        #data = F.interpolate(data, size=size, mode='nearest')
        #target = F.interpolate(target.unsqueeze(1), size=size, mode='nearest')

        data = F.interpolate(data, size=size, mode='bilinear', align_corners=False)
        target = F.interpolate(target.unsqueeze(1), size=size, mode='bilinear', align_corners=False)

        data_size = data.size()[2:]
        
        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
                data_transform = RandomCrop(size=self.crop_size)
                target_transform = RandomCrop(size=self.crop_size)
                data = data_transform(data)
                target = target_transform(target)
            
        target_mask = (target.cpu().numpy() != 0).astype(np.float32)  
        target_mask = torch.from_numpy(target_mask).to(self.device)  
        
        for i in range(target_mask.shape[0]):
            target_mask[i] = target_mask[i].reshape(self.crop_size, self.crop_size)
        
        target_mask = target_mask.squeeze(1)
            
        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(self.device)
        
        # Combine the masks
        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float()
        
        if torch.sum(combined_mask) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        data = torch.clamp(data, min=0, max=1)
        
        output = self(data.float(),embedding.float()) 
        output = output.to(self.device)

        loss = self.criterion(output, target, combined_mask)

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
                train_dir   
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel())

        self.log('train_rmse_step', (rmse * -self.norm_param_depth), on_step=True)
        self.log('train_mae_step', (mae * -self.norm_param_depth), on_step=True) 
        self.log('train_std_dev_step', (std_dev * -self.norm_param_depth), on_step=True) 

        # Append to the lists for epoch-level calculation
        self.epoch_rmse_list.append(rmse * -self.norm_param_depth)
        self.epoch_mae_list.append(mae * -self.norm_param_depth)
        self.epoch_std_dev_list.append(std_dev * -self.norm_param_depth)
        
        # Logging (train function example - adjust as needed):
        print('Mean RMSE (per image):', rmse * -self.norm_param_depth)
        print('Mean MAE (per image):', mae * -self.norm_param_depth)
        print('Mean Std Dev (per image):', std_dev * -self.norm_param_depth)
        self.log('train_loss', loss)

        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        val_dir = "validation_results"
        data, target, embedding = batch
        data, target,embedding  = Variable(data.to(self.device)), Variable(target.to(self.device)), Variable(embedding.to(self.device))
        size = (256, 256)
        #data = F.interpolate(data, size=size, mode='nearest')
        #target = F.interpolate(target.unsqueeze(1), size=size, mode='nearest')

        data = F.interpolate(data, size=size, mode='bilinear', align_corners=False)
        target = F.interpolate(target.unsqueeze(1), size=size, mode='bilinear', align_corners=False)
            
        data_size = data.size()[2:]
        
        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
                data_transform = RandomCrop(size=self.crop_size)
                target_transform = RandomCrop(size=self.crop_size)
                data = data_transform(data)
                target = target_transform(target)
                
        target_mask = (target.cpu().numpy() != 0).astype(np.float32)  
        target_mask = torch.from_numpy(target_mask).to(self.device)  
        
        for i in range(target_mask.shape[0]):
            target_mask[i] = target_mask[i].reshape(self.crop_size, self.crop_size)
        
        target_mask = target_mask.squeeze(1)
            
        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask).to(self.device)
        
        # Combine the masks
        combined_mask = target_mask * data_mask
        print(f"DEBUG: combined_mask (pre-threshold) sum: {torch.sum(combined_mask)}")
        combined_mask = (combined_mask >= 0.5).float()

        print(f"DEBUG: Batch {batch_idx}")
        print(f"DEBUG: target_mask sum: {torch.sum(target_mask)}")
        print(f"DEBUG: data_mask sum: {torch.sum(data_mask)}")
        
        print(f"DEBUG: combined_mask (post-threshold) sum: {torch.sum(combined_mask)}")

        
        if torch.sum(combined_mask) == 0:
            print("sum of combined mask is 0")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        data = torch.clamp(data, min=0, max=1)
        
        output = self(data.float(),embedding.float()) 
        print(f"DEBUG: output sum: {torch.sum(output)}")
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

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel()) # Assuming you have masking in validation too

        self.log('val_rmse', (rmse * -self.norm_param_depth), on_step=True) 
        self.log('val_mae', (mae * -self.norm_param_depth), on_step=True)  
        self.log('val_std_dev', (std_dev * -self.norm_param_depth), on_step=True) 

        # Append to lists
        self.val_rmse_list.append(rmse * -self.norm_param_depth)
        self.val_mae_list.append(mae * -self.norm_param_depth)
        self.val_std_dev_list.append(std_dev * -self.norm_param_depth)

        # Logging (train function example - adjust as needed):
        print('Mean RMSE (per image):', rmse * -self.norm_param_depth)
        print('Mean MAE (per image):', mae * -self.norm_param_depth)
        print('Mean Std Dev (per image):', std_dev * -self.norm_param_depth)

        self.log('val_loss', val_loss)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1
        return val_loss

    def test_step(self, batch, batch_idx):
        test_dir = "test_results"
        pad_size = 32
        crop_size = 256
        ratio = crop_size / self.window_size[0]

        test_data, targets, embeddings = batch
        size = (256, 256)
        idx = 0

        for data, target,embedding in zip(test_data, targets,embeddings):
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
            embedding = embedding.unsqueeze(0)

            target_e = target.clone()
            data, target,embedding = Variable(data.to(self.device)), Variable(target.to(self.device)), Variable(embedding.to(self.device))

            data = scipy.ndimage.zoom(data.cpu().numpy(), (1,1,ratio, ratio), order=1)
            target = scipy.ndimage.zoom(target.cpu(), (1, ratio, ratio), order=1)
            target_e = scipy.ndimage.zoom(target_e.cpu(), (1, ratio, ratio), order=1)

            data = np.pad(data, ((0, 0),(0, 0),(pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            target = np.pad(target, ((0,0),(pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            target_e = np.pad(target_e, ((0,0),(pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

            data = data.transpose((1, 2, 3, 0)).squeeze(3)
            data = np.expand_dims(data, axis=0)
            data = torch.from_numpy(data).cuda()

            # Do the inference on the whole image
            with torch.no_grad():
                outs = self(data.float(),embedding.float())
                pred = outs.data.cpu().numpy().squeeze()

            # Remove padding from prediction
            pred = pred[pad_size:-pad_size, pad_size:-pad_size]
            data = data[:,:,pad_size:-pad_size, pad_size:-pad_size]
            target = target[:,pad_size:-pad_size, pad_size:-pad_size]
            target_e = target_e[:,pad_size:-pad_size, pad_size:-pad_size]

            # Generate mask for non-annotated pixels in depth data
            target_mask = (target_e != 0).astype(np.float32)
            target_mask = torch.from_numpy(target_mask).unsqueeze(0)
            target_mask = target_mask.reshape(crop_size, crop_size)
            target_mask = target_mask.to(self.device)

            data_mask = (data.cpu().numpy()!= 0).astype(np.float32)
            data_mask = np.mean(data_mask, axis=1)
            data_mask = torch.from_numpy(data_mask)
            data_mask = data_mask.to(self.device)

            combined_mask = data_mask * target_mask

            # Crucial check: only calculate metrics if there are valid pixels in the mask
            if torch.sum(combined_mask) == 0:
                print(f"Warning: Combined mask is empty for image {idx} in batch {batch_idx}. Skipping metric calculation.")
                rmse, mae, std_dev = 0.0, 0.0, 0.0 # Or some other placeholder value like -1 or log a specific message
            else:
                masked_pred = pred * combined_mask.cpu().numpy()
                masked_gt_e = target_e * combined_mask.cpu().numpy()
                rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt_e.ravel())


            self.log_images(
                    data.cpu().numpy()[0],
                    masked_pred[0] if 'masked_pred' in locals() else np.zeros_like(pred[0]), # Handle case where masked_pred might not be defined
                    masked_gt_e[0] if 'masked_gt_e' in locals() else np.zeros_like(target_e[0]),
                    test_dir
                )

            idx +=1
            self.log(f'test_rmse_step for image {idx} ', (rmse * -self.norm_param_depth), on_step=True)
            self.log(f'test_mae_step for image {idx} ', (mae * -self.norm_param_depth), on_step=True)
            self.log(f'test_std_dev_step for image {idx} ', (std_dev * -self.norm_param_depth), on_step=True)

            # Append to lists (if you want to calculate metrics on the entire test set at the end)
            self.test_rmse_list.append(rmse * -self.norm_param_depth)
            self.test_mae_list.append(mae * -self.norm_param_depth)
            self.test_std_dev_list.append(std_dev * -self.norm_param_depth)

            # Logging (train function example - adjust as needed):
            print(f'Mean RMSE for image {idx} :', rmse * -self.norm_param_depth)
            print(f'Mean MAE for image {idx} :', mae * -self.norm_param_depth)
            print(f'Mean Std Dev for image {idx} :', std_dev * -self.norm_param_depth)

    def on_train_start(self):
        self.log_results()

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch {self.current_epoch} - Current learning rate: {current_lr}") # Include epoch number

        # Initialize lists to store metrics for the epoch
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

        # Calculate and log epoch-level metrics
        avg_rmse = torch.tensor(self.epoch_rmse_list).mean()
        avg_mae = torch.tensor(self.epoch_mae_list).mean()
        avg_std_dev = torch.tensor(self.epoch_std_dev_list).mean()

        self.log('avg_train_rmse', avg_rmse)
        self.log('avg_train_mae', avg_mae)
        self.log('avg_train_std_dev', avg_std_dev)

        # Clear the lists for the next epoch - CRUCIAL
        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []

    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

        # Calculate and log epoch-level validation metrics
        avg_rmse = torch.tensor(self.val_rmse_list).mean()
        avg_mae = torch.tensor(self.val_mae_list).mean()
        avg_std_dev = torch.tensor(self.val_std_dev_list).mean()

        self.log('avg_val_rmse', avg_rmse)
        self.log('avg_val_mae', avg_mae)
        self.log('avg_val_std_dev', avg_std_dev)

        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []


    def on_test_epoch_end(self): 
        
        avg_rmse = torch.tensor(self.test_rmse_list).mean()
        avg_mae = torch.tensor(self.test_mae_list).mean()
        avg_std_dev = torch.tensor(self.test_std_dev_list).mean()

        self.log('avg_test_rmse', avg_rmse)
        self.log('avg_test_mae', avg_mae)
        self.log('avg_test_std_dev', avg_std_dev)

        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []


    def on_train_end(self):
        self.writer.close()

    def log_images(self, data: torch.Tensor, reconstructed_images: torch.Tensor, depth: torch.Tensor,dir) -> None:
        self.log_results()
        bgr = np.asarray(np.transpose(data,(1,2,0)), dtype='float32') #data.cpu().numpy()
        rgb = bgr[:, :, [2, 1, 0]] 
        depth_denorm = depth * self.norm_param_depth
        ratio = self.crop_size / self.window_size[0]
        pred_normalized = reconstructed_images#.squeeze(0)
        pred_denormalized = pred_normalized * self.norm_param_depth 
        pred_processed = scipy.ndimage.zoom(pred_normalized, (1/ratio, 1/ratio), order=1)

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
        if dir == "test_results":
            filename = os.path.join(dir_abs, f"depth_comparison_epoch_{self.current_epoch}_image_{self.test_image_count}.png")
            self.test_image_count += 1
        else:
            filename = os.path.join(dir_abs, f"depth_comparison_epoch_{self.current_epoch}.png")
        plt.savefig(filename)  # Save using absolute path
        print(f"Saving to: {filename}") # Print to check where you are saving

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
        # 1. Parameter-specific learning rates (if needed):
        params_dict = dict(self.projection_head.named_parameters())
        params = []
        lr = 0.0001
        for key, value in params_dict.items():
            if '_D' in key: 
                params+= [{'params': [value], 'lr': lr}]
            else:
                params += [{'params':[value],'lr': lr}] 

        optimizer = optim.Adam(params, lr=lr) # Pass the list of parameter dicts
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}