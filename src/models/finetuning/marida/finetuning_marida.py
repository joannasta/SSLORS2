
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm
import torchvision.transforms.functional as F

from torch.utils.tensorboard import SummaryWriter
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

from .marida_unet  import UNet_Marida
from src.data.marida.marida_dataset import gen_weights
from config import get_marida_means_and_stds, labels_marida,cat_mapping_marida
from src.utils.finetuning_utils import metrics_marida,confusion_matrix 
from src.models.mae import MAE
from src.models.moco import MoCo
from src.models.geography_aware import GeographyAware
from src.models.ocean_aware import OceanAware
from src.data.hydro.mae.hydro_dataset import HydroDataset


class InputChannelAdapter(nn.Module):
    """1x1 conv to adapt input channels to expected model channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class FineTuningMARIDA(pl.LightningModule):
    """Fine-tune UNet_Marida on MARIDA segmentation with optional SSL embeddings."""
    def __init__(self, src_channels=11, mask_ratio=0.5, learning_rate=1e-4,pretrained_weights=None,pretrained_model=None,model_type="mae",full_finetune=True,location="agia_napa"):
        super().__init__()
        self.writer = SummaryWriter()
        self.train_step_losses = []
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_hyperparameters()
        
        # Run/output dirs
        self.run_dir = None 
        self.location = location 
        self.base_dir = "marine_debris_results"
        
        # SSL model setup
        self.pretrained_model=pretrained_model
        self.model_type = model_type
        self.full_finetune = full_finetune
        
        if self.full_finetune:
            for param in self.parameters():
                param.requires_grad = True

        # Data normalization and learning params
        self.src_channels = src_channels  
        print("number of source channels:",self.src_channels)
        if self.pretrained_model is not None:
            self.pretrained_in_channels = 11
        else:
            self.pretrained_in_channels = src_channels
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.means,self.stds,self.pos_weight = get_marida_means_and_stds()

        self.test_image_count = 0

        # Adapter from current input channels to SSL/pretrained expected channels
        self.input_adapter = InputChannelAdapter(self.src_channels, self.pretrained_in_channels)
        
        # UNet head outputs 11 classes
        self.projection_head = UNet_Marida(input_channels=self.pretrained_in_channels, 
                                           out_channels=11,model_type=self.model_type)
        
        self.test_step_outputs = []

        global class_distr

         # Class distribution for weightin
        self.agg_to_water = True
        self.weight_param = 1.03
        self.class_distr = torch.Tensor([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
                                        0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052])
        if self.agg_to_water:
            agg_distr = sum(self.class_distr[-4:]) # Density of Mixed Water, Wakes, Cloud Shadows, Waves
            self.class_distr[6] += agg_distr       # To Water
            self.class_distr = self.class_distr[:-4]    # Drop Mixed Water, Wakes, Cloud Shadows, Waves  

        self.weight =  gen_weights(self.class_distr, c = self.weight_param)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=self.weight)

        # Tracking for losses
        self.total_train_loss = 0.0
        self.total_test_loss = 0.0
        self.train_batch_count = 0
        self.test_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def forward(self, images, embedding):
        """Forward pass, optionally encode embedding with SSL backbone, then UNet head."""
        processed_embedding = None

        if self.full_finetune:
            if self.model_type == "mae" or self.model_type == "mae_ocean":
                # Expect ViT-like token embeddings
                embedding = embedding.squeeze(1)
                processed_embedding = self.pretrained_model.forward_encoder(embedding)
                processed_embedding = processed_embedding.unsqueeze(0)
            elif self.model_type == "moco":
                embedding = embedding.squeeze(1)
                processed_embedding = self.pretrained_model.backbone(embedding).flatten(start_dim=1)
            elif self.model_type == "geo_aware":
                embedding = embedding.squeeze(1)
                processed_embedding = self.pretrained_model.backbone(embedding).flatten(start_dim=1)
            elif self.model_type == "ocean_aware":
                embedding = embedding.squeeze(1)
                processed_embedding = self.pretrained_model.backbone(embedding).flatten(start_dim=1)
        else:
            processed_embedding = embedding

        return self.projection_head(images, processed_embedding)
    
    def training_step(self, batch, batch_idx):
        """Compute CE loss and log images periodically."""
        train_dir = "train_results"
        data, target,embedding = batch
        images = data
        batch_size = data.shape[0]
        #data = data.permute(0,3,1,2)
        logits = self(data,embedding)
        data = images
        target = target.long()
        target = target.squeeze(1)
        loss = self.criterion(logits, target)


        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),     
                logits[0],    
                target[0].cpu(),   
                log_dir = train_dir 
            )

        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1
        print("training loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        """Validation step: CE loss + optional visualization."""
        val_dir="val_results"
        data, target,embedding = batch
        images = data
        batch_size = data.shape[0]
        #data = data.permute(0,3,1,2)
        logits = self(data,embedding)
        data = images
        target = target.long()
        target = target.squeeze(1)
        loss = self.criterion(logits, target)

        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),     
                logits[0],    
                target[0].cpu(),   
                log_dir = val_dir 
            )
            
        self.log('val_loss', loss)
        if not hasattr(self, 'total_val_loss'):
            self.total_val_loss = 0.0
            self.val_batch_count = 0
            
        self.total_val_loss += loss.item()
        self.val_batch_count += 1
        print("validation loss", loss)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        """Test step: compute CE loss, collect predictions for metrics, and save visualization."""
        test_dir = "test_results"
        data, target, embedding = batch
        batch_size = data.shape[0]
        logits = self(data,embedding)
        target = target.long()
        target = target.squeeze(1)
        loss = self.criterion(logits, target)
        self.log("test_loss", loss)

        # Accuracy metrics only on annotated pixels
        logits_moved = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
        logits_reshaped = logits_moved.reshape((-1, 11))  
        target_reshaped = target.reshape(-1)
        mask = target_reshaped != -1
        logits_masked = logits_reshaped[mask]
        target_masked = target_reshaped[mask]
        probs = nn.functional.softmax(logits_masked, dim=1).cpu().numpy()
        target_cpu = target_masked.cpu().numpy()

        # Store predictions and ground truth in lists
        self.test_step_outputs.append({
            "predictions": probs.argmax(1).tolist(),
            "targets": target_cpu.tolist(),
        })

        # Visualize pixel labels
        mask_2d = target != -1
        pixel_locations = np.where(mask_2d[0].cpu().numpy())

        predicted_labels = probs.argmax(1)
        target_labels = target_cpu

        visual_image = np.zeros((256, 256))
        min_length = min(len(predicted_labels), len(pixel_locations[0]))

        for i in range(min_length):
            row = pixel_locations[0][i]
            col = pixel_locations[1][i]
            visual_image[row, col] = predicted_labels[i]


        self.log_images(
                data[0].cpu(),
                logits[0],
                target[0].cpu(),
                log_dir=test_dir)

        visual_target_image = np.zeros((256, 256))
        min_length = min(len(target_labels), len(pixel_locations[0]))

        for i in range(min_length):
            row = pixel_locations[0][i]
            col = pixel_locations[1][i]
            visual_target_image[row, col] = target_labels[i]

        # Denormalize and prepare RGB image for plotting
        img = data[0].clone().cpu()
        img = (img *self.stds[:,None, None]) + ( self.means[:,None, None]) 
        img = img[1:4, :, :]  
        img = img[[2, 1, 0], :, :]  # Swap BGR to RGB

        img = np.clip(img, 0, np.percentile(img, 99))
        img = (img / img.max() * 255).to(torch.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(14, 6)) 
        img = img.permute(1,2,0)
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(visual_target_image)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(visual_image)  
        axes[2].set_title(f"Prediction for {self.model_type}")
        axes[2].axis('off')

        # Create a colormap
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=12) 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=axes[2], shrink=0.8, aspect=20) 
        cbar.set_ticks(np.arange(0, 13)) 
        keys = list(cat_mapping_marida.keys())[:-4]
        cbar.set_ticklabels(list(cat_mapping_marida.keys())[:-2])

        dir_rel = os.path.join(self.run_dir, test_dir)  
        dir_abs = os.path.abspath(dir_rel)  

        os.makedirs(dir_abs, exist_ok=True)  
        if test_dir == "test_results":
            filename = os.path.join(dir_abs, f"segmentation_comparison_{self.current_epoch}_image_{self.test_image_count}.png")
            self.test_image_count += 1
        else:
            filename = os.path.join(dir_abs, f"segmentation_comparison_epoch_{self.current_epoch}.png")
            
        plt.savefig(filename)  
        plt.close()
        return loss


    def on_train_start(self):
        """Create run directory."""
        self.log_results()

    def on_test_epoch_end(self):
        """Aggregate test predictions and report MARIDA metrics + confusion matrix."""
        all_predictions = []
        all_targets = []
        for output in self.test_step_outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        labels_for_cm = list(cat_mapping_marida.keys())[:-4]

        acc = metrics_marida(all_targets, all_predictions)
        for key, value in acc.items():
            prefix = "test"
            self.log(f"{prefix}_{key}", value)
        
        conf_mat = confusion_matrix(all_targets, all_predictions, labels_for_cm)
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))

        self.test_step_outputs.clear()
        

    def on_train_epoch_start(self):
        """Log current LR at epoch start."""
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch - Current learning rate: {current_lr}")
    
    def on_validation_epoch_end(self):
        """Log average validation loss for the epoch."""
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss', avg_val_loss, on_epoch=True)
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

    def on_train_epoch_end(self):
        """Log average training loss for the epoch."""
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss', avg_train_loss)
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

    def on_train_end(self):
        """Close TensorBoard writer."""
        self.writer.close()
    
    def log_images(self, original_images: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor,log_dir) -> None:
        """Save side-by-side RGB, GT segmentation, and predicted segmentation."""
        self.log_results()
        img= original_images.cpu().numpy()
        target = target.detach().cpu().numpy()
        prediction = torch.argmax(prediction, dim=0).detach().cpu().numpy()

        img = (img *self.stds[:, None, None]) + ( self.means[:, None, None]) 
        img = img[1:4, :, :]  
        img = img[[2, 1, 0], :, :]  # Swap BGR to RGB

        img = np.clip(img, 0, np.percentile(img, 99))
        img = (img / img.max() * 255).astype('uint8')
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))  
        img = img.transpose(1,2,0)

        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(target)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(prediction)  
        axes[2].set_title(f"Prediction for model {self.model_type}")
        axes[2].axis('off')

        dir_rel = os.path.join(self.run_dir, log_dir)  
        dir_abs = os.path.abspath(dir_rel) 

        os.makedirs(dir_abs, exist_ok=True)  
        if dir == "test_results":
            filename = os.path.join(dir_abs, f"segmentation_comparison_{self.current_epoch}_image_{self.test_image_count}.png")
            self.test_image_count += 1
        else:
            filename = os.path.join(dir_abs, f"segmentation_comparison_epoch_{self.current_epoch}.png")
        plt.savefig(filename)  
        print(f"Saving to: {filename}") 
        plt.close()

    def log_results(self):
        """Create unique run directory under base_dir."""
        if self.run_dir is None:  
            run_index = 0
            while os.path.exists(os.path.join(self.base_dir, f"run_{run_index}")):
                run_index += 1
            self.run_dir = os.path.join(self.base_dir, f"run_{run_index}")
            os.makedirs(self.run_dir, exist_ok=True)
                
    def configure_optimizers(self):
        """Adam optimizer with MultiStep LR scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40], gamma=0.1)
        return [optimizer], [scheduler]