# Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import scipy

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

# Libraries for models and utilities
import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import torchvision.transforms.functional as F
import matplotlib.patches as mpatches

# Project-specific imports
from .marida_unet  import UNet_Marida
from src.data.marida.marida_dataset import gen_weights
from src.utils.finetuning_utils import calculate_metrics
from config import get_marida_means_and_stds, labels_marida,cat_mapping_marida
from src.utils.finetuning_utils import metrics_marida,confusion_matrix 
from src.models.mae import MAE
from src.models.moco import MoCo
from src.models.moco_geo import MoCoGeo
from src.data.hydro.hydro_dataset import HydroDataset


class InputChannelAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)

class MAEFineTuning(pl.LightningModule):
    def __init__(self, src_channels=11, mask_ratio=0.5, learning_rate=1e-4,pretrained_weights=None,pretrained_model=None,model_type="mae",full_finetune=True,location="agia_napa"):
        super().__init__()
        self.writer = SummaryWriter()
        self.train_step_losses = []
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_hyperparameters()
        self.run_dir = None 
        self.location = location 
        self.base_dir = "marine_debris_results"
        self.pretrained_model=pretrained_model
        self.model_type = model_type
        self.full_finetune = full_finetune
        
        if self.full_finetune:
            for param in self.parameters():
                param.requires_grad = True


        self.src_channels = src_channels  
        if self.pretrained_model is not None:
            self.pretrained_in_channels = 11
        else:
            self.pretrained_in_channels = src_channels
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.means,self.stds,self.pos_weight = get_marida_means_and_stds()

        self.test_image_count = 0

        self.input_adapter = InputChannelAdapter(self.src_channels, self.pretrained_in_channels)
        self.projection_head = UNet_Marida(input_channels=self.pretrained_in_channels, 
                                           out_channels=src_channels,model_type=self.model_type)
        
        self.test_step_outputs = []

        self.agg_to_water = True
        self.weight_param = 1.03
        self.class_distr = torch.Tensor([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
                                        0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052])
        if self.agg_to_water:
            agg_distr = sum(self.class_distr[-4:])
            self.class_distr[6] += agg_distr       
            self.class_distr = self.class_distr[:-4]    

        self.weight = gen_weights(self.class_distr, c = self.weight_param)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=self.weight)

        self.total_train_loss = 0.0
        self.total_test_loss = 0.0
        self.train_batch_count = 0
        self.test_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def forward(self, images, embedding):

        # Adapt input images channels for UNet's encoder path
        #images = self.input_adapter(images) 
        print("images shape", images.shape)
        print("self.full_finetune", self.full_finetune)
        print("self.model_type", self.model_type)
        
        processed_embedding_for_unet = None

        if self.full_finetune:
            if self.model_type == "mae":
                processed_embedding_for_unet = embedding.squeeze(0)
                processed_embedding_for_unet = self.pretrained_model.forward_encoder(processed_embedding_for_unet)
                processed_embedding_for_unet = processed_embedding_for_unet.unsqueeze(0)
            elif self.model_type == "moco":
                moco_features = self.pretrained_model.backbone(embedding).flatten(start_dim=1)
                processed_embedding_for_unet = moco_features
            elif self.model_type == "mocogeo":
                processed_embedding_for_unet = embedding.squeeze(0)
                processed_embedding_for_unet = self.pretrained_model.backbone(processed_embedding_for_unet).flatten(start_dim=1)
        else:
            processed_embedding_for_unet = embedding

        return self.projection_head(images, processed_embedding_for_unet)


        return self.projection_head(images,embedding)
    
    def training_step(self, batch, batch_idx):
        train_dir = "train_results"
        data, target,embedding = batch
        target = target.squeeze(1) 
        batch_size = data.shape[0]
        logits = self(data,embedding)
        target = target.long()
        loss = self.criterion(logits, target)


        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),     
                logits[0],    
                target[0].cpu(),   
                dir = train_dir 
            )

        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1

        return loss


    def validation_step(self, batch, batch_idx):
        val_dir="val_results"
        data, target,embedding = batch
        target = target.squeeze(1) 
        batch_size = data.shape[0]
        logits = self(data,embedding)
        target = target.long()
        loss = self.criterion(logits, target)

        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),     
                logits[0],    
                target[0].cpu(),   
                dir = val_dir 
            )

        self.log('val_loss', loss)
        if not hasattr(self, 'total_val_loss'):
            self.total_val_loss = 0.0
            self.val_batch_count = 0
        self.total_val_loss += loss.item()
        self.val_batch_count += 1

        return loss
    
    
    def test_step(self, batch, batch_idx):
        test_dir = "test_results"
        data, target, embedding = batch
        target = target.squeeze(1) 
        batch_size = data.shape[0]
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        print(f"Batch {batch_idx}: Unique target values: {torch.unique(target)}")

        logits = self(data, embedding)
        target = target.long()
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
                dir=test_dir)

        visual_target_image = np.zeros((256, 256))
        min_length = min(len(target_labels), len(pixel_locations[0]))

        for i in range(min_length):
            row = pixel_locations[0][i]
            col = pixel_locations[1][i]
            visual_target_image[row, col] = target_labels[i]

        img = data[0].clone().cpu()
        img = (img *self.stds[:, None, None]) + ( self.means[:, None, None]) 
        img = img[1:4, :, :]  
        img = img[[2, 1, 0], :, :]  # Swap BGR to RGB

        img = np.clip(img, 0, np.percentile(img, 99))
        img = (img / img.max() * 255).to(torch.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))  # Adjusted figure size
        img = img.permute(1,2,0)
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(visual_target_image)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(visual_image)  # Use a colormap for labels
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        # Create a colormap
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=12) # Use 12, because there are 13 classes 0-12

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for matplotlib >= 3.1
        cbar = fig.colorbar(sm, ax=axes[2], shrink=0.8, aspect=20) # Adjust shrink and aspect as needed
        cbar.set_ticks(np.arange(0, 13)) # Set ticks for each class
        keys = list(cat_mapping_marida.keys())[:-4]
        cbar.set_ticklabels(list(cat_mapping_marida.keys())[:-2]) # Set tick labels from category mapping

        # Save and show
        dir_rel = os.path.join(self.run_dir, test_dir)
        dir_abs = os.path.abspath(dir_rel)
        filename = os.path.join(dir_abs, f"segmentation_comparison_{self.current_epoch}_image_{self.test_image_count}.png")
        self.test_image_count += 1
        
        plt.tight_layout()
        plt.savefig(filename)  # Save using absolute path
        plt.show()
        return loss


    def on_train_start(self):
        self.log_results()

    def on_test_epoch_end(self):
        all_predictions = []
        all_targets = []
        for output in self.test_step_outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        acc = metrics_marida( all_targets,all_predictions,)
        for key, value in acc.items():
            prefix = "test"
            self.log(f"{prefix}_{key}", value) 
        print(f"Evaluation: {acc}")
        conf_mat = confusion_matrix(all_targets, all_predictions, labels_marida)
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))


        self.test_step_outputs.clear() # Clear the outputs.
        

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch - Current learning rate: {current_lr}")
    
    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss', avg_val_loss, on_epoch=True)
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss', avg_train_loss)
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

    def on_train_end(self):
        self.writer.close()
    
    def log_images(self, original_images: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor,dir) -> None:
        print("log images")
        self.log_results()
        img= original_images.cpu().numpy()
        target = target.detach().cpu().numpy()
        prediction = torch.argmax(prediction, dim=0).detach().cpu().numpy()

        img = (img *self.stds[:, None, None]) + ( self.means[:, None, None]) 
        img = img[1:4, :, :]  
        img = img[[2, 1, 0], :, :]  # Swap BGR to RGB

        img = np.clip(img, 0, np.percentile(img, 99))
        img = (img / img.max() * 255).astype('uint8')
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with 1 row and 2 columns
        img = img.transpose(1,2,0)
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Plot original image
        axes[1].imshow(target)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        # Plot prediction (as class labels)
        axes[2].imshow(prediction)  # Use a colormap for labels
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        dir_rel = os.path.join(self.run_dir, dir)  # Relative path
        dir_abs = os.path.abspath(dir_rel)  # Absolute path

        os.makedirs(dir_abs, exist_ok=True)  # Create (or do nothing) using absolute path
        if dir == "test_results":
            filename = os.path.join(dir_abs, f"segmentation_comparison_{self.current_epoch}_image_{self.test_image_count}.png")
            self.test_image_count += 1
        else:
            filename = os.path.join(dir_abs, f"segmentation_comparison_epoch_{self.current_epoch}.png")
        plt.savefig(filename)  # Save using absolute path
        print(f"Saving to: {filename}") # Print to check where you are saving

        plt.close()

    def log_results(self):
        if self.run_dir is None:  
            run_index = 0
            while os.path.exists(os.path.join(self.base_dir, f"run_{run_index}")):
                run_index += 1
            
            self.run_dir = os.path.join(self.base_dir, f"run_{run_index}")
            os.makedirs(self.run_dir, exist_ok=True)
                
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40], gamma=0.1, verbose=True)
        return [optimizer], [scheduler]