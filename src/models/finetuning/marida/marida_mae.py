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
    def __init__(self, src_channels=11, mask_ratio=0.5, learning_rate=1e-4,pretrained_weights=None,pretrained_model=None,**kwargs):
        super().__init__()
        self.writer = SummaryWriter()
        self.train_step_losses = []
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_hyperparameters()
        self.run_dir = None  
        self.base_dir = "marine_debris_results"
        self.pretrained_model=pretrained_model


        self.src_channels = src_channels  
        if self.pretrained_model is not None:
            self.pretrained_in_channels = 3
        else:
            self.pretrained_in_channels = src_channels
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.means,self.stds,self.pos_weight = get_marida_means_and_stds()


        # Create Vision Transformer backbone
        vit = timm.create_model('vit_base_patch32_224', in_chans=self.src_channels, img_size=256, patch_size=16)
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.y_predicted =[]
        self.y_true = []
        self.test_image_count = 0

        # Define the MAE decoder
        self.decoder = MAEDecoderTIMM(
            in_chans=self.src_channels,
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=512,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.input_adapter = InputChannelAdapter(self.src_channels, self.pretrained_in_channels)
        
        # Determine embedding_channels for UNet_Marida
        embedding_dim_for_unet = self._get_embedding_dim()
        self.projection_head = UNet_Marida(input_channels=self.pretrained_in_channels, 
                                           embedding_channels=embedding_dim_for_unet, # Set embedding_channels here
                                           out_channels=src_channels)
        
        self.fully_finetuning = True
        if self.fully_finetuning:
            for param in self.parameters():
                param.requires_grad = True
            self.projection_head.requires_grad_(True)
            self.input_adapter.requires_grad_(True)
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
        print("weights",self.weight)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=self.weight)

        self.total_train_loss = 0.0
        self.total_test_loss = 0.0
        self.train_batch_count = 0
        self.test_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def _get_embedding_dim(self):
        if self.pretrained_model:
            if isinstance(self.pretrained_model, MAE):
                return self.pretrained_model.encoder.embed_dim
            elif isinstance(self.pretrained_model, MoCoGeo):
                return self.pretrained_model.encoder_q.fc.out_features
            elif isinstance(self.pretrained_model, MoCo):
                return self.pretrained_model.projection_head.output_dim
        # Default embedding dimension if no pretrained model or type not recognized
        return 128 # A common default for many image embeddings

    def forward(self, images, original_embedding_input):
        print("\n--- MAEFineTuning Forward Pass ---")
        print(f"Initial input images shape: {images.shape}")
        print(f"Initial input original_embedding_input shape: {original_embedding_input.shape}")

        # Adapt input images channels for UNet's encoder path
        images_for_unet = self.input_adapter(images) 
        print(f"images_for_unet (after input_adapter) shape: {images_for_unet.shape}") 

        processed_embedding = original_embedding_input

        if self.fully_finetuning and self.pretrained_model is not None:
            print("\nProcessing embedding with pretrained model:")
            # Attempt to squeeze extra singleton dimensions if they exist
            print(f"processed_embedding before initial squeeze: {processed_embedding.shape}")
            while processed_embedding.dim() > 2 and processed_embedding.shape[1] == 1:
                processed_embedding = processed_embedding.squeeze(1)
            print(f"processed_embedding after initial squeeze: {processed_embedding.shape}")
            
            # Pass through the pretrained model
            if isinstance(self.pretrained_model, MAE):
                print("Using MAE's forward_encoder.")
                processed_embedding = self.pretrained_model.forward_encoder(processed_embedding)
            elif isinstance(self.pretrained_model, MoCoGeo):
                print("Using MoCoGeo's forward.")
                processed_embedding = self.pretrained_model(processed_embedding)
            elif isinstance(self.pretrained_model, MoCo):
                print("Using MoCo's backbone and projection_head.")
                processed_embedding = self.pretrained_model.backbone(processed_embedding).flatten(start_dim=1)
                processed_embedding = self.pretrained_model.projection_head(processed_embedding)
            else:
                print(f"Warning: Unsupported pretrained model type: {type(self.pretrained_model)}. Embedding may not be processed correctly.")
            
            print(f"processed_embedding after pretrained model pass: {processed_embedding.shape}")

            # Ensure processed_embedding is a 2D feature vector (B, D) before spatial expansion
            if processed_embedding.dim() > 2:
                print(f"Flattening processed_embedding from {processed_embedding.shape} to (B, D).")
                processed_embedding = processed_embedding.flatten(start_dim=1)
            
            print(f"processed_embedding (after potential flatten to B, D): {processed_embedding.shape}")

            # Spatially expand the (B, D) feature vector to (B, D, H_target, W_target)
            target_h = images_for_unet.shape[2] // (2**4) # Assuming 4 downsampling steps in UNet
            target_w = images_for_unet.shape[3] // (2**4)
            
            print(f"Expanding processed_embedding {processed_embedding.shape} to target spatial dimensions ({target_h}, {target_w}).")
            processed_embedding = processed_embedding.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, target_h, target_w
            )
            print(f"processed_embedding (after spatial expand for UNet): {processed_embedding.shape}")
        else:
            print("\nNo pretrained model or not fully finetuning. Preparing embedding directly for UNet.")
            # If not using a pretrained model, ensure `processed_embedding` has the correct (B, C, H, W) shape for UNet.
            print(f"processed_embedding before processing for UNet: {processed_embedding.shape}")

            if processed_embedding.dim() == 5:
                # If it's (B, 1, C, H, W), remove the singleton dim
                processed_embedding = processed_embedding.squeeze(1)
                print(f"processed_embedding after squeezing 5D to 4D: {processed_embedding.shape}")
            
            # Now, it should be (B, C, H, W) or (B, D).
            # If it's (B, C, H, W), resize it to UNet bottleneck dimensions.
            if processed_embedding.dim() == 4:
                target_h = images_for_unet.shape[2] // (2**4) 
                target_w = images_for_unet.shape[3] // (2**4)
                print(f"Resizing 4D processed_embedding {processed_embedding.shape} to target spatial dimensions ({target_h}, {target_w}).")
                processed_embedding = F.interpolate(processed_embedding, size=(target_h, target_w), mode='bilinear', align_corners=False)
            elif processed_embedding.dim() == 2:
                # If it's (B, D), treat D as channel and expand spatially
                embedding_dim_for_unet = processed_embedding.shape[1]
                target_h = images_for_unet.shape[2] // (2**4) 
                target_w = images_for_unet.shape[3] // (2**4)
                print(f"Expanding 2D processed_embedding {processed_embedding.shape} to target spatial dimensions ({target_h}, {target_w}).")
                processed_embedding = processed_embedding.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, target_h, target_w
                )
            else:
                # Fallback: create a zero tensor if shape is unexpected
                embedding_dim_for_unet = self._get_embedding_dim() if self._get_embedding_dim() > 0 else 128
                target_h = images_for_unet.shape[2] // (2**4) 
                target_w = images_for_unet.shape[3] // (2**4)
                print(f"Warning: Unexpected embedding shape {processed_embedding.shape}. Creating zero tensor for embedding.")
                processed_embedding = torch.zeros(
                    images.shape[0], embedding_dim_for_unet, target_h, target_w,
                    device=images.device
                )
            print(f"processed_embedding (final shape before UNet call, without pretrained model processing): {processed_embedding.shape}")


        print("\n--- Final shapes before UNet_Marida call ---")
        print(f"images_for_unet shape: {images_for_unet.shape}")
        print(f"processed_embedding shape: {processed_embedding.shape}")

        return self.projection_head(processed_embedding, images_for_unet)
    
    def training_step(self, batch, batch_idx):
        train_dir = "train_results"
        data, target,embedding = batch
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

        # In on_test_epoch_end:
        print(f"Number of predictions: {len(all_predictions)}, Number of targets: {len(all_targets)}")
        print(f"Unique predictions: {np.unique(all_predictions)}, Unique targets: {np.unique(all_targets)}")
        print(f"Number of masked values: {np.sum(all_targets == -1)}")


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