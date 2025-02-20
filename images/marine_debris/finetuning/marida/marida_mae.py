# Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
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

# Project-specific imports
from .marida_unet  import UNet_Marida
from src.utils.finetuning_utils import calculate_metrics
from config import get_marida_means_and_stds

# TODOs and planned experiments
# - Perform ablation studies to test input combinations.
# - Add auxiliary losses for decoded data.
# - Implement attention mechanisms.
# - Experiment with freezing and unfreezing encoder/decoder.
# - Preprocess and verify decoded data quality.
# - Experiment with learnable contributions for input balance.


def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)

class MAEFineTuning(pl.LightningModule):
    def __init__(self, src_channels=11, mask_ratio=0.5, learning_rate=1e-4,pretrained_weights=None):
        super().__init__()
        self.writer = SummaryWriter()
        self.train_step_losses = []
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_hyperparameters()
        self.run_dir = None  
        self.base_dir = "bathymetry_results"

        self.src_channels = 11
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.means,self.stds,self.pos_weight = get_marida_means_and_stds()


        # Create Vision Transformer backbone
        vit = timm.create_model('vit_base_patch32_224', in_chans=self.src_channels, img_size=256, patch_size=16)
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

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

        if pretrained_weights:
            self.mask_ratio = 0
            self.load_pretrained_weights(pretrained_weights)

        self.adapter_layer = nn.Conv2d(3, 12, kernel_size=1)
        self.projection_head = UNet_Marida(input_channels=11, out_channels=1)
        self. criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean')

        
        # Initialize variables to accumulate the loss
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def forward(self, images,embedding):
        #batch_size = images.shape[0]
        #idx_keep, idx_mask = utils.random_token_mask(size=(batch_size, self.sequence_length), mask_ratio=self.mask_ratio, device=images.device)
        return self.projection_head(embedding,images)
    
    
    #TODO Extracting maybe 3 channels first and if not change pretraining
    #TODO EXTRACTING BEFORE OR AFTER ENCODER TRY BOTH
    #TODO COMPARE WIHT MAGICBATHYNET
    #TODO ENCODER / DECODER / PROJECTION HEAD
    #TODO OVERFITTING ONE BATCH TO SEE IF THAT WORKS
    #TODO NO SHUFFLING IN DATALOADER
    #TODO LOKAL TESTING
    def training_step(self, batch, batch_idx):

        data, target,embedding = batch
        batch_size = data.shape[0]
        prediction = self(data,embedding)
        print("target",target.shape)
        print("target min, max",target.min(),target.max())
        print("target mean",target.mean())
        print("target unique",np.unique(target))
        print("target",target[0])
        print("prediction",prediction.shape)
        print("prediction",prediction[0])
        loss = self.criterion(prediction, target)
        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),
                prediction[0],
                target[0].cpu(),
            )

        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1

        return loss 


    def validation_step(self, batch, batch_idx):
        data, target,embedding = batch
        batch_size = data.shape[0]
        prediction = self(data,embedding)
        val_loss = self.criterion(prediction, target)
        if batch_idx % 100 == 0:
            self.log_images(
                data[0].cpu(),  
                prediction[0],  
                target[0].cpu(),  
            )

        self.log('val_loss', val_loss)
        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1

        return val_loss
    
    def test_step(self, batch, batch_idx):
        data, target,embedding = batch
        batch_size = data.shape[0]
        prediction = self(data,embedding)
        loss = self.criterion(prediction, target)
        self.log("test_loss", loss)

    def on_train_start(self):
        self.log_results()

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
    
    def log_images(self, original_images: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> None:
        img= original_images.cpu().numpy()
        target = target.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()

        img = img.transpose(2,0,1)
        img = (img - self.means[:, None, None]) / self.stds[:, None, None]
        img = img[1:4, :, :]  
        img = img[[2, 1, 0], :, :]  # Swap BGR to RGB
        if img.shape[0] == 3:  
            img = np.transpose(img, (1, 2, 0))

        prediction = prediction.squeeze(0)
        target = target.squeeze(0)

        img = np.clip(img, 0, np.percentile(img, 99))
        img = (img / img.max() * 255).astype('uint8')
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(target)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        # Plot prediction (as class labels)
        axes[2].imshow(prediction)  # Use a colormap for labels
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        plt.savefig(f'epoch-{self.current_epoch}-image_sample.png', bbox_inches='tight')
        plt.close()
  
    def log_results(self):
        if self.run_dir is None:  
            run_index = 0
            while os.path.exists(os.path.join(self.base_dir, f"run_{run_index}")):
                run_index += 1
            
            self.run_dir = os.path.join(self.base_dir, f"run_{run_index}")
            os.makedirs(self.run_dir, exist_ok=True)
                
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
        return [optimizer], [scheduler]

