import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
import torchvision 
import numpy as np

from config import get_means_and_stds
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform

class MAE(pl.LightningModule):
    def __init__(self, src_channels=3, mask_ratio=0.90, decoder_dim=512, pretrained_weights=None):
        super().__init__()
        print(f"Initializing MAE with {src_channels} source channels")
        self.src_channels = src_channels
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio

        # Create the Vision Transformer backbone
        vit = timm.create_model(
            'vit_base_patch32_224',
            in_chans=self.src_channels,
            img_size=256,
            patch_size=16,
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

        # Define the MAE decoder
        self.decoder = MAEDecoderTIMM(
            in_chans=self.src_channels,
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        # Adapter layer to project 3 channels to 12 channels (only for fine-tuning on RGB images)
        self.adapter_layer = nn.Conv2d(3, 12, kernel_size=1) if self.src_channels == 3 else None

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

        # Mean Squared Error loss for reconstruction
        self.criterion = nn.MSELoss()

        # Initialize variables to accumulate the loss
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def load_pretrained_weights(self, pretrained_weights):
        """Load pretrained weights into the model."""
        checkpoint = torch.load(pretrained_weights)
        
        # Load only the encoder and decoder weights, skipping classifier if any.
        model_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        self.load_state_dict(model_state_dict)
        print("Pretrained weights loaded successfully.")

    def forward_encoder(self, images, idx_keep=None):
        # Apply adapter layer only if using 3 channels (RGB)
        #if self.adapter_layer:
        #    images = self.adapter_layer(images)
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # Build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)

        # Masked input
        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # Decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # Predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)

        x_pred = self.decoder.predict(x_pred)
        
        
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]

        # Random token masking
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Forward pass through encoder
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)

        # Forward pass through decoder
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # Get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)
        
        pred_patches = utils.set_at_index(patches, idx_mask - 1, x_pred)
        pred_img = utils.unpatchify(pred_patches, self.patch_size, self.src_channels)

        masked_patches = utils.set_at_index(patches, idx_mask - 1, torch.zeros_like(x_pred))
        masked_img = utils.unpatchify(masked_patches, self.patch_size, self.src_channels)
        
        return x_pred, target, pred_img, masked_img
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        images= batch #, min_value, max_value = batch  

        batch_size = images.shape[0]

        # Handle image shape (ensure it aligns with your model input requirements)
        images = images.squeeze(1)  # Only if the second dimension is guaranteed to be 1
        
        # Forward pass
        x_pred, target, pred_img, masked_img = self(images)

        # Compute reconstruction loss (e.g., MSE Loss)
        loss = self.criterion(x_pred, target)

        # Log loss at the step level
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Optionally log images every 100 steps
        if batch_idx % 100 == 0:
            self.log_images(
                images[10], 
                pred_img[10], 
                masked_img[10]
            )

        # Optionally accumulate loss for manual tracking (can use self.log for epochs instead)
        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0  # Initialize if not already defined
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1

        # Debug: Check for NaN or Inf gradients (if debugging gradient issues)
        for name, param in self.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"WARNING: NaN or Inf gradients detected in {name}")

        return loss

    def on_train_epoch_end(self):
        # Log the average training loss for the epoch
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss', avg_train_loss)

        # Reset accumulation variables for next epoch
        self.total_train_loss = 0.0
        self.train_batch_count = 0

        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

    def validation_step(self, batch, batch_idx):
        images = batch
        # Forward pass
        x_pred, target,pred_img,masked_img = self(images)

        # Compute reconstruction loss
        val_loss = self.criterion(x_pred, target)

        # Accumulate validation loss
        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1

        return val_loss

    def on_validation_epoch_end(self):
        # Log the average validation loss for the epoch
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss', avg_val_loss, on_epoch=True)

        # Reset accumulation variables for next epoch
        self.total_val_loss = 0.0
        self.val_batch_count = 0

        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")


    def log_images(self, original_images, reconstructed_images, masked_images):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        means, stds = get_means_and_stds()
        means = means.to(device)
        stds = stds.to(device)
        
        # Denormalize the images
        original_images = (original_images  * stds[1:4,None,None]) + means[1:4,None,None]  # #(max_value - min_value + 1e-7) + min_value
        reconstructed_images = (reconstructed_images  * stds[1:4,None,None]) + means[1:4,None,None] # #(max_value - min_value + 1e-7) + min_value
        masked_images = (masked_images  * stds[1:4,None,None]) + means[1:4,None,None] #(max_value - min_value + 1e-7) + min_value

        # Select RGB channels and permute dimensions
        original_images = original_images[ [2, 1, 0], :, :].permute(1, 2, 0).detach().cpu().numpy()
        reconstructed_images = reconstructed_images[ [2, 1, 0], :, :].permute(1, 2, 0).detach().cpu().numpy()
        masked_images = masked_images[ [2, 1, 0], :, :].permute(1, 2, 0).detach().cpu().numpy()

        # Clip values and scale to [0, 255]
        def prepare_image(image):
            image = np.clip(image, 0, np.percentile(image, 99))
            image = (image / image.max() * 255).astype('uint8')
            return image

        original_images = prepare_image(original_images)
        reconstructed_images = prepare_image(reconstructed_images)
        masked_images = prepare_image(masked_images)

        # Log the images to TensorBoard
        self.logger.experiment.add_image('Original Images', original_images, self.current_epoch, dataformats='HWC')
        self.logger.experiment.add_image('Reconstructed Images', reconstructed_images, self.current_epoch, dataformats='HWC')
        self.logger.experiment.add_image('Masked Images', masked_images, self.current_epoch, dataformats='HWC')

    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-4)

        # Use a learning rate scheduler that adapts over epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step the scheduler every epoch
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        # Log the current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch - Current learning rate: {current_lr}")


    def get_embeddings_from_image(self,image):
        #Bild laden
        #
        self.forward_encoder(image)