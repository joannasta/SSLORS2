import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
import numpy as np

from config import _Hydro
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM

class MAE_Ocean(pl.LightningModule):
    """MAE with ViT backbone and fuses image CLS embedding with projected ocean features."""
    def __init__(self, src_channels=3, mask_ratio=0.90, decoder_dim=512, pretrained_weights=None, 
                 num_ocean_features: int = 3):
        super().__init__()
        self.src_channels = src_channels
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio
        self.pretrained_weights = pretrained_weights
        self.num_ocean_features = num_ocean_features

        # ViT backbone (base, 224, patch16) with configurable input channels
        vit = timm.create_model(
            'vit_base_patch16_224',
            in_chans=self.src_channels,
            img_size=224,          
            patch_size=16,        
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.vit_backbone = vit 
        self.num_image_patches = vit.patch_embed.num_patches
        self.encoder_sequence_length = self.num_image_patches + 1 # +1 for CLS token

        # MAE decoder to reconstruct masked patches
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

        # Project ocean feature vector to ViT embed_dim
        self.linear_projection_ocean_features = nn.Linear(
            in_features=self.num_ocean_features,
            out_features=vit.embed_dim
        )
        
        # Optional adapter to adapt source channels
        self.adapter_layer = nn.Conv2d(3, 12, kernel_size=1) if self.src_channels == 3 else None

        if self.pretrained_weights:
            self.load_pretrained_weights(self.pretrained_weights)

        self.criterion = nn.MSELoss()

        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def load_pretrained_weights(self, pretrained_weights):
         """Load pretrained weights."""
        checkpoint = torch.load(pretrained_weights)
        model_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        self.load_state_dict(model_state_dict)

    def forward_encoder(self, images, idx_keep=None):
        """Manual ViT forward on tokens, supports masking via idx_keep."""
        x_patches = self.vit_backbone.patch_embed(images) # (B, N, C)
        cls_token = self.vit_backbone.cls_token.expand(x_patches.shape[0], -1, -1)
        x = torch.cat((cls_token, x_patches), dim=1) # (B, 1+N, C)
        x = x + self.vit_backbone.pos_embed
        if idx_keep is not None:
            x = utils.get_at_index(x, idx_keep) # keep selected tokens
        for blk in self.vit_backbone.blocks:
            x = blk(x)
        x = self.vit_backbone.norm(x)
        return x # (B, kept_tokens, C)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        """Decode and predict masked tokens."""
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)

        # insert decoded visible tokens, fill masked with mask_token
        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.encoder_sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # Decode and predict masked tokens only
        x_decoded = self.decoder.decode(x_masked)
        
        x_pred_full_sequence = utils.get_at_index(x_decoded, idx_mask)
        x_pred = x_pred_full_sequence
        x_pred = self.decoder.predict(x_pred)
        
        return x_pred  # (B, num_masked, patch_dim)

    def forward(self, images, ocean_features: torch.Tensor):
        """MAE forward + feature fusion: returns reconstructions and fused embedding."""
        batch_size = images.shape[0]

        # Random mask on patch tokens
        idx_keep_patches_only, idx_mask_patches_only = utils.random_token_mask(
            size=(batch_size, self.num_image_patches),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        idx_keep_cls = torch.zeros((batch_size, 1), dtype=torch.long, device=images.device)
        idx_keep_absolute = torch.cat(
            (idx_keep_cls, idx_keep_patches_only + 1),
            dim=1
        )
        
        idx_mask_absolute = idx_mask_patches_only + 1

        # Encode visible tokens and decode masked
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep_absolute)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep_absolute, idx_mask=idx_mask_absolute
        )

        # Targets: true masked patches
        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask_patches_only)

        # Reconstruct full image from predicted masked patches
        pred_patches_unmasked_idx = idx_mask_absolute - 1
        pred_patches_full = utils.set_at_index(patches, pred_patches_unmasked_idx, x_pred)
        pred_img = utils.unpatchify(pred_patches_full, self.patch_size, self.src_channels)

        masked_patches_full = utils.set_at_index(patches, pred_patches_unmasked_idx, torch.zeros_like(x_pred))
        masked_img = utils.unpatchify(masked_patches_full, self.patch_size, self.src_channels)
        
        cls_embedding = x_encoded[:, 0, :]

        projected_ocean_features = self.linear_projection_ocean_features(ocean_features)

        combined_embedding = torch.cat((cls_embedding, projected_ocean_features), dim=1)

        return x_pred, target, pred_img, masked_img, combined_embedding
    
    def training_step(self, batch, batch_idx):
        '''Training Step'''
        images, ocean_features = batch
        x_pred, target, pred_img, masked_img, combined_embedding = self(images, ocean_features)
        loss = self.criterion(x_pred, target)

        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if batch_idx % 100 == 0:
            self.log_images(
                images[0], 
                pred_img[0], 
                masked_img[0]
            )

        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1
        
        return loss

    def on_train_epoch_end(self):
        """Log average training loss at epoch end."""
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss', avg_train_loss)

        self.total_train_loss = 0.0
        self.train_batch_count = 0

    def validation_step(self, batch, batch_idx):
        """Validation step """
        images, ocean_features = batch
        
        x_pred, target, pred_img, masked_img, combined_embedding = self(images, ocean_features)

        val_loss = self.criterion(x_pred, target)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1

        return val_loss

    def on_validation_epoch_end(self):
        """Log average validation loss at epoch end."""
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss', avg_val_loss, on_epoch=True)

        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def log_images(self, original_image, reconstructed_image, masked_image):
        """Log original, reconstructed, and masked images to TensorBoard in RGB."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        means, stds = get_Hydro_means_and_stds()
        
        means_rgb = means[[2,1,0]].to(device).unsqueeze(-1).unsqueeze(-1)
        stds_rgb = stds[[2,1,0]].to(device).unsqueeze(-1).unsqueeze(-1)

        original_image = (original_image[[2,1,0],:,:] * stds_rgb) + means_rgb
        reconstructed_image = (reconstructed_image[[2,1,0],:,:] * stds_rgb) + means_rgb
        masked_image = (masked_image[[2,1,0],:,:] * stds_rgb) + means_rgb

        original_image = original_image.permute(1, 2, 0).detach().cpu().numpy()
        reconstructed_image = reconstructed_image.permute(1, 2, 0).detach().cpu().numpy()
        masked_image = masked_image.permute(1, 2, 0).detach().cpu().numpy()

        def prepare_image(image):
            image = np.clip(image, 0, np.percentile(image, 99))
            image = (image / image.max())
            image = (image * 255).astype('uint8')
            return image

        original_image = prepare_image(original_image)
        reconstructed_image = prepare_image(reconstructed_image)
        masked_image = prepare_image(masked_image)

        self.logger.experiment.add_image('Original Images', original_image, self.current_epoch, dataformats='HWC')
        self.logger.experiment.add_image('Reconstructed Images', reconstructed_image, self.current_epoch, dataformats='HWC')
        self.logger.experiment.add_image('Masked Images', masked_image, self.current_epoch, dataformats='HWC')

    def configure_optimizers(self):
        """AdamW + MultiStepLR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        """Log current learning rate at epoch start."""
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)

    def get_embeddings_from_image(self,image, ocean_features: torch.Tensor):
        """Encode image to CLS embedding and concatenate with projected ocean features."""
        image_embedding = self.forward_encoder(image)[:, 0, :]
        projected_ocean_features = self.linear_projection_ocean_features(ocean_features)
        
        return torch.cat((image_embedding, projected_ocean_features), dim=1)