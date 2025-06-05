import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
import numpy as np

from config import get_means_and_stds
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

class MAE(pl.LightningModule):
    def __init__(self, src_channels=3, mask_ratio=0.90, decoder_dim=512, pretrained_weights=None):
        super().__init__()
        self.src_channels = src_channels
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio

        vit = timm.create_model(
            'vit_base_patch32_224',
            in_chans=self.src_channels,
            img_size=224,
            patch_size=32, # Ensure this matches the timm model name's patch size
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

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

        self.adapter_layer = nn.Conv2d(3, 12, kernel_size=1) if self.src_channels == 3 else None

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

        self.criterion = nn.MSELoss()

        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def load_pretrained_weights(self, pretrained_weights):
        """Load pretrained weights into the model."""
        checkpoint = torch.load(pretrained_weights)
        model_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        self.load_state_dict(model_state_dict)
        print("Pretrained weights loaded successfully.")

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)

        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        x_decoded = self.decoder.decode(x_masked)

        x_pred = utils.get_at_index(x_decoded, idx_mask)

        x_pred = self.decoder.predict(x_pred)
        
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]

        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)

        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)
        
        pred_patches = utils.set_at_index(patches, idx_mask - 1, x_pred)
        pred_img = utils.unpatchify(pred_patches, self.patch_size, self.src_channels)

        masked_patches = utils.set_at_index(patches, idx_mask - 1, torch.zeros_like(x_pred))
        masked_img = utils.unpatchify(masked_patches, self.patch_size, self.src_channels)
        
        return x_pred, target, pred_img, masked_img
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple) and len(batch) == 3: 
            images = batch[0]
        else:
            images = batch 

        x_pred, target, pred_img, masked_img = self(images)

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

        for name, param in self.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"WARNING: NaN or Inf gradients detected in {name}")

        return loss

    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss', avg_train_loss)

        self.total_train_loss = 0.0
        self.train_batch_count = 0

        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple) and len(batch) == 3:
            images = batch[0]
        else:
            images = batch

        x_pred, target,pred_img,masked_img = self(images)

        val_loss = self.criterion(x_pred, target)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1

        return val_loss

    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss', avg_val_loss, on_epoch=True)

        self.total_val_loss = 0.0
        self.val_batch_count = 0

        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

    def log_images(self, original_image, reconstructed_image, masked_image):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        means, stds = get_means_and_stds()
        
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
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch - Current learning rate: {current_lr}")

    def get_embeddings_from_image(self,image):
        return self.forward_encoder(image)