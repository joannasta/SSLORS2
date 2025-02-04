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
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG

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
    def __init__(self, src_channels=3, mask_ratio=0.5, pretrained_weights=None):
        super().__init__()
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
        self.projection_head = UNet_Marida(in_channels=3, out_channels=1)
        self.criterion = nn.CrossEntropyLoss()

        
        # Initialize variables to accumulate the loss
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, strict=False, **kwargs):
        # Initialize the model
        model = cls(**kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        
        # Adjust patch embedding weights for BGR channels
        if 'backbone.vit.patch_embed.proj.weight' in state_dict:
            original_weights = state_dict['backbone.vit.patch_embed.proj.weight']
            state_dict['backbone.vit.patch_embed.proj.weight'] = original_weights
            print("Adjusted patch embedding weights for BGR channels.")
        
        # Remove decoder weights from state_dict
        keys_to_remove = [k for k in state_dict.keys() if k.startswith('decoder')]
        for key in keys_to_remove:
            del state_dict[key]
        print("Removed decoder weights from checkpoint.")
        
        # Load state_dict into model
        model.load_state_dict(state_dict, strict=strict)
        print("Model loaded successfully with adjusted weights.")
        return model
    
    def forward_encoder(self, images, idx_keep=None):
        if self.adapter_layer:
            images = images.float() # (7,3,256,256)
            #images = self.adapter_layer(images) 
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

    def forward(self, batch):
        images = batch
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(size=(batch_size, self.sequence_length), mask_ratio=self.mask_ratio, device=images.device)
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        return self.projection_head(x_pred, images)
    
    
    #TODO Extracting maybe 3 channels first and if not change pretraining
    #TODO EXTRACTING BEFORE OR AFTER ENCODER TRY BOTH
    #TODO COMPARE WIHT MAGICBATHYNET
    #TODO ENCODER / DECODER / PROJECTION HEAD
    #TODO OVERFITTING ONE BATCH TO SEE IF THAT WORKS
    #TODO NO SHUFFLING IN DATALOADER
    #TODO LOKAL TESTING
    def training_step(self, batch, batch_idx):
        data, target = batch
        batch_size = data.shape[0]
        prediction = self(data)
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
        data, target = batch
        batch_size = data.shape[0]
        

        prediction = self(data)
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
        data, target = batch
        batch_size = data.shape[0]
        prediction = self(x)
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
        orig = original_images.cpu().numpy()
        rgb = np.transpose(orig, (1, 2, 0)) 
        rgb = rgb[:, :, [2, 1, 0]] 
        
        target = target.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
  
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

