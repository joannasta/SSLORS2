import torch
import torch.nn as nn
import torchvision
import timm
import wandb
from functools import partial
from typing import Optional, Tuple
from transformers import get_cosine_schedule_with_warmup
import lightning as L
from torch import optim, nn, utils, Tensor
from timm.models.vision_transformer import PatchEmbed, Block
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import numpy as np
import torchvision
from lightning import LightningModule
from models.mae.mae_vit import MAE_VIT
# typing
from typing import Optional, Tuple
from torchinfo import summary

class MAE_LIT(LightningModule):
    def __init__(
        self,
        src_channels = 3,
        base_lr: float = 3e-5,
        num_gpus: int = 1,
        batch_size: int = 512,
        warmup_epochs: int = 1,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95)
    ):
        super().__init__()

        self.src_channels= src_channels
        self.base_lr = base_lr
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.betas = betas
        self.model = MAE_VIT(src_channels=src_channels)
            
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor
            mask_ratio: Ratio of masking applied to the input

        Returns:
            Tuple containing the loss, predicted output, and mask
        """
        _,x_pred,target, pred_img, masked_img = self.model(x)
        return _,x_pred,target, pred_img, masked_img  

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step of the model.

        Args:
            batch: Tuple containing the input batch and labels
            batch_idx: Index of the current batch

        Returns:
            Loss value
        """
        # Do optimization with optimizer and scheduler
        opt = self.optimizers()
        opt.zero_grad()

        x, _ = batch
        loss, _, _,_ = self(x)

        self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()
        sch.step()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step of the model.

        Args:
            batch: Tuple containing the input batch and labels
            batch_idx: Index of the current batch
        """
        x, _ = batch
        loss, x_pred,target, pred_img, masked_img = self(x)
        
        # Create grid using torchvision of images, predicted and mask
        if batch_idx == 0:
            full_grid = self.generate_grid_of_samples(x, x_pred, masked_img)
            self.logger.experiment.log({"sample_images": [wandb.Image(full_grid, caption="Predicted Images")]})
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        self.lr = self.base_lr * self.num_gpus * self.batch_size / 256
        epoch_steps = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        num_warmup_steps = int(epoch_steps * self.warmup_epochs)
        
        # Define the optimizer and scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)
        
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=(self.trainer.estimated_stepping_batches),
                num_warmup_steps=num_warmup_steps,
            )
        
        return [optimizer], [scheduler]

    def generate_grid_of_samples(self, true_images: torch.Tensor, pred_patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate a grid of samples for visualization.

        Args:
            true_images: Tensor containing the true images
            pred_patch: Tensor containing the predicted patches
            mask: Tensor containing the mask

        Returns:
            Tensor representing the grid of samples
        """
        x_patch = self.model.patchify(true_images)
        # Merge original and predicted patches
        pred_patch = pred_patch * mask[:, :, None]
        x_patch = x_patch * (~mask[:, :, None].bool()).float()
        res = self.model.unpatchify(x_patch) + self.model.unpatchify(pred_patch)
        # Select three random samples from the dataset
        sample_indices = torch.randperm(len(true_images))[:3]
        
        # Get the RGB bands for each sample
        pred_images = res[sample_indices][:, [3, 2, 1]]
        true_image_grid = true_images[sample_indices][:, [3, 2, 1]]

        grid_images = torch.cat([true_image_grid, pred_images], dim=0)

        # Create a grid for displaying the images
        full_grid = torchvision.utils.make_grid(grid_images, nrow=3)
        # Convert the grid to a numpy array and transpose the dimensions to (height, width, channels)
        full_grid = full_grid.permute(1, 2, 0).float().cpu().numpy()
        
        # Log the image to Weights and Biases
        return full_grid



if __name__ == '__main__':
    import torch
    import torchsummary
    model = MAE_LIT(src_channels=202)
    #print(model)

    #torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')
    summary(model, input_size=(1,202, 128, 128))

    in_tensor = torch.randn(1, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)
