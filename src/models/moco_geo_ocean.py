import copy
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError # Changed from MulticlassAccuracy for regression
from typing import Callable, Tuple
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class OceanFeatureRegressor(nn.Module):
    """
    A regression head for predicting oceanographic features.
    """
    def __init__(self, input_dim: int = 128, output_features_dim: int = 3): # 3 for bathy, chlorophyll, secchi
        super().__init__()
        self.layers = nn.Sequential(
            # ReLU is generally not used in the final layer of a regression model
            # if the target values can span the full real number range.
            nn.Linear(input_dim, output_features_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input features if they come from a convolutional backbone
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class MoCoOceanFeatures(pl.LightningModule): # Renamed class for clarity
    """
    MoCo-OceanFeatures: A PyTorch Lightning module for MoCo v2 style self-supervised
    learning with an auxiliary oceanographic feature regression task.
    """
    def __init__(
        self,
        base_encoder: Callable = torchvision.models.resnet18,
        dim: int = 128,      # Output dimension of the projection head (for contrastive learning)
        K: int = 4096,       # Memory bank size for NTXentLoss
        m: float = 0.99,     # Momentum coefficient for momentum encoder update
        T: float = 0.07,     # Temperature for NTXentLoss
        src_channels: int = 3, # Number of input channels for the images
        output_features_dim: int = 3 # Number of oceanographic features to predict (bathy, chlorophyll, secchi)
    ):
        super().__init__()
        # Save all __init__ arguments as hyperparameters for logging and reproducibility
        self.save_hyperparameters()

        # Initialize the base encoder (e.g., ResNet) and modify its first convolution
        resnet = base_encoder(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            self.hparams.src_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove the final classification layer from the backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # The projection head remains for the contrastive learning task
        self.projection_head = MoCoProjectionHead(512, 512, self.hparams.dim)

        # Create momentum encoder and projection head, and deactivate their gradients
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Loss functions
        self.contrastive_loss = NTXentLoss(
            memory_bank_size=(self.hparams.K, self.hparams.dim),
            temperature=self.hparams.T
        )
        self.regression_criterion = nn.MSELoss() # Changed to MSELoss for regression

        # Auxiliary ocean feature regressor and its metric
        self.ocean_regressor = OceanFeatureRegressor( # Renamed
            input_dim=self.hparams.dim, output_features_dim=self.hparams.output_features_dim
        )
        self.regression_metric = MeanSquaredError() # Changed to MeanSquaredError for regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the online (query) encoder.
        """
        features = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(features)
        return query

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the momentum (key) encoder. Gradients are detached.
        """
        with torch.no_grad(): # Ensure no gradients are computed for momentum encoder
            features = self.backbone_momentum(x).flatten(start_dim=1)
            key = self.projection_head_momentum(features)
        return key

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, including MoCo update and ocean feature regression.
        """
        # Update momentum encoder's weights based on cosine schedule
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        # Unpack the batch: (query_image, key_image, ocean_features_labels)
        # Note: For self-supervised learning, typically two augmented views of the same image are used.
        # If your DataLoader yields (image_view1, image_view2, features_for_image), this unpacking is correct.
        x_query, x_key, ocean_features_labels = batch 

        # Get features from online and momentum encoders
        query_features = self.forward(x_query)
        key_features = self.forward_momentum(x_key)

        # Calculate contrastive loss
        loss_contrastive = self.contrastive_loss(query_features, key_features)

        # Perform ocean feature regression and calculate its loss
        predicted_ocean_features = self.ocean_regressor(query_features)
        loss_regression = self.regression_criterion(predicted_ocean_features, ocean_features_labels)

        # Combine losses
        total_loss = loss_contrastive + loss_regression

        # Log training metrics
        self.log("train_loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss", loss_regression, on_step=False, on_epoch=True, logger=True)
        
        # Log training regression metric (e.g., MSE)
        train_regression_metric = self.regression_metric(predicted_ocean_features, ocean_features_labels)
        self.log("train_regression_metric", train_regression_metric, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Performs a single validation step to evaluate performance on unseen data.
        """
        # Unpack the batch (for validation, typically only one view and the label is needed)
        # However, to match training_step's batch structure for a shared DataLoader, we keep x_key here.
        x_query, _, ocean_features_labels = batch 
        
        # Get query features (no momentum update in validation)
        query_features = self.forward(x_query)

        # Perform ocean feature regression and calculate its loss
        predicted_ocean_features = self.ocean_regressor(query_features)
        loss_regression = self.regression_criterion(predicted_ocean_features, ocean_features_labels)

        # Calculate validation metric
        val_regression_metric = self.regression_metric(predicted_ocean_features, ocean_features_labels)

        # Log validation metrics
        self.log("val_loss", loss_regression, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_regression_metric", val_regression_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss_regression, "val_regression_metric": val_regression_metric}

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
        return optimizer