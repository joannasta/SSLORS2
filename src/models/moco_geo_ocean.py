import copy
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from typing import Callable, Optional, List
import random
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

# --- Mock config.py functions for demonstration ---
# In a real setup, you would ensure these are imported from your actual config.py
try:
    from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, get_means_and_stds, get_marida_means_and_stds
except ImportError:
    print("Warning: 'config.py' not found. Using mock normalization parameters and paths.")
    NORM_PARAM_DEPTH = None
    NORM_PARAM_PATHS = {"agia_napa": "mock_norm_params_path.npy"} # Path for mock means/stds

    def get_means_and_stds():
        # Mock means and stds for 12 bands (e.g., Sentinel-2, B01-B12 excluding B10)
        return np.array([0.1]*12), np.array([0.05]*12)

    def get_marida_means_and_stds():
        # Mock means and stds for 11 MARIDA bands + depth (not directly used here for img norm)
        return np.array([0.1]*11), np.array([0.05]*11), np.array([100.0])

# --- Original GeoClassifier (no changes needed) ---
class GeoClassifier(nn.Module):
    """
    A simple classifier head for geo-classification.
    It flattens input features if they are from a convolutional backbone.
    """
    def __init__(self, input_dim: int = 128, num_classes: int = 100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input features if they come from a convolutional backbone
        # This check is still relevant if features are from image backbone
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)

# --- New Tabular Feature Projector ---
class TabularFeatureProjector(nn.Module):
    """
    A small MLP to project tabular features into a compatible dimension
    for fusion with image embeddings.
    """
    def __init__(self, input_dim: int, output_dim: int = 64): # e.g., 5 input features -> 64 output dim
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float()) # Ensure input is float

# --- Modified MoCoGeo for Hybrid Data ---
class MoCoGeo(pl.LightningModule):
    """
    MoCo-Geo: A PyTorch Lightning module for MoCo v2 style self-supervised
    learning with an auxiliary geo-classification task, now integrating
    both image and tabular features.
    """
    def __init__(
        self,
        base_encoder: Callable = torchvision.models.resnet18, # Image backbone
        dim: int = 128,      # Output dimension of the image projection head
        K: int = 4096,       # Memory bank size for NTXentLoss
        m: float = 0.99,     # Momentum coefficient for momentum encoder update
        T: float = 0.07,     # Temperature for NTXentLoss
        src_channels: int = 3, # Number of input channels for the images
        num_geo_classes: int = 100, # Number of geographical classes
        input_dim_tabular: int = 5, # Dimension of tabular ocean features (lat, lon, bathy, chlorophyll, secchi)
        tabular_feature_projection_dim: int = 64 # Dimension for projected tabular features
    ):
        super().__init__()
        # Save all __init__ arguments as hyperparameters for logging and reproducibility
        self.save_hyperparameters()

        # --- Image Backbone ---
        resnet = base_encoder(pretrained=False)
        # Modify the first convolution to accept src_channels if needed
        # (original code had 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = nn.Conv2d(
            self.hparams.src_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove the final classification layer from the backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output is 512 for ResNet18
        self.projection_head = MoCoProjectionHead(512, 512, self.hparams.dim) # Projects image features

        # Create momentum encoder and projection head for images
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # --- Tabular Feature Projector ---
        self.tabular_feature_projector = TabularFeatureProjector(
            input_dim=self.hparams.input_dim_tabular,
            output_dim=self.hparams.tabular_feature_projection_dim
        )

        # Loss functions
        self.contrastive_loss = NTXentLoss(
            memory_bank_size=(self.hparams.K, self.hparams.dim), # Memory bank for image features
            temperature=self.hparams.T
        )
        self.geo_criterion = nn.CrossEntropyLoss()

        # Auxiliary geo-classifier and its metric
        # Input dimension for GeoClassifier is sum of image projection dim and tabular projection dim
        self.geo_classifier = GeoClassifier(
            input_dim=self.hparams.dim + self.hparams.tabular_feature_projection_dim,
            num_classes=self.hparams.num_geo_classes
        )
        # For MulticlassAccuracy, specify 'task' as 'multiclass' for recent torchmetrics versions
        self.geo_accuracy = MulticlassAccuracy(
            num_classes=self.hparams.num_geo_classes, average='micro', task='multiclass'
        )

    def forward(self, x_image: torch.Tensor, x_tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the online (query) encoder, processing both image and tabular data.
        Returns image features (for contrastive loss) and combined features (for geo-classifier).
        """
        # Process image data
        image_features = self.backbone(x_image).flatten(start_dim=1)
        query_image_embedding = self.projection_head(image_features)

        # Process tabular data
        projected_tabular_features = self.tabular_feature_projector(x_tabular)

        # Concatenate image embedding and projected tabular features for the classifier
        combined_features_for_classifier = torch.cat(
            (query_image_embedding, projected_tabular_features), dim=1
        )
        return query_image_embedding, combined_features_for_classifier

    def forward_momentum(self, x_image: torch.Tensor, x_tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the momentum (key) encoder. Gradients are detached.
        Returns image features (for contrastive loss) and combined features (for geo-classifier).
        """
        with torch.no_grad(): # Ensure no gradients are computed for momentum encoder
            # Process image data
            image_features = self.backbone_momentum(x_image).flatten(start_dim=1)
            key_image_embedding = self.projection_head_momentum(image_features)

            # Process tabular data
            projected_tabular_features = self.tabular_feature_projector(x_tabular) # Tabular projector does not have momentum

            # Concatenate image embedding and projected tabular features for the classifier
            combined_features_for_classifier = torch.cat(
                (key_image_embedding, projected_tabular_features), dim=1
            )
        return key_image_embedding, combined_features_for_classifier


    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, including MoCo update and geo-classification.
        """
        # Update momentum encoder's weights based on current epoch progress
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        # Unpack the batch: (query_image, key_image, tabular_features, geo_label)
        x_query_image, x_key_image, tabular_features, geo_labels = batch

        # Get features from online and momentum encoders
        # query_image_features are for contrastive loss, query_combined_features for classification
        query_image_features, query_combined_features = self.forward(x_query_image, tabular_features)
        key_image_features, _ = self.forward_momentum(x_key_image, tabular_features) # _ because we don't need combined features from momentum for contrastive or aux task

        # Calculate contrastive loss using only image features
        loss_contrastive = self.contrastive_loss(query_image_features, key_image_features)

        # Perform geo-classification and calculate its loss using combined features
        geo_class_output = self.geo_classifier(query_combined_features)
        loss_geo_classification = self.geo_criterion(geo_class_output, geo_labels)

        # Combine losses
        total_loss = loss_contrastive + loss_geo_classification

        # Log training metrics
        self.log("train_loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_geo_cls", loss_geo_classification, on_step=False, on_epoch=True, logger=True)
        
        # Log training accuracy
        train_geo_accuracy = self.geo_accuracy(geo_class_output.softmax(dim=-1), geo_labels)
        self.log("train_geo_acc", train_geo_accuracy, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Performs a single validation step to evaluate performance on unseen data.
        """
        # Unpack the batch (only query image, tabular features, and label are typically used for validation)
        x_query_image, _, tabular_features, geo_labels = batch 
        
        # Get query features (no momentum update in validation)
        # query_image_features are for contrastive loss (though not used in validation step here),
        # query_combined_features for classification
        _, query_combined_features = self.forward(x_query_image, tabular_features)

        # Perform geo-classification and calculate its loss
        geo_class_output = self.geo_classifier(query_combined_features)
        loss_geo_classification = self.geo_criterion(geo_class_output, geo_labels)

        # Calculate validation accuracy
        val_geo_accuracy = self.geo_accuracy(geo_class_output.softmax(dim=-1), geo_labels)

        # Log validation metrics
        self.log("val_loss_geo_cls", loss_geo_classification, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_geo_acc", val_geo_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss_geo_cls": loss_geo_classification, "val_geo_acc": val_geo_accuracy}

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
        return optimizer

