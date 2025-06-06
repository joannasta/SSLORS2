import copy
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from typing import Callable
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class GeoClassifier(nn.Module):
    """
    A simple classifier head for geo-classification.
    """
    def __init__(self, input_dim: int = 128, num_classes: int = 100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input features if they come from a convolutional backbone
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class MoCoGeo(pl.LightningModule):
    """
    MoCo-Geo: A PyTorch Lightning module for MoCo v2 style self-supervised
    learning with an auxiliary geo-classification task.
    """
    def __init__(
        self,
        base_encoder: Callable = torchvision.models.resnet18,
        dim: int = 128,      # Output dimension of the projection head
        K: int = 4096,       # Memory bank size for NTXentLoss
        m: float = 0.99,     # Momentum coefficient for momentum encoder update
        T: float = 0.07,     # Temperature for NTXentLoss
        src_channels: int = 3, # Number of input channels for the images
        num_geo_classes: int = 100 # Number of geographical classes
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
        self.geo_criterion = nn.CrossEntropyLoss()

        # Auxiliary geo-classifier and its metric
        self.geo_classifier = GeoClassifier(
            input_dim=self.hparams.dim, num_classes=self.hparams.num_geo_classes
        )
        self.geo_accuracy = MulticlassAccuracy(
            num_classes=self.hparams.num_geo_classes, average='micro'
        )

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

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, including MoCo update and geo-classification.
        """
        # Update momentum encoder's weights
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        # Unpack the batch: (query_image, key_image, geo_label)
        x_query, x_key, geo_labels = batch

        # Get features from online and momentum encoders
        query_features = self.forward(x_query)
        key_features = self.forward_momentum(x_key)

        # Calculate contrastive loss
        loss_contrastive = self.contrastive_loss(query_features, key_features)

        # Perform geo-classification and calculate its loss
        geo_class_output = self.geo_classifier(query_features)
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
        # Unpack the batch (only query image and label are typically used for validation)
        x_query, _, geo_labels = batch 
        
        # Get query features (no momentum update in validation)
        query_features = self.forward(x_query)

        # Perform geo-classification and calculate its loss
        geo_class_output = self.geo_classifier(query_features)
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