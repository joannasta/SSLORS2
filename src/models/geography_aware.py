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
    """Simple MLP head for geographic classification on top of projected features."""
    def __init__(self, input_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten if needed and apply classifier."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class GeographyAware(pl.LightningModule):
     """Contrastive pretraining with momentum encoder + additional geography classifier."""
    def __init__(
        self,
        base_encoder: Callable = torchvision.models.resnet18,
        dim: int = 128,      # Output dimension of the projection head
        K: int = 4096,       # Memory bank size for NTXentLoss
        m: float = 0.99,     # Momentum coefficient for momentum encoder update
        T: float = 0.07,     # Temperature for NTXentLoss
        src_channels: int = 3, 
        num_geo_classes: int = 10 
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone: ResNet without final FC
        resnet = base_encoder(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            self.hparams.src_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head maps backbone features to contrastive embedding space
        self.projection_head = MoCoProjectionHead(512, 512, self.hparams.dim)

        # Momentum (EMA) copies of backbone and head (frozen grads)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Contrastive loss with memory bank and temperature
        self.contrastive_loss = NTXentLoss(
            memory_bank_size=(self.hparams.K, self.hparams.dim),
            temperature=self.hparams.T
        )
        
        # Additional geography classifier + loss/metric
        self.geo_criterion = nn.CrossEntropyLoss()

        self.geo_classifier = GeoClassifier(
            input_dim=self.hparams.dim, num_classes=self.hparams.num_geo_classes
        )
        self.geo_accuracy = MulticlassAccuracy(
            num_classes=self.hparams.num_geo_classes, average='micro'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images and project to contrastive embedding."""
        features = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(features)
        return query

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Momentum key branch uses EMA-updated encoder."""
        with torch.no_grad():
            features = self.backbone_momentum(x).flatten(start_dim=1)
            key = self.projection_head_momentum(features)
        return key

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute contrastive + geo classification losses and update EMA encoder."""
        
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        x_query, x_key, geo_labels = batch

        # Forward passes
        query_features = self.forward(x_query)
        key_features = self.forward_momentum(x_key)

        # Losses: contrastive + geo classification
        loss_contrastive = self.contrastive_loss(query_features, key_features)
        geo_class_output = self.geo_classifier(query_features)
        loss_geo_classification = self.geo_criterion(geo_class_output, geo_labels)

        total_loss = loss_contrastive + loss_geo_classification

        self.log("train_loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_geo_cls", loss_geo_classification, on_step=False, on_epoch=True, logger=True)
        
        train_geo_accuracy = self.geo_accuracy(geo_class_output.softmax(dim=-1), geo_labels)
        self.log("train_geo_acc", train_geo_accuracy, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """Validation: compute total loss and geo accuracy."""
        x_query, x_key, geo_labels = batch 
        
        query_features = self.forward(x_query)
        key_features = self.forward_momentum(x_key)

        loss_contrastive = self.contrastive_loss(query_features, key_features)
        geo_class_output = self.geo_classifier(query_features)
        loss_geo_classification = self.geo_criterion(geo_class_output, geo_labels)

        total_loss = loss_contrastive + loss_geo_classification

        val_geo_accuracy = self.geo_accuracy(geo_class_output.softmax(dim=-1), geo_labels)

        self.log("val_loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log("val_loss_geo_cls", loss_geo_classification, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_geo_acc", val_geo_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "val_loss_total": total_loss,
            "val_loss_contrastive": loss_contrastive,
            "val_loss_geo_cls": loss_geo_classification,
            "val_geo_acc": val_geo_accuracy
        }

    def configure_optimizers(self):
        """SGD optimizer; add scheduler if needed."""
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
        return optimizer