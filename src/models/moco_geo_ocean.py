import copy
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from typing import Callable, Tuple
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class OceanFeatureClassifier(nn.Module):
    def __init__(self, input_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class MoCoOceanFeatures(pl.LightningModule):
    def __init__(
        self,
        base_encoder: Callable = torchvision.models.resnet18,
        dim: int = 128,
        K: int = 4096,
        m: float = 0.99,
        T: float = 0.07,
        src_channels: int = 3,
        num_classes: int = 3
    ):
        super().__init__()
        self.save_hyperparameters()

        resnet = base_encoder(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            self.hparams.src_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = MoCoProjectionHead(512, 512, self.hparams.dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.contrastive_loss = NTXentLoss(
            memory_bank_size=(self.hparams.K, self.hparams.dim),
            temperature=self.hparams.T
        )
        self.classification_criterion = nn.CrossEntropyLoss()

        self.ocean_classifier = OceanFeatureClassifier(
            input_dim=self.hparams.dim, num_classes=self.hparams.num_classes
        )
        self.classification_metric = MulticlassAccuracy(
            num_classes=self.hparams.num_classes, average='micro'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(features)
        return query

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone_momentum(x).flatten(start_dim=1)
            key = self.projection_head_momentum(features)
        return key

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.LongTensor], batch_idx: int) -> torch.Tensor:
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        x_query, x_key, cluster_labels = batch 

        query_features = self.forward(x_query)
        key_features = self.forward_momentum(x_key)

        loss_contrastive = self.contrastive_loss(query_features, key_features)

        predicted_cluster_logits = self.ocean_classifier(query_features)
        loss_classification = self.classification_criterion(predicted_cluster_logits, cluster_labels)

        total_loss = loss_contrastive + loss_classification

        self.log("train_loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_classification", loss_classification, on_step=False, on_epoch=True, logger=True)
        
        train_classification_metric = self.classification_metric(predicted_cluster_logits, cluster_labels)
        self.log("train_accuracy", train_classification_metric, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.LongTensor], batch_idx: int) -> dict:
        x_query, x_key, cluster_labels = batch 
        
        query_features = self.forward(x_query)
        key_features = self.forward_momentum(x_key)

        loss_contrastive = self.contrastive_loss(query_features, key_features)
        predicted_cluster_logits = self.ocean_classifier(query_features)
        loss_classification = self.classification_criterion(predicted_cluster_logits, cluster_labels)
        total_loss = loss_contrastive + loss_classification

        val_classification_metric = self.classification_metric(predicted_cluster_logits, cluster_labels)

        self.log("val_loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log("val_loss_classification", loss_classification, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", val_classification_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "val_loss_total": total_loss,
            "val_loss_contrastive": loss_contrastive,
            "val_loss_classification": loss_classification,
            "val_accuracy": val_classification_metric
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
        return optimizer