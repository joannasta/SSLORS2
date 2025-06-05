import copy
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from torchinfo import summary

# Geo-classifier model (classifier head for MoCo with 100 geo-classes)
class GeoClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=100):
        super(GeoClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# MoCo backbone (MoCo v2 like architecture)
class MoCoGeo(pl.LightningModule):
    def __init__(self, base_encoder=torchvision.models.resnet18, dim=128, K=4096, m=0.99, T=0.07, input_channels=12):
        super(MoCoGeo, self).__init__()
        # Encoder network (backbone + projection head)
        resnet = base_encoder(pretrained=False)
        
        # Modify first layer to accept 12 input channels
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.projection_head = MoCoProjectionHead(512, 512, dim)

        # Momentum encoder
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Freeze momentum encoder
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Loss function
        self.criterion = NTXentLoss(memory_bank_size=(K, dim))
        self.momentum = m
        self.temperature = T

        # Geo-classifier (for downstream task)
        self.geo_classifier = GeoClassifier(input_dim=dim, num_classes=100)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)

        # Contrastive loss
        loss_contrastive = self.criterion(query, key)

        # Downstream geo-classification task
        geo_class_output = self.geo_classifier(query)
        geo_labels = batch[1]  # Assume second element in the batch is the geo-class label
        loss_geo_classification = nn.CrossEntropyLoss()(geo_class_output, geo_labels)

        # Total loss: contrastive loss + classification loss
        total_loss = loss_contrastive + loss_geo_classification
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
        return optimizer

