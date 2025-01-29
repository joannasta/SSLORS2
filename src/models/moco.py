import copy

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
from torchinfo import summary

class MoCo(pl.LightningModule):
    def __init__(self, input_channels=12):
        super().__init__()
        # Define a custom backbone for 12-channel input (e.g., using a simple CNN or a modified ResNet)
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Additional layers or modifications as needed
        )

        # Define the projection head with the appropriate input dimension
        self.projection_head = MoCoProjectionHead(512, 512, 128)  # Adjust 512 as needed

        # Create momentum backbone and projection head
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=4096)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        
        # Assuming batch provides views with 12 channels
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

# Example usage in a training script
if __name__ == '__main__':
    model = MoCo(input_channels=12)
    summary(model, input_size=(1, 12, 128, 128))

    # Sample input
    in_tensor = torch.randn(1, 12, 128, 128)
    print("Input shape:", in_tensor.shape)
