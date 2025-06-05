import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils

class MoCo(pl.LightningModule):
    def __init__(self,learning_rate=3e-3,src_channels=12):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        self.src_channels = src_channels
        self.learning_rate = learning_rate
        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=( 4096, 128) #4096 #65536
        )
        # Initialize variables to accumulate the loss
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0

    def load_pretrained_weights(self, pretrained_weights):
        """Load pretrained weights into the model."""
        checkpoint = torch.load(pretrained_weights)
        # Load only the encoder and decoder weights, skipping classifier if any.
        model_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        self.load_state_dict(model_state_dict)
        print("Pretrained weights loaded successfully.")


    def training_step(self, batch, batch_idx):
        x_q, x_k = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("epoch", float(self.current_epoch), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Optionally accumulate loss for manual tracking (can use self.log for epochs instead)
        if not hasattr(self, 'total_train_loss'):
            self.total_train_loss = 0.0 # Initialize if not already defined
            self.train_batch_count = 0
        self.total_train_loss += loss.item()
        self.train_batch_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x_q, x_k = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Accumulate validation loss
        self.total_val_loss += loss.item()
        self.val_batch_count += 1

        return loss

    def on_train_epoch_end(self):
        # Log the average training loss for the epoch
        avg_train_loss = self.total_train_loss / self.train_batch_count
        # Reset accumulation variables for next epoch
        self.total_train_loss = 0.0
        self.train_batch_count = 0

        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")
        self.custom_histogram_weights()

    def on_validation_epoch_end(self):
        # Log the average validation loss for the epoch
        avg_val_loss = self.total_val_loss / self.val_batch_count
        # Reset accumulation variables for next epoch
        self.total_val_loss = 0.0
        self.val_batch_count = 0

        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")
        self.custom_histogram_weights()

    def on_train_epoch_start(self):
        # Log the current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch - Current learning rate: {current_lr}")


    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100)
        return [optim], [scheduler]

class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y= batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        # calculate number of correct predictions
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.validation_step_outputs.append((num, correct))
        return num, correct

    def on_validation_epoch_end(self):
        # calculate and log top1 accuracy
        if self.validation_step_outputs:
            total_num = 0
            total_correct = 0
            for num, correct in self.validation_step_outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=self.learning_rate, weight_decay=5e-4,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 200)
        return [optim], [scheduler]