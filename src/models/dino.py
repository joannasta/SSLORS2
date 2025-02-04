import copy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from torchinfo import summary

class DINO_LIT(pl.LightningModule):
    def __init__(self, input_channels=12):
        super().__init__()
        
        # Load the DINO backbone and modify the input channels
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        self.modify_input_channels(backbone, input_channels)
        input_dim = backbone.embed_dim

        # Initialize student and teacher networks with the modified backbone
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def modify_input_channels(self, backbone, input_channels):
        # Modify the first layer to accept `input_channels` instead of 3
        if hasattr(backbone, 'conv1'):
            weight = backbone.conv1.weight.clone()
            backbone.conv1 = nn.Conv2d(input_channels, weight.shape[0], kernel_size=backbone.conv1.kernel_size, stride=backbone.conv1.stride, padding=backbone.conv1.padding, bias=backbone.conv1.bias)
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = weight
                if input_channels > 3:
                    backbone.conv1.weight[:, 3:] = weight[:, :input_channels - 3]
        elif hasattr(backbone, 'patch_embed'):
            backbone.patch_embed.proj = nn.Conv2d(input_channels, backbone.patch_embed.proj.out_channels, kernel_size=backbone.patch_embed.proj.kernel_size, stride=backbone.patch_embed.proj.stride, padding=backbone.patch_embed.proj.padding)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


if __name__ == '__main__':
    model = DINO_LIT(input_channels=12)
    
    # Display model summary for 12-channel 256x256 images
    summary(model, input_size=(1, 12, 256, 256))

    in_tensor = torch.randn(1, 12, 256, 256)
    print("in shape:\t\t", in_tensor.shape)
