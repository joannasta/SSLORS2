import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomCrop


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels_from_prev_layer, in_channels_skip_connection, out_channels_for_this_block):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_from_prev_layer, out_channels_for_this_block, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels_for_this_block + in_channels_skip_connection, out_channels_for_this_block)

    def forward(self, x_upsampled, x_skip):
        x_upsampled = self.up(x_upsampled)
        diffY = x_skip.size()[2] - x_upsampled.size()[2]
        diffX = x_skip.size()[3] - x_upsampled.size()[3]
        x_upsampled = F.pad(x_upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_skip, x_upsampled], dim=1)
        return self.conv(x)


class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, model_type="mae", full_finetune=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.full_finetune = full_finetune

        self.n_channels = in_channels
        self.n_outputs = out_channels

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        cat_channels_in = 256

        if model_type in ["mae", "moco"]:
            cat_channels_in += 256
            self.emb_proj = nn.Linear(512, 256 * 32 * 32)
        elif model_type == "mocogeo":
            cat_channels_in += 512
        else:
            raise NotImplementedError(f"Model type '{model_type}' not supported for embedding processing in UNet_bathy.")

        self.cat_to_bot_proj = nn.Linear(cat_channels_in, 512)

        self.bottleneck = DoubleConv(512, 512)

        self.up1 = Up(512, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x_embedding, images):
        images = images.to(self.device)

        x1 = self.inc(images)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        fused_features = None

        if self.model_type in ["mae", "moco"]:
            emb_spat_proj = self.emb_proj(x_embedding).view(x_embedding.shape[0], 256, 32, 32)
            emb_interp = F.interpolate(emb_spat_proj, size=x4.shape[2:], mode='bilinear', align_corners=False)
            fused_features = torch.cat([x4, emb_interp], dim=1)

        elif self.model_type == "mocogeo":
            emb_expanded = x_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, x4.shape[2], x4.shape[3])
            fused_features = torch.cat([x4, emb_expanded], dim=1)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not supported during forward pass.")

        bs, ch_fused, h_fused, w_fused = fused_features.shape
        fused_reshaped = fused_features.permute(0, 2, 3, 1).reshape(bs * h_fused * w_fused, ch_fused)

        bottleneck_in = self.cat_to_bot_proj(fused_reshaped).reshape(bs, 512, h_fused, w_fused)

        bottleneck_out = self.bottleneck(bottleneck_in)

        x = self.up1(bottleneck_out, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        output = self.outc(x)
        return output