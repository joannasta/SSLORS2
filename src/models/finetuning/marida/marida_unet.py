import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_Marida(nn.Module):
    def __init__(self, input_channels=11, out_channels=1, hidden_channels=16, embedding_dim=128,model_type="mae"):
        super(UNet_Marida, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))

        self.mae_feature_fusion_projection = nn.Linear(8 * hidden_channels + embedding_dim, 8 * hidden_channels)
        self.moco_projection = nn.Linear(in_features=512, out_features=embedding_dim * 16 * 16)
        self.mocogeo_projection = nn.Linear(in_features=512, out_features=embedding_dim * 16 * 16)

        self.combined_projection = nn.Linear(256, 256)

        self.down1 = Down(hidden_channels, 2 * hidden_channels)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)

        self.up1 = Up(256 + (8 * hidden_channels), 4 * hidden_channels)
        self.up2 = Up((4 * hidden_channels) + (4 * hidden_channels), 2 * hidden_channels)
        self.up3 = Up((2 * hidden_channels) + (2 * hidden_channels), hidden_channels)
        self.up4 = Up(hidden_channels + hidden_channels, hidden_channels)

        self.outc = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        
    def forward(self, image, x_embedding):
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        combined_projected = None

        if self.model_type == "mae":
            x_embedding_spatial = x_embedding
            combined = torch.cat([x_embedding_spatial, x5], dim=1)
            
            batch_size, channels, height, width = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
            
            processed_linear_output = self.combined_projection(combined_reshaped)
            
            combined_projected = processed_linear_output.reshape(batch_size, self.combined_projection.out_features, height, width)

        elif self.model_type == "moco":
            x_embedding_flat_projected = self.moco_projection(x_embedding)
            processed_x_embedding = x_embedding_flat_projected.view(
                x5.shape[0], self.embedding_dim, x5.shape[2], x5.shape[3]
            )
            combined = torch.cat([processed_x_embedding, x5], dim=1)

            batch_size, channels_combined, height, width = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels_combined)
            
            processed_linear_output = self.combined_projection(combined_reshaped)
            combined_projected = processed_linear_output.reshape(batch_size, self.combined_projection.out_features, height, width)

        elif self.model_type == "mocogeo":
            x_embedding_flat_projected = self.mocogeo_projection(x_embedding)
            
            processed_x_embedding = x_embedding_flat_projected.view(
                x5.shape[0], self.embedding_dim, x5.shape[2], x5.shape[3]
            )
            combined = torch.cat([processed_x_embedding, x5], dim=1)

            batch_size, channels_combined, height, width = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels_combined)
            
            processed_linear_output = self.combined_projection(combined_reshaped)
            combined_projected = processed_linear_output.reshape(batch_size, self.embedding_dim, height, width)


        x6 = self.up1(combined_projected, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.outc(x9)
        return logits