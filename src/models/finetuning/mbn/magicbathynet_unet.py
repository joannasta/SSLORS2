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
        print(f"Initializing DoubleConv ({in_channels} -> {out_channels})...")  # Print initialization info
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.double_conv(x)
        return output


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        print(f"Initializing Down ({in_channels} -> {out_channels})...")  # Print initialization info

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        output = self.maxpool_conv(x)
        return output


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculate the difference in shape and pad x1 if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        output = self.conv(x)
        return output


class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels=192, latent_size=32,model_type="mae", full_finetune=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.full_finetune = full_finetune

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels  
        self.latent_size = latent_size  

        self.channel_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # Projection for x4
        self.combined_projection = nn.Linear(257, 256)  # Linear projection layer, adjusted input channels

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256)
        )

        self.decoder = nn.Sequential(
            Up(256, 128),
            Up(128, 64),
            Up(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

        

        if model_type in ["mae", "moco"]:
            self.bottleneck_input_channels = 256 + 256
            # Change the input features for the linear projector to match the embedding dimension (768)
            self.mae_embedding_projector = nn.Linear(768, 256 * 32 * 32)
            self.moco_embedding_projector = nn.Linear(768, 256 * 32 * 32)
        elif model_type == "mocogeo":
            self.bottleneck_input_channels = 256 + 512

        self.bottleneck_conv = DoubleConv(self.bottleneck_input_channels, 512)

        

    def forward(self, x_embedding, images):
        images = images.to(self.device)
        print(f"DEBUG: x_embedding initial shape: {x_embedding.shape}") # ADD THIS LINE
        print(f"DEBUG: images initial shape: {images.shape}") # ADD THIS LINE
        
        images = images.float()  # Ensure images are float type

        x1 = self.encoder[0](images)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)


        if self.model_type == "mae":
            x_resized = F.interpolate(x_embedding, size=x4.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([x_resized, x4], dim=1)  # Concatenate along channel dimension
    
            batch_size, channels, height, width = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
            combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)

        elif self.model_type == "moco":
            embedding = self.moco_embedding_projector(x_embedding)
            processed_embedding = embedding.view(embedding.shape[0], 256, 32, 32)
            combined = torch.cat([x4, processed_embedding], dim=1)
            batch_size, channels, height, width = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
            combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)

        elif self.model_type == "mocogeo":
            embedding = x_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, x4.shape[2], x4.shape[3])
            combined = torch.cat([x4, embedding], dim=1)
            batch_size, channels, height, width = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
            combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)

        #x = self.decoder[0](combined_projected, x3)
        x = self.decoder[0](x4, x3)
        x = self.decoder[1](x, x2)
        x = self.decoder[2](x, x1)
        output = self.decoder[3](x)

        return output