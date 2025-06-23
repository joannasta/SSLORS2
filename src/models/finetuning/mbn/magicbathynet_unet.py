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
        super().__init__()
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
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels_from_prev_layer, in_channels_skip_connection, out_channels_for_this_block):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels_from_prev_layer, out_channels_for_this_block, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels_for_this_block + in_channels_skip_connection, out_channels_for_this_block)
    def forward(self, x1, x2): 
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, model_type="mae", full_finetune=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.full_finetune = full_finetune
        
        self.target_embedding_flat_size = 256 * 32 * 32 
        
        self.embedding_projector = nn.Linear(768, self.target_embedding_flat_size)
        self.moco_projection = nn.Linear(in_features=512, out_features=self.target_embedding_flat_size)
        self.mocogeo_projection = nn.Linear(in_features=512, out_features=self.target_embedding_flat_size)
        
        self.combined_projection = nn.Linear(256 + 256, 256) 

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256)
        )
        self.decoder = nn.Sequential(
            Up(256, 128, 128), 
            Up(128, 64, 64),   
            Up(64, 32, 32),    
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x_embedding, images):
        images = images.to(self.device) 
        print("unet")
        print(f"Input images shape: {images.shape}")
        
        x1 = self.encoder[0](images) 
        x2 = self.encoder[1](x1)    
        x3 = self.encoder[2](x2)    
        x4 = self.encoder[3](x3)    

        # Process the pretrained embedding (x_embedding) into a spatial feature map matching x4's dimensions
        processed_x_embedding_spatial = None
        if self.model_type == "mae":
            x_embedding_flat = self.embedding_projector(x_embedding)
            processed_x_embedding_spatial = x_embedding_flat.view(x_embedding_flat.shape[0], x4.shape[1], x4.shape[2], x4.shape[3])
        elif self.model_type == "moco": 
            x_embedding_flat = self.moco_projection(x_embedding)
            processed_x_embedding_spatial = x_embedding_flat.view(x_embedding_flat.shape[0], x4.shape[1], x4.shape[2], x4.shape[3])
        elif self.model_type == "mocogeo": 
            x_embedding_flat = self.mocogeo_projection(x_embedding)
            processed_x_embedding_spatial = x_embedding_flat.view(x_embedding_flat.shape[0], x4.shape[1], x4.shape[2], x4.shape[3])
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not supported for embedding processing in UNet_bathy.")

        # --- Universal bottleneck integration using your preferred logic ---
        # `processed_x_embedding_spatial` is already the same size as `x4`.
        # We directly concatenate them as F.interpolate is redundant for sizing here.
        
        combined_features = torch.cat([processed_x_embedding_spatial, x4], dim=1) # Result: (B, 512, 32, 32)

        batch_size, channels_combined, height, width = combined_features.shape
        print(f"Combined features shape before reshaping: {combined_features.shape}")
        
        combined_reshaped = combined_features.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels_combined)
        
        bottleneck_output = self.combined_projection(combined_reshaped) 
        
        bottleneck_output = bottleneck_output.reshape(batch_size, 256, height, width)
        # --- End of universal integration logic ---

        x = self.decoder[0](bottleneck_output, x3) 
        x = self.decoder[1](x, x2) 
        x = self.decoder[2](x, x1)
        output = self.decoder[3](x)
        return output