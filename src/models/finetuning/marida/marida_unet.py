import torch
import numpy as np
import random
import torch.nn.functional as F

from torch import nn

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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_Marida(nn.Module):
    def __init__(self, input_channels=11, out_channels=1, hidden_channels=16, embedding_dim=128, model_type="mae"):
        super(UNet_Marida, self).__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.embedding_dim = embedding_dim

        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))
        
        bottleneck_input_channels = 8 * hidden_channels
        
        if model_type == "mae":
            bottleneck_input_channels += embedding_dim
            self.mae_pre_proj = nn.Linear(768, embedding_dim) 
            self.mae_spatial_proj = nn.Linear(embedding_dim, embedding_dim * 16 * 16)
        elif model_type == "mae_ocean":
            bottleneck_input_channels += embedding_dim
            self.mae_ocean_pre_proj = nn.Linear(768, embedding_dim) 
            self.mae_ocean_spatial_proj = nn.Linear(embedding_dim, embedding_dim * 16 * 16)
        elif model_type in ["moco", "geo_aware", "ocean_aware"]:
            bottleneck_input_channels += embedding_dim
            self.moco_pre_proj = nn.Linear(512, embedding_dim)
            self.moco_spatial_proj = nn.Linear(embedding_dim, embedding_dim * 16 * 16)
        
        self.feature_fusion_proj = nn.Linear(bottleneck_input_channels, bottleneck_input_channels)

        self.down1 = Down(hidden_channels, 2 * hidden_channels)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)

        self.up1 = Up(bottleneck_input_channels + (8 * hidden_channels), 4 * hidden_channels)
        self.up2 = Up((4 * hidden_channels) + (4 * hidden_channels), 2 * hidden_channels)
        self.up3 = Up((2 * hidden_channels) + (2 * hidden_channels), hidden_channels)
        self.up4 = Up(hidden_channels + hidden_channels, hidden_channels)

        self.outc = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
            
    def forward(self, image, x_embedding):
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        fused_features = None
        
        if self.model_type in ["mae", "mae_ocean"]:
            batch_size, _, H_x5, W_x5 = x5.shape 
            
            if x_embedding.dim() == 4 and x_embedding.shape[1] == 1:
                x_embedding = x_embedding.squeeze(1) 
            
            cls_token = x_embedding[:, 0, :]
            
            if self.model_type == "mae":
                transformed_embedding = self.mae_pre_proj(cls_token)
                spatial_proj_flat = self.mae_spatial_proj(transformed_embedding)
            else: # mae_ocean
                transformed_embedding = self.mae_ocean_pre_proj(cls_token)
                spatial_proj_flat = self.mae_ocean_spatial_proj(transformed_embedding)
            
            embedding_spatial = spatial_proj_flat.view(batch_size, self.embedding_dim, H_x5, W_x5)

            combined = torch.cat([embedding_spatial, x5], dim=1)
            
            batch_size_c, channels_c, height_c, width_c = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(-1, channels_c)
            
            processed_linear_output = self.feature_fusion_proj(combined_reshaped)
            
            fused_features = processed_linear_output.reshape(batch_size_c, self.feature_fusion_proj.out_features, height_c, width_c)

        elif self.model_type in ["moco", "geo_aware", "ocean_aware"]:
            if x_embedding.dim() == 4 and x_embedding.shape[1] == 1:
                x_embedding = x_embedding.squeeze(1)
            if x_embedding.dim() == 3 and x_embedding.shape[2] == 512:
                x_embedding = torch.mean(x_embedding, dim=1) 

            spatial_proj_flat = self.moco_spatial_proj(x_embedding)
            
            embedding_spatial = spatial_proj_flat.view(x5.shape[0], self.embedding_dim, x5.shape[2], x5.shape[3])
            
            combined = torch.cat([embedding_spatial, x5], dim=1)

            batch_size_c, channels_c, height_c, width_c = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(-1, channels_c)
            
            processed_linear_output = self.feature_fusion_proj(combined_reshaped)
            fused_features = processed_linear_output.reshape(batch_size_c, self.feature_fusion_proj.out_features, height_c, width_c)
        else:
            fused_features = x5 

        x6 = self.up1(fused_features, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits