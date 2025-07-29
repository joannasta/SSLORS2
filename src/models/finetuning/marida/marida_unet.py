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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_Marida(nn.Module):
    def __init__(self, input_channels=11, out_channels=1, hidden_channels=16, embedding_dim=128, model_type="mae"):
        super(UNet_Marida, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))

        self.mae_spatial_projection = nn.Linear(embedding_dim, embedding_dim * 16 * 16)
        
        self.mae_input_embedding_transform = nn.Linear(768, embedding_dim) 


        self.mae_feature_fusion_projection = nn.Linear(embedding_dim + (8 * hidden_channels), embedding_dim)

        self.moco_projection = nn.Linear(in_features=512, out_features=embedding_dim * 16 * 16)
        self.mocogeo_projection = nn.Linear(in_features=512, out_features=embedding_dim * 16 * 16)

        self.combined_projection = nn.Linear(embedding_dim + (8 * hidden_channels), embedding_dim)


        self.down1 = Down(hidden_channels, 2 * hidden_channels)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)

        self.up1 = Up(embedding_dim + (8 * hidden_channels), 4 * hidden_channels)
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
            batch_size, _, H_x5, W_x5 = x5.shape 
            
            # Ensure x_embedding is 3D [batch_size, seq_len, features] before processing.
            # If it's 4D [1, batch_size, seq_len, features], you need to remove the first dimension.
            if x_embedding.dim() == 4 and x_embedding.shape[0] == 1:
                x_embedding = x_embedding.squeeze(0) # Remove the extra 1st dimension

            # Reduce `x_embedding` from [batch_size, seq_len, features] to [batch_size, features]
            reduced_x_embedding = torch.mean(x_embedding, dim=1) 
            
            # Project the features from `768` to `embedding_dim`
            transformed_x_embedding = self.mae_input_embedding_transform(reduced_x_embedding) 
            
            # Project the `embedding_dim` vector into a flattened spatial size
            projected_embedding_flat = self.mae_spatial_projection(transformed_x_embedding)
            
            # Reshape into a 4D tensor with the same spatial dimensions as x5
            x_embedding_spatial = projected_embedding_flat.view(batch_size, self.embedding_dim, H_x5, W_x5)

            combined = torch.cat([x_embedding_spatial, x5], dim=1)
            
            batch_size_c, channels_c, height_c, width_c = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size_c * height_c * width_c, channels_c)
            
            processed_linear_output = self.mae_feature_fusion_projection(combined_reshaped)
            
            combined_projected = processed_linear_output.reshape(batch_size_c, self.mae_feature_fusion_projection.out_features, height_c, width_c)

        elif self.model_type == "moco":
            # Handle x_embedding's potential 4D shape for consistency
            if x_embedding.dim() == 4 and x_embedding.shape[0] == 1:
                x_embedding = x_embedding.squeeze(0)
            # If moco/mocogeo x_embedding is [batch_size, seq_len, 512], reduce it
            if x_embedding.dim() == 3 and x_embedding.shape[2] == 512:
                x_embedding = torch.mean(x_embedding, dim=1) # Reduce seq_len

            x_embedding_flat_projected = self.moco_projection(x_embedding)
            processed_x_embedding = x_embedding_flat_projected.view(
                x5.shape[0], self.embedding_dim, x5.shape[2], x5.shape[3]
            )
            combined = torch.cat([processed_x_embedding, x5], dim=1)

            batch_size_c, channels_c, height_c, width_c = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size_c * height_c * width_c, channels_c)
            
            processed_linear_output = self.combined_projection(combined_reshaped)
            combined_projected = processed_linear_output.reshape(batch_size_c, self.combined_projection.out_features, height_c, width_c)

        elif self.model_type == "geo_aware ":
            # Handle x_embedding's potential 4D shape for consistency
            if x_embedding.dim() == 4 and x_embedding.shape[0] == 1:
                x_embedding = x_embedding.squeeze(0)
            # If moco/mocogeo x_embedding is [batch_size, seq_len, 512], reduce it
            if x_embedding.dim() == 3 and x_embedding.shape[2] == 512:
                x_embedding = torch.mean(x_embedding, dim=1) # Reduce seq_len
            
            x_embedding_flat_projected = self.mocogeo_projection(x_embedding)
            
            processed_x_embedding = x_embedding_flat_projected.view(
                x5.shape[0], self.embedding_dim, x5.shape[2], x5.shape[3]
            )
            combined = torch.cat([processed_x_embedding, x5], dim=1)

            batch_size_c, channels_c, height_c, width_c = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size_c * height_c * width_c, channels_c)
            
            processed_linear_output = self.combined_projection(combined_reshaped)
            combined_projected = processed_linear_output.reshape(batch_size_c, self.embedding_dim, height_c, width_c)
        elif self.model_type == "ocean_aware":
            # Handle x_embedding's potential 4D shape for consistency
            if x_embedding.dim() == 4 and x_embedding.shape[0] == 1:
                x_embedding = x_embedding.squeeze(0)
            # If moco/mocogeo x_embedding is [batch_size, seq_len, 512], reduce it
            if x_embedding.dim() == 3 and x_embedding.shape[2] == 512:
                x_embedding = torch.mean(x_embedding, dim=1) # Reduce seq_len
            
            x_embedding_flat_projected = self.mocogeo_projection(x_embedding)
            
            processed_x_embedding = x_embedding_flat_projected.view(
                x5.shape[0], self.embedding_dim, x5.shape[2], x5.shape[3]
            )
            combined = torch.cat([processed_x_embedding, x5], dim=1)

            batch_size_c, channels_c, height_c, width_c = combined.shape
            combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size_c * height_c * width_c, channels_c)
            
            processed_linear_output = self.combined_projection(combined_reshaped)
            combined_projected = processed_linear_output.reshape(batch_size_c, self.embedding_dim, height_c, width_c)


        else:
            combined_projected = x5 

        x6 = self.up1(combined_projected, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.outc(x9)
        return logits