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
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out

class UNet_Marida(nn.Module):
    def __init__(self, input_channels=11, out_channels=11, hidden_channels=16, embedding_dim=128):
        super(UNet_Marida, self).__init__()
        
        # Store hidden_channels as an instance variable
        self.hidden_channels = hidden_channels 

        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True))

        self.down1 = Down(self.hidden_channels, 2 * self.hidden_channels)
        self.down2 = Down(2 * self.hidden_channels, 4 * self.hidden_channels)
        self.down3 = Down(4 * self.hidden_channels, 8 * self.hidden_channels)
        self.down4 = Down(8 * self.hidden_channels, 8 * self.hidden_channels)

        self.up1 = Up(16 * self.hidden_channels, 4 * self.hidden_channels)
        self.up2 = Up(8 * self.hidden_channels, 2 * self.hidden_channels)
        self.up3 = Up(4 * self.hidden_channels, self.hidden_channels)
        self.up4 = Up(2 * self.hidden_channels, self.hidden_channels)

        self.outc = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        self.embedding_dim = embedding_dim 
        expected_in_features = self.embedding_dim + (8 * self.hidden_channels)
        self.combined_projection = nn.Linear(expected_in_features, 8 * self.hidden_channels)
        print(f"DEBUG: Initializing combined_projection with in_features={expected_in_features} (embedding_dim={self.embedding_dim}, hidden_channels={self.hidden_channels})")


    def forward(self, x, image):
        
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        x5 = self.down4(x4)

        if x.dim() == 3 and x.shape[1] == 1: # (batch_size, 1, features)
            x_reshaped_for_interp = x.squeeze(1).unsqueeze(2).unsqueeze(3) # -> (batch_size, features, 1, 1)
            print(f"DEBUG: Handled 3D input (B,1,F): {x.shape} -> {x_reshaped_for_interp.shape}")
        elif x.dim() == 2: # (batch_size, features)
            x_reshaped_for_interp = x.unsqueeze(2).unsqueeze(3) # -> (batch_size, features, 1, 1)
            print(f"DEBUG: Handled 2D input (B,F): {x.shape} -> {x_reshaped_for_interp.shape}")
        elif x.dim() == 4: # (batch_size, channels, H, W)
            x_reshaped_for_interp = x
            print(f"DEBUG: Handled 4D input (B,C,H,W): {x.shape} -> {x_reshaped_for_interp.shape}")
        else:
            raise ValueError(f"Unsupported dimension for input x: {x.dim()}. Expected (B,F), (B,1,F), or (B,C,H,W). Received: {x.shape}")

        x_resized = torch.nn.functional.interpolate(x_reshaped_for_interp, size=x5.shape[2:], mode='bilinear', align_corners=True)
        
        combined = torch.cat([x_resized, x5], dim=1)

        batch_size, channels, height, width = combined.shape
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        
        # DEBUG PRINTS to see what's actually going into the linear layer
        print(f"DEBUG: x_resized shape: {x_resized.shape}")
        print(f"DEBUG: x5 shape: {x5.shape}")
        print(f"DEBUG: combined shape: {combined.shape}")
        print(f"DEBUG: combined_reshaped shape (input to linear): {combined_reshaped.shape}")
        print(f"DEBUG: self.combined_projection.weight.shape (linear layer's mat2): {self.combined_projection.weight.shape}")

        # Use self.hidden_channels
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 8 * self.hidden_channels, height, width)

        x6 = self.up1(combined_projected, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.outc(x9)
        return logits