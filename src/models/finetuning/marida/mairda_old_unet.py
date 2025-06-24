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
        print(f"  Up: x1 (upsampled input) before upsample: {x1.shape}")
        x1 = self.up(x1)
        print(f"  Up: x1 (upsampled input) after upsample: {x1.shape}")
        print(f"  Up: x2 (skip connection) shape: {x2.shape}")

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            print(f"  Up: Padding x1. DiffY={diffY}, DiffX={diffX}")
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            print(f"  Up: x1 after padding: {x1.shape}")
        
        x = torch.cat([x2, x1], dim=1)
        print(f"  Up: Concatenated tensor shape: {x.shape}")
        out = self.conv(x)
        print(f"  Up: Output shape: {out.shape}")
        return out

class UNet_Marida(nn.Module):
    def __init__(self, input_channels=11, out_channels=11, hidden_channels=16):
        super(UNet_Marida, self).__init__()
        print(f"UNet_Marida Init: input_channels={input_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}")
        
        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))

        self.down1 = Down(hidden_channels, 2 * hidden_channels)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)

        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels)
        self.up3 = Up(4 * hidden_channels, hidden_channels)
        self.up4 = Up(2 * hidden_channels, hidden_channels)

        self.outc = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.channel_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.combined_projection = nn.Linear(129, 128) 


    def forward(self, x, image):
        print(f"\nUNet_Marida Forward Input: image.shape={image.shape}, x (embedding).shape={x.shape}")

        x1 = self.inc(image)
        print(f"Encoder: x1 (inc output) shape: {x1.shape}")

        x2 = self.down1(x1)
        print(f"Encoder: x2 (down1 output) shape: {x2.shape}")
        x3 = self.down2(x2)
        print(f"Encoder: x3 (down2 output) shape: {x3.shape}")
        x4 = self.down3(x3)
        print(f"Encoder: x4 (down3 output) shape: {x4.shape}")
        x5 = self.down4(x4)
        print(f"Encoder: x5 (down4 output - bottleneck) shape: {x5.shape}")

        print(f"Attempting to interpolate x (embedding): {x.shape} to target size: {x5.shape[2:]}")
        x_resized = torch.nn.functional.interpolate(x, size=x5.shape[2:], mode='bilinear')
        print(f"x_resized (interpolated embedding) shape: {x_resized.shape}")
        
        print(f"Concatenating x_resized {x_resized.shape} and x5 {x5.shape}")
        combined = torch.cat([x_resized, x5], dim=1)
        print(f"Combined tensor after concat: {combined.shape}")

        batch_size, channels, height, width = combined.shape
        print(f"Reshaping combined for linear projection: {combined.shape} -> ({batch_size * height * width}, {channels})")
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        print(f"Shape after permute and reshape: {combined_reshaped.shape}")
        
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 128, 16, 16)
        print(f"combined_projected (after linear and reshape) shape: {combined_projected.shape}")

        print("\nDecoder Path:")
        x6 = self.up1(combined_projected, x4)
        print(f"Decoder: x6 (up1 output) shape: {x6.shape}")
        x7 = self.up2(x6, x3)
        print(f"Decoder: x7 (up2 output) shape: {x7.shape}")
        x8 = self.up3(x7, x2)
        print(f"Decoder: x8 (up3 output) shape: {x8.shape}")
        x9 = self.up4(x8, x1)
        print(f"Decoder: x9 (up4 output) shape: {x9.shape}")

        logits = self.outc(x9)
        print(f"Output: logits shape: {logits.shape}")
        return logits