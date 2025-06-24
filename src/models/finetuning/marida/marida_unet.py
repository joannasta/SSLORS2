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
    def __init__(self, input_channels=11, out_channels=11, hidden_channels=16, embedding_dim=128, model_type="mae"):
        super(UNet_Marida, self).__init__()

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.embedding_dim = embedding_dim

        if self.model_type == "mae":
            self.embedding_projector = nn.Linear(768, self.embedding_dim)
        elif self.model_type == "moco" or self.model_type == "mocogeo":
            self.embedding_projector = nn.Linear(512, self.embedding_dim)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Based on analysis, this expects self.embedding_dim (128) + 8*self.hidden_channels (128) = 256 input features
        # And outputs 128 * 16 * 16 elements, which will be reshaped to (B, 128, 16, 16)
        self.combined_projection = nn.Linear(self.embedding_dim + (8 * self.hidden_channels), 128 * 16 * 16)

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

        self.up1 = Up(128 + (8 * self.hidden_channels), 4 * self.hidden_channels)
        self.up2 = Up(8 * self.hidden_channels, 2 * self.hidden_channels)
        self.up3 = Up(4 * self.hidden_channels, self.hidden_channels)
        self.up4 = Up(2 * self.hidden_channels, self.hidden_channels)

        self.outc = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

    def forward(self, image, x_embedding):
        print(f"--- UNet_Marida Forward Pass ---")
        print(f"Input image shape: {image.shape}")
        print(f"Input x_embedding shape: {x_embedding.shape}")

        x1 = self.inc(image)
        print(f"x1 (inc output) shape: {x1.shape}")
        x2 = self.down1(x1)
        print(f"x2 (down1 output) shape: {x2.shape}")
        x3 = self.down2(x2)
        print(f"x3 (down2 output) shape: {x3.shape}")
        x4 = self.down3(x3)
        print(f"x4 (down3 output) shape: {x4.shape}")
        x5 = self.down4(x4)
        print(f"x5 (down4 output - bottleneck) shape: {x5.shape}")
        
        print(f"\n--- Embedding Processing ---")
        print(f"x_embedding before projection: {x_embedding.shape}")
        projected_embedding = self.embedding_projector(x_embedding)
        print(f"projected_embedding (after embedding_projector) shape: {projected_embedding.shape}")

        projected_embedding_spatial = projected_embedding.unsqueeze(2).unsqueeze(3)
        print(f"projected_embedding_spatial (unsqueeze to 4D) shape: {projected_embedding_spatial.shape}")
        
        print(f"Attempting to interpolate projected_embedding_spatial: {projected_embedding_spatial.shape} to target size: {x5.shape[2:]}")
        x_resized = torch.nn.functional.interpolate(projected_embedding_spatial, size=x5.shape[2:], mode='bilinear', align_corners=True)
        print(f"x_resized (interpolated embedding) shape: {x_resized.shape}")
        
        print(f"Concatenating x_resized {x_resized.shape} and x5 {x5.shape}")
        combined = torch.cat([x_resized, x5], dim=1)
        print(f"Combined tensor after concat: {combined.shape}")

        batch_size, channels, height, width = combined.shape
        print(f"Reshaping combined for linear projection: {combined.shape} -> ({batch_size * height * width}, {channels})")
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        print(f"Shape after permute and reshape: {combined_reshaped.shape}")
        
        # Check the actual input size to the linear layer vs expected in_features
        expected_in_features = self.embedding_dim + (8 * self.hidden_channels) # Should be 256
        print(f"Linear layer (self.combined_projection) expects in_features: {expected_in_features}")
        print(f"Actual input features to linear layer (combined_reshaped's last dim): {combined_reshaped.shape[1]}")

        # Check the actual output size of the linear layer vs expected for reshape
        expected_linear_output_elements = 128 * 16 * 16 # Should be 32768
        print(f"Linear layer (self.combined_projection) expects to output {expected_linear_output_elements} elements")
        
        linear_output = self.combined_projection(combined_reshaped)
        print(f"Shape after linear projection: {linear_output.shape}")
        
        print(f"Attempting to reshape linear output: {linear_output.shape} to ({batch_size}, 128, 16, 16)")
        combined_projected = linear_output.reshape(batch_size, 128, 16, 16)
        print(f"combined_projected (after linear and reshape) shape: {combined_projected.shape}")

        print(f"\n--- Decoder Path ---")
        print(f"up1 input (upsampled previous layer) shape: {combined_projected.shape}")
        print(f"up1 skip connection (x4) shape: {x4.shape}")
        x6 = self.up1(combined_projected, x4)
        print(f"x6 (up1 output) shape: {x6.shape}")

        print(f"up2 input (upsampled x6) shape for Up class conv: {x6.shape} after upsample")
        print(f"up2 skip connection (x3) shape: {x3.shape}")
        x7 = self.up2(x6, x3)
        print(f"x7 (up2 output) shape: {x7.shape}")

        print(f"up3 input (upsampled x7) shape for Up class conv: {x7.shape} after upsample")
        print(f"up3 skip connection (x2) shape: {x2.shape}")
        x8 = self.up3(x7, x2)
        print(f"x8 (up3 output) shape: {x8.shape}")

        print(f"up4 input (upsampled x8) shape for Up class conv: {x8.shape} after upsample")
        print(f"up4 skip connection (x1) shape: {x1.shape}")
        x9 = self.up4(x8, x1)
        print(f"x9 (up4 output) shape: {x9.shape}")

        logits = self.outc(x9)
        print(f"logits (final output) shape: {logits.shape}")
        print(f"--- UNet_Marida Forward Pass End ---")
        return logits