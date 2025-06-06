import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

# CORRECTED UP MODULE DEFINITION
class Up(nn.Module):
    def __init__(self, in_channels_from_prev_layer, in_channels_skip_connection, out_channels_for_this_block):
        super().__init__()
        # self.up takes input from the previous decoder layer
        self.up = nn.ConvTranspose2d(in_channels_from_prev_layer, out_channels_for_this_block, kernel_size=2, stride=2)
        
        # self.conv takes concatenated input: (upsampled_channels + skip_channels)
        self.conv = DoubleConv(out_channels_for_this_block + in_channels_skip_connection, out_channels_for_this_block)
        
    def forward(self, x1, x2): # x1 comes from previous decoder stage (to be upsampled), x2 is the skip connection
        x1 = self.up(x1)
        
        # Pad x1 if dimensions don't perfectly match x2 (due to odd/even sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate x2 (skip connection) and x1 (upsampled output)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.embedding_projector = nn.Linear(768, 256)
        self.combined_projection = nn.Linear(256 + 256, 256) 

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256) 
        )

        # CORRECTED DECODER INITIALIZATION
        self.decoder = nn.Sequential(
            # Input to first Up: combined_projected (256 channels), Skip: x3 (128 channels), Output: 128 channels
            Up(256, 128, 128), 
            # Input to second Up: from prev (128 channels), Skip: x2 (64 channels), Output: 64 channels
            Up(128, 64, 64),  
            # Input to third Up: from prev (64 channels), Skip: x1 (32 channels), Output: 32 channels
            Up(64, 32, 32),   
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x_embedding, images):
        print(f"Initial images shape: {images.shape}")
        print(f"Initial x_embedding shape: {x_embedding.shape}")

        images = images.float()
        
        x1 = self.encoder[0](images)
        print(f"x1 shape (after DoubleConv): {x1.shape}")
        x2 = self.encoder[1](x1)
        print(f"x2 shape (after Down): {x2.shape}")
        x3 = self.encoder[2](x2)
        print(f"x3 shape (after Down): {x3.shape}")
        x4 = self.encoder[3](x3) 
        print(f"x4 shape (encoder bottleneck): {x4.shape}")

        projected_embedding = self.embedding_projector(x_embedding) 
        print(f"projected_embedding shape (after embedding_projector): {projected_embedding.shape}")

        patch_tokens = projected_embedding[:, 1:, :] 
        print(f"patch_tokens shape (after removing CLS token): {patch_tokens.shape}")
        
        batch_size, num_patches, features = patch_tokens.shape 
        spatial_size = int(num_patches**0.5) 

        spatial_embedding = patch_tokens.permute(0, 2, 1).reshape(
            batch_size, features, spatial_size, spatial_size
        ) 
        print(f"spatial_embedding shape (after reshaping to grid): {spatial_embedding.shape}")

        interpolated_embedding = F.interpolate(
            spatial_embedding, 
            size=(x4.shape[2], x4.shape[3]), 
            mode='bilinear', 
            align_corners=False
        ) 
        print(f"interpolated_embedding shape (after interpolation): {interpolated_embedding.shape}")
        
        combined = torch.cat([interpolated_embedding, x4], dim=1) 
        print(f"combined shape (after concatenation): {combined.shape}")

        batch_size, channels_combined, height, width = combined.shape
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels_combined)
        print(f"combined_reshaped shape (before final projection): {combined_reshaped.shape}")
        
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)
        print(f"combined_projected shape (after final projection): {combined_projected.shape}")

        # Pass combined_projected as x1 (input to upsample) and x3 as x2 (skip connection)
        x = self.decoder[0](combined_projected, x3)
        print(f"x shape (after decoder[0]): {x.shape}")
        x = self.decoder[1](x, x2)
        print(f"x shape (after decoder[1]): {x.shape}")
        x = self.decoder[2](x, x1)
        print(f"x shape (after decoder[2]): {x.shape}")
        output = self.decoder[3](x)
        print(f"Final output shape: {output.shape}")
        return output