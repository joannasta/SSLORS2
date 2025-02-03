import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels=192, latent_size=32):
        super(UNet_bathy, self).__init__()
        print("Initializing UNet_bathy...")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels  
        self.latent_size = latent_size  

        self.channel_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # Projection for x4
        self.combined_projection = nn.Linear(256 + 256, 256)  # Linear projection layer, adjusted input channels

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
        print("UNet_bathy initialized.")

    def forward(self, x, images):
        print("Forward pass of UNet_bathy...")

        print(f"Input images shape: {images.shape}")
        print(f"Input x shape: {x.shape}")


        images = images.float()  # Ensure images are float type

        x1 = self.encoder[0](images)
        print(f"x1 shape: {x1.shape}")
        x2 = self.encoder[1](x1)
        print(f"x2 shape: {x2.shape}")
        x3 = self.encoder[2](x2)
        print(f"x3 shape: {x3.shape}")
        x4 = self.encoder[3](x3)
        print(f"x4 shape: {x4.shape}")

        x_resized = F.interpolate(x, size=x4.shape[2:], mode='bilinear')
        print(f"Resized x shape: {x_resized.shape}")

        combined = torch.cat([x_resized, x4], dim=1)  # Concatenate along channel dimension
        print(f"Combined shape: {combined.shape}")

        batch_size, channels, height, width = combined.shape
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        print(f"Combined reshaped shape: {combined_reshaped.shape}")
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)
        print(f"Combined projected shape: {combined_projected.shape}")

        x = self.decoder[0](combined_projected, x3)
        print(f"Decoder output 1 shape: {x.shape}")
        x = self.decoder[1](x, x2)
        print(f"Decoder output 2 shape: {x.shape}")
        x = self.decoder[2](x, x1)
        print(f"Decoder output 3 shape: {x.shape}")
        output = self.decoder[3](x)
        print(f"Final output shape: {output.shape}")
        return output

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
        print("DoubleConv initialized.")

    def forward(self, x):
        print(f"DoubleConv forward pass, input shape: {x.shape}")
        output = self.double_conv(x)
        print(f"DoubleConv forward pass, output shape: {output.shape}")
        return output

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        print(f"Initializing Down ({in_channels} -> {out_channels})...")  # Print initialization info

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        print("Down initialized.")

    def forward(self, x):
        print(f"Down forward pass, input shape: {x.shape}")
        output = self.maxpool_conv(x)
        print(f"Down forward pass, output shape: {output.shape}")

        return output

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        print(f"Initializing Up ({in_channels} -> {out_channels})...")  # Print initialization info
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        print("Up initialized.")

    def forward(self, x1, x2):
        print(f"Up forward pass, x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        x1 = self.up(x1)
        print(f"Up output shape (after transpose conv): {x1.shape}")

        # Calculate the difference in shape and pad x1 if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        print(f"Up output shape (after padding): {x1.shape}")

        x = torch.cat([x2, x1], dim=1)
        print(f"Concatenated shape: {x.shape}")
        output = self.conv(x)
        print(f"Up forward pass, final output shape: {output.shape}")

        return output