import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels=192, latent_size=32, dropout_rate=0.0, embedding_dim=128):
        super(UNet_bathy, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256)
        )

        self.embedding_to_spatial_map = nn.Linear(self.embedding_dim, self.latent_size * self.latent_size)
        self.channel_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.combined_projection = nn.Linear(256 + 1, 256) # 256 (from x4) + 1 (from embedding map)

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        self.decoder = nn.Sequential(
            Up(256, 128),
            Up(128, 64),
            Up(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x, images):
        print(f"--- Start of UNet_bathy Forward Pass ---")
        print(f"Input 'x' (embedding) shape: {x.shape}")
        print(f"Input 'images' shape: {images.shape}")

        images = images.float()
        print(f"Images after float() conversion shape: {images.shape}")

        x1 = self.encoder[0](images)
        print(f"Shape after encoder[0] (DoubleConv): {x1.shape}")
        x2 = self.encoder[1](x1)
        print(f"Shape after encoder[1] (Down): {x2.shape}")
        x3 = self.encoder[2](x2)
        print(f"Shape after encoder[2] (Down): {x3.shape}")
        x4 = self.encoder[3](x3)
        print(f"Shape after encoder[3] (Down): {x4.shape}")

        # x is the embedding vector
        if x.dim() == 3 and x.shape[1] == 1:
            x_squeezed = x.squeeze(1)
            print(f"Shape of 'x' after squeezing dimension 1: {x_squeezed.shape}")
        else:
            x_squeezed = x
            print(f"Shape of 'x' (no squeeze applied): {x_squeezed.shape}")

        x_spatial_map_flat = self.embedding_to_spatial_map(x_squeezed)
        print(f"Shape after embedding_to_spatial_map (flat): {x_spatial_map_flat.shape}")
        x_spatial_map = x_spatial_map_flat.view(-1, 1, self.latent_size, self.latent_size)
        print(f"Shape of x_spatial_map after initial view: {x_spatial_map.shape}")

        # --- NEW CODE HERE to resize x_spatial_map ---
        # Get target dimensions from x4
        target_height, target_width = x4.shape[2], x4.shape[3]
        x_spatial_map_resized = F.interpolate(x_spatial_map, size=(target_height, target_width), mode='bilinear', align_corners=False)
        print(f"Shape of x_spatial_map after resize to match x4: {x_spatial_map_resized.shape}")
        # --- END NEW CODE ---

        # Concatenate the embedding map directly with x4
        # Use the resized x_spatial_map
        print(f"Shape of x_spatial_map_resized before cat: {x_spatial_map_resized.shape}")
        print(f"Shape of x4 before cat: {x4.shape}")
        combined = torch.cat([x_spatial_map_resized, x4], dim=1) # Use x_spatial_map_resized here
        print(f"Shape after concatenation (combined): {combined.shape}")

        batch_size, channels, height, width = combined.shape
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        print(f"Shape after permute and reshape for combined_projection: {combined_reshaped.shape}")
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)
        print(f"Shape after combined_projection and reshape: {combined_projected.shape}")

        #keep this line
        x = self.decoder[0](combined_projected, x3)
        print(f"Shape after decoder[0] (Up): {x.shape}")
        x = self.decoder[1](x, x2)
        print(f"Shape after decoder[1] (Up): {x.shape}")
        x = self.decoder[2](x, x1)
        print(f"Shape after decoder[2] (Up): {x.shape}")
        output = self.decoder[3](x)
        print(f"Final output shape after decoder[3] (Conv2d): {output.shape}")
        print(f"--- End of UNet_bathy Forward Pass ---")
        return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(f"  DoubleConv input shape: {x.shape}") # Uncomment for deeper inspection
        output = self.double_conv(x)
        # print(f"  DoubleConv output shape: {output.shape}") # Uncomment for deeper inspection
        return output

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # print(f"  Down input shape: {x.shape}") # Uncomment for deeper inspection
        output = self.maxpool_conv(x)
        # print(f"  Down output shape: {output.shape}") # Uncomment for deeper inspection
        return output

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels) # Corrected input channels for DoubleConv


    def forward(self, x1, x2):
        # print(f"  Up input x1 shape (from previous layer): {x1.shape}") # Uncomment for deeper inspection
        # print(f"  Up input x2 shape (skip connection): {x2.shape}") # Uncomment for deeper inspection
        x1 = self.up(x1)
        # print(f"  Up x1 shape after ConvTranspose2d: {x1.shape}")

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # print(f"  Up padding diffY: {diffY}, diffX: {diffX}")
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # print(f"  Up x1 shape after padding: {x1.shape}")
        
        x = torch.cat([x2, x1], dim=1)
        # print(f"  Up concatenated shape: {x.shape}")
        output = self.conv(x)
        # print(f"  Up output shape: {output.shape}")
        return output