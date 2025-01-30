import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels=192, latent_size=32):
        super(UNet_bathy, self).__init__()
        self.n_channels = in_channels
        self.n_outputs = out_channels
        self.latent_channels = latent_channels  
        self.latent_size = latent_size  
        self.fc = nn.Linear(
            in_features=3072,
            out_features=self.latent_channels * self.latent_size * self.latent_size,)  
        self.channel_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.linear_projection = nn.Linear(in_features=3072, out_features=3072)

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

    def forward(self, x, images):
        images = images.float()
        batch_size, channels, features = x.shape
        target_height = 32
        target_width = features // target_height
        print("x.shape:", x.shape)
        x = x.view(batch_size, channels, target_height, target_width)
        x = F.interpolate(x, size=(32, 32), mode="nearest")
        x = self.channel_projection(x)

        x1 = self.encoder[0](images)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)

        combined = torch.cat([x, x4], dim=-1)  # Concatenate along the feature dimension
        x = self.linear(combined)

        x = self.decoder[0](x, x3)
        x = self.decoder[1](x, x2)
        x = self.decoder[2](x, x1)
        output = self.decoder[3](x)
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
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculate the difference in shape and pad x2 if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
