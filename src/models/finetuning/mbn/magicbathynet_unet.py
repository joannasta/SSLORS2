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

class Up(nn.Module):
    def __init__(self, in_channels_from_prev_layer, in_channels_skip_connection, out_channels_for_this_block):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels_from_prev_layer, out_channels_for_this_block, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels_for_this_block + in_channels_skip_connection, out_channels_for_this_block)
        
    def forward(self, x1, x2): 
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels,model_type="mae"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.embedding_projector = nn.Linear(768, 256)
        self.combined_projection = nn.Linear(256 + 256, 256) 
        self.moco_projection =  nn.Linear(in_features=512, out_features=262144)
        self.mocogeo_projection =  nn.Linear(in_features=512, out_features=262144)


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
        #print(f"Initial images shape: {images.shape}")
        #print(f"Initial x_embedding shape: {x_embedding.shape}")

        images = images.float()
        
        x1 = self.encoder[0](images)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3) 

        if self.model_type == "mae":
            projected_embedding = self.embedding_projector(x_embedding) 
            patch_tokens = projected_embedding[:, 1:, :] 
            batch_size, num_patches, features = patch_tokens.shape 
            spatial_size = int(num_patches**0.5) 

            spatial_embedding = patch_tokens.permute(0, 2, 1).reshape(
                batch_size, features, spatial_size, spatial_size
            )

            x_embedding = F.interpolate(
                spatial_embedding, 
                size=(x4.shape[2], x4.shape[3]), 
                mode='bilinear', 
                align_corners=False
            ) 

        elif self.model_type == "moco":
            x_embedding = x_embedding.squeeze().unsqueeze(0)
            x_embedding = self.moco_projection(x_embedding)
            x_embedding = x_embedding.view(x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3])
            combined_projected = torch.cat([x_embedding, x4], dim=1) 
        
        elif self.model_type == "mocogeo":
            print("x_embedding shape before projection:", x_embedding.shape)
            x_embedding = x_embedding.squeeze().unsqueeze(0)
            print("x_embedding shape after unsqueeze:", x_embedding.shape)
            print("x4 shape:", x4.shape)
            x_embedding = self.moco_projection(x_embedding)
            x_embedding = x_embedding.view(x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3])
            
        combined = torch.cat([x_embedding, x4], dim=1) 
        batch_size, channels_combined, height, width = combined.shape
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels_combined)
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)
            
        # Pass combined_projected as x1 (input to upsample) and x3 as x2 (skip connection)
        x = self.decoder[0](combined_projected, x3)
        x = self.decoder[1](x, x2)
        x = self.decoder[2](x, x1)
        output = self.decoder[3](x)
        return output