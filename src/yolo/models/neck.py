import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock

class PANet(nn.Module):
    """
    Path Aggregation Network for feature fusion
    """
    def __init__(self):
        super(PANet, self).__init__()
        
        # Upsampling branch
        self.up_conv1 = ConvBlock(512, 256, 1)
        self.up_conv2 = ConvBlock(256, 256, 3)
        self.up_conv3 = ConvBlock(256, 256, 1)
        
        # For the first route connection
        self.route_conv1 = ConvBlock(512, 256, 1)
        
        # After first concatenation
        self.cat_conv1 = nn.Sequential(
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
        )
        
        # Upsampling branch for second route
        self.up2_conv1 = ConvBlock(256, 128, 1)
        self.up2_conv2 = ConvBlock(128, 128, 3)
        self.up2_conv3 = ConvBlock(128, 128, 1)
        
        # For the second route connection
        self.route_conv2 = ConvBlock(256, 128, 1)
        
        # After second concatenation
        self.cat_conv2 = nn.Sequential(
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),
        )
        
        # Downsampling branches
        self.down_conv1 = ConvBlock(128, 256, 3, stride=2)
        self.down_cat_conv1 = nn.Sequential(
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
        )
        
        self.down_conv2 = ConvBlock(256, 512, 3, stride=2)
        self.down_cat_conv2 = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
        )
    
    def forward(self, route_1, route_2, route_3):
        # First upsample from route_3 to route_2
        up1 = self.up_conv1(route_3)
        up1 = F.interpolate(up1, size=route_2.shape[2:], mode='nearest')
        up1 = self.up_conv2(up1)
        up1 = self.up_conv3(up1)
        
        # Connect with route_2
        route_2_in = self.route_conv1(route_2)
        cat1 = torch.cat([route_2_in, up1], dim=1)
        processed1 = self.cat_conv1(cat1)
        
        # Second upsample from processed1 to route_1
        up2 = self.up2_conv1(processed1)
        up2 = F.interpolate(up2, size=route_1.shape[2:], mode='nearest')
        up2 = self.up2_conv2(up2)
        up2 = self.up2_conv3(up2)
        
        # Connect with route_1
        route_1_in = self.route_conv2(route_1)
        cat2 = torch.cat([route_1_in, up2], dim=1)
        processed2 = self.cat_conv2(cat2)  # Small scale output
        
        # Downsample back to medium scale
        down1 = self.down_conv1(processed2)
        cat_down1 = torch.cat([down1, processed1], dim=1)
        processed_down1 = self.down_cat_conv1(cat_down1)  # Medium scale output
        
        # Downsample back to large scale
        down2 = self.down_conv2(processed_down1)
        cat_down2 = torch.cat([down2, route_3], dim=1)
        processed_down2 = self.down_cat_conv2(cat_down2)  # Large scale output
        
        return processed2, processed_down1, processed_down2