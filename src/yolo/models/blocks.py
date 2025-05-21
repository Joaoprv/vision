import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """
    Mish Activation Function:
    f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvBlock(nn.Module):
    """
    Standard convolution block with BatchNorm and activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation="mish"):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "mish":
            self.activation = Mish()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "linear":
            self.activation = nn.Identity()
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    
class ResidualBlock(nn.Module):
    """
    Residual block used in CSPDarknet53 backbone
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, 1),
            ConvBlock(channels // 2, channels, 3)
        )
    
    def forward(self, x):
        return x + self.block(x)


class CSPBlock(nn.Module):
    """
    Cross Stage Partial block
    """
    def __init__(self, in_channels, out_channels, num_residuals):
        super(CSPBlock, self).__init__()
        
        self.downsample = ConvBlock(in_channels, out_channels, 3, stride=2)
        
        self.part1_conv = ConvBlock(out_channels, out_channels // 2, 1)
        self.part2_conv = ConvBlock(out_channels, out_channels // 2, 1)
        
        self.residuals = nn.Sequential(
            *[ResidualBlock(out_channels // 2) for _ in range(num_residuals)]
        )
        
        self.transition = ConvBlock(out_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.downsample(x)
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)
        
        part1 = self.residuals(part1)
        
        # Concatenate along channel dimension
        combined = torch.cat([part1, part2], dim=1)
        return self.transition(combined)

class SPPBlock(nn.Module):
    """
    Spatial Pyramid Pooling block
    """
    def __init__(self, in_channels, out_channels):
        super(SPPBlock, self).__init__()
        
        mid_channels = in_channels // 2
        
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(mid_channels * 4, out_channels, 1)
        
        self.maxpool_5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool_9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool_13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Apply max pooling with different kernel sizes
        p5 = self.maxpool_5(x)
        p9 = self.maxpool_9(x)
        p13 = self.maxpool_13(x)
        
        # Concatenate along channel dimension
        cat = torch.cat([x, p5, p9, p13], dim=1)
        
        return self.conv2(cat)

