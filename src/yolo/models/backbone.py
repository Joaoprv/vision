from torch import nn
from blocks import ConvBlock, CSPBlock


class CSPDarknet53(nn.Module):
    """
    CSPDarknet53 backbone used in YOLOv4
    """
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        
        self.conv1 = ConvBlock(3, 32, 3)
        
        # These match the residual blocks in the original Darknet53
        self.csp1 = CSPBlock(32, 64, 1)
        self.csp2 = CSPBlock(64, 128, 2)
        self.csp3 = CSPBlock(128, 256, 8)  # Returns this layer for the route
        self.csp4 = CSPBlock(256, 512, 8)  # Returns this layer for the route
        self.csp5 = CSPBlock(512, 1024, 4)  # Returns this layer for the route
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.csp1(x)
        x = self.csp2(x)
        
        # Route layers
        route_1 = self.csp3(x)
        route_2 = self.csp4(route_1)
        route_3 = self.csp5(route_2)
        
        return route_1, route_2, route_3