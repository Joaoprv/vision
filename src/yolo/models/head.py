import torch.nn as nn

class YOLOHead(nn.Module):
    """
    YOLO Head for object detection
    """
    def __init__(self, in_channels, num_classes):
        super(YOLOHead, self).__init__()
        
        # Number of anchors per scale (typically 3)
        self.num_anchors = 3
        self.num_classes = num_classes
        
        # Each anchor predicts 5 + num_classes values:
        # (x, y, w, h, objectness, class_probs)
        self.output_size = self.num_anchors * (5 + num_classes)
        
        self.conv = nn.Conv2d(in_channels, self.output_size, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)