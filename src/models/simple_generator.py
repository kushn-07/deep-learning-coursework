"""
Simplified generator for testing - bypasses U-Net dimension issues
"""
import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    """Simple generator for testing loss functions and training"""
    def __init__(self, config):
        super().__init__()
        
        # Just a few convolution layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Residual blocks
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            
            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def forward(self, content_image, style_vector=None, text_embedding=None):
        # Ignore style and text for now
        return self.model(content_image)