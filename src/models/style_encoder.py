"""
Style Encoder for one-shot face stylization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    """
    Encodes artistic style from reference images
    """
    def __init__(self, config):
        super().__init__()
        
        style_config = config['model']['style_encoder']
        base_channels = style_config['base_channels']
        style_dim = style_config['style_dim']
        
        # Feature extraction with instance normalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, base_channels, 7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Style projection
        self.style_projection = nn.Sequential(
            nn.Linear(base_channels * 8, style_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(style_dim * 2, style_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, style_image):
        """
        Args:
            style_image: [B, 3, H, W] in range [-1, 1]
        Returns:
            style_vector: [B, style_dim] in range [-1, 1]
        """
        features = self.features(style_image)
        features = features.view(features.size(0), -1)
        style_vector = self.style_projection(features)
        
        return style_vector


def test_style_encoder():
    """Test style encoder"""
    print("Testing Style Encoder...")
    
    config = {
        'model': {
            'style_encoder': {
                'base_channels': 32,
                'style_dim': 256
            }
        }
    }
    
    encoder = StyleEncoder(config)
    
    batch_size = 2
    image = torch.randn(batch_size, 3, 256, 256)
    style_vec = encoder(image)
    
    print(f"Input shape: {image.shape}")
    print(f"Style vector shape: {style_vec.shape}")
    print(f"Range: [{style_vec.min():.3f}, {style_vec.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    return encoder


if __name__ == "__main__":
    test_style_encoder()