"""
Style Encoder for extracting style from a single reference image
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    """
    Encodes style information from a single reference image
    Should extract texture, color, brush strokes but NOT identity
    """
    def __init__(self, config):
        super().__init__()
        
        style_config = config['model']['style_encoder']
        base_channels = style_config['base_channels']
        style_dim = style_config['style_dim']
        
        # Feature extraction CNN
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, base_channels, 7, stride=2, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling and projection to style vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.style_projection = nn.Sequential(
            nn.Linear(base_channels * 8, style_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(style_dim * 2, style_dim),
            nn.Tanh()  # Constrain to [-1, 1]
        )
        
        # Initialize weights
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
        Extract style vector from style image
        
        Args:
            style_image: Tensor [B, 3, H, W]
            
        Returns:
            style_vector: Tensor [B, style_dim]
        """
        # Extract features
        features = self.features(style_image)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Project to style vector
        style_vector = self.style_projection(pooled)
        
        return style_vector
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_style_encoder():
    """Test style encoder"""
    print("Testing Style Encoder...")
    
    test_config = {
        'model': {
            'style_encoder': {
                'base_channels': 32,
                'style_dim': 128
            }
        }
    }
    
    encoder = StyleEncoder(test_config)
    
    # Test input
    batch_size = 2
    image_size = 256
    style_image = torch.randn(batch_size, 3, image_size, image_size)
    
    # Forward pass
    style_vector = encoder(style_image)
    
    print(f"Input shape: {style_image.shape}")
    print(f"Style vector shape: {style_vector.shape}")
    print(f"Style vector range: [{style_vector.min():.3f}, {style_vector.max():.3f}]")
    print(f"Parameters: {encoder.get_num_parameters():,}")
    
    assert style_vector.shape == (batch_size, 128), "Wrong style vector shape"
    assert torch.all(style_vector >= -1.1) and torch.all(style_vector <= 1.1), "Style vector not in tanh range"
    
    print("\n✅ Style encoder tests passed!")
    return encoder


if __name__ == "__main__":
    test_style_encoder()