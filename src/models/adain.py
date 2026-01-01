"""
Adaptive Instance Normalization (AdaIN) for style transfer
Core component for one-shot stylization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization
    Normalizes content features and applies style statistics
    """
    def __init__(self, num_features, style_dim=512, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.style_dim = style_dim
        
        # Learnable parameters for style modulation
        self.style_scale = nn.Linear(style_dim, num_features)  # Style vector -> scale
        self.style_shift = nn.Linear(style_dim, num_features)  # Style vector -> shift
        
    def forward(self, content, style_vector):
        """
        Args:
            content: Tensor [B, C, H, W] - content feature maps
            style_vector: Tensor [B, style_dim] - encoded style vector
            
        Returns:
            Tensor [B, C, H, W] - stylized feature maps
        """
        batch_size, num_channels, height, width = content.shape
        
        # Normalize content (Instance Normalization)
        content_reshaped = content.view(batch_size, num_channels, -1)
        content_mean = content_reshaped.mean(dim=2, keepdim=True)
        content_std = content_reshaped.std(dim=2, keepdim=True) + self.eps
        content_normalized = (content_reshaped - content_mean) / content_std
        
        # Reshape back
        content_normalized = content_normalized.view(batch_size, num_channels, height, width)
        
        # Generate scale and shift parameters from style vector
        scale = self.style_scale(style_vector)  # [B, C]
        shift = self.style_shift(style_vector)  # [B, C]
        
        # Reshape for broadcasting
        scale = scale.view(batch_size, num_channels, 1, 1)
        shift = shift.view(batch_size, num_channels, 1, 1)
        
        # Apply style: y = scale * x + shift
        stylized = content_normalized * scale + shift
        
        return stylized

    def extra_repr(self):
        return f'num_features={self.num_features}, style_dim={self.style_dim}, eps={self.eps}'


class AdaptiveResidualBlock(nn.Module):
    """
    Residual block with AdaIN for style transfer
    """
    def __init__(self, channels, style_dim=512, use_adain=True):
        super().__init__()
        self.use_adain = use_adain
        
        # Main convolution path
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels) if not use_adain else None
        self.adain1 = AdaIN(channels, style_dim) if use_adain else None
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels) if not use_adain else None
        self.adain2 = AdaIN(channels, style_dim) if use_adain else None
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, style_vector=None):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        if self.use_adain and style_vector is not None:
            out = self.adain1(out, style_vector)
        elif self.norm1 is not None:
            out = self.norm1(out)
        out = self.relu(out)
        
        # Second convolution
        out = self.conv2(out)
        if self.use_adain and style_vector is not None:
            out = self.adain2(out, style_vector)
        elif self.norm2 is not None:
            out = self.norm2(out)
        
        # Residual connection
        out = out + residual
        out = self.relu(out)
        
        return out


def test_adain():
    """Test AdaIN implementation"""
    print("Testing AdaIN...")
    
    # Test with different style dimensions
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    
    # Test 1: Style dim 512 (original)
    print("\nTest 1: style_dim=512")
    style_dim = 512
    content = torch.randn(batch_size, channels, height, width)
    style_vector = torch.randn(batch_size, style_dim)
    
    adain = AdaIN(channels, style_dim)
    output = adain(content, style_vector)
    print(f"  Input shape: {content.shape}")
    print(f"  Style vector shape: {style_vector.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test 2: Style dim 128 (our test config)
    print("\nTest 2: style_dim=128")
    style_dim = 128
    style_vector = torch.randn(batch_size, style_dim)
    
    adain = AdaIN(channels, style_dim)
    output = adain(content, style_vector)
    print(f"  Input shape: {content.shape}")
    print(f"  Style vector shape: {style_vector.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test AdaptiveResidualBlock
    print("\nTest 3: AdaptiveResidualBlock with style_dim=128")
    res_block = AdaptiveResidualBlock(channels, style_dim=128, use_adain=True)
    output_res = res_block(content, style_vector)
    print(f"  Residual block output shape: {output_res.shape}")
    
    # Verify shapes
    assert output.shape == content.shape, "AdaIN output shape mismatch"
    assert output_res.shape == content.shape, "Residual block output shape mismatch"
    
    print("\n✅ AdaIN tests passed!")
    return adain, res_block


if __name__ == "__main__":
    test_adain()