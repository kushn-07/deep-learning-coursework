"""
U-Net Generator with AdaIN for face stylization
"""
import torch
import torch.nn as nn
from .adain import AdaIN, AdaptiveResidualBlock

class UNetEncoder(nn.Module):
    """Encoder part of U-Net"""
    def __init__(self, in_channels=3, base_channels=64, num_downsampling=4):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_downsampling):
            next_channels = min(current_channels * 2, 512)  # Cap at 512
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(next_channels, next_channels, 3, padding=1),
                    nn.InstanceNorm2d(next_channels),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = next_channels
        
        # Store channel info for decoder
        self.channel_sizes = [base_channels]
        current = base_channels
        for i in range(num_downsampling):
            current = min(current * 2, 512)
            self.channel_sizes.append(current)
    
    def forward(self, x):
        """Returns list of feature maps at each level (including input)"""
        features = []
        
        # Initial convolution
        x = self.initial_conv(x)
        features.append(x)
        
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)
            features.append(x)
        
        return features


class UNetDecoder(nn.Module):
    """Decoder part of U-Net with AdaIN"""
    def __init__(self, base_channels=64, num_downsampling=4, num_residual_blocks=6, 
                 style_dim=512, use_adain=True):
        super().__init__()
        self.use_adain = use_adain
        self.style_dim = style_dim
        self.num_downsampling = num_downsampling
        self.base_channels = base_channels  # Store for debugging
        
        # Residual blocks in bottleneck
        bottleneck_channels = min(base_channels * (2 ** num_downsampling), 512)
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(
                AdaptiveResidualBlock(bottleneck_channels, style_dim, use_adain)
            )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        # Calculate channels for each upsampling block
        # Start with bottleneck channels
        current_channels = bottleneck_channels
        
        for i in range(num_downsampling):
            # For upsampling level i:
            # - Input: decoder_features + encoder_features from level -(i+2)
            # - After transpose conv: channels reduce
            
            # Skip connection comes from encoder level -(i+2)
            # For i=0: skip from level -2 (second to last encoder feature)
            # For i=1: skip from level -3, etc.
            
            # The skip connection has channels: base_channels * 2^(num_downsampling - i - 2)
            skip_channels = min(base_channels * (2 ** (num_downsampling - i - 1)), 512)
            
            # Output channels after this block
            if i < num_downsampling - 1:
                out_channels = min(base_channels * (2 ** (num_downsampling - i - 2)), 512)
            else:
                out_channels = base_channels
            
            # Input channels to this block
            in_channels = current_channels + skip_channels
            
            # Create upsampling block
            layers = []
            
            # Transposed convolution for upsampling
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, 
                                  3, stride=2, padding=1, output_padding=1)
            )
            
            # Normalization/AdaIN
            if use_adain:
                layers.append(AdaIN(out_channels, style_dim))
            else:
                layers.append(nn.InstanceNorm2d(out_channels))
            
            layers.append(nn.ReLU(inplace=True))
            
            # Additional convolution
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            
            # Second normalization/AdaIN
            if use_adain:
                layers.append(AdaIN(out_channels, style_dim))
            else:
                layers.append(nn.InstanceNorm2d(out_channels))
            
            layers.append(nn.ReLU(inplace=True))
            
            self.up_blocks.append(nn.Sequential(*layers))
            current_channels = out_channels
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def forward(self, encoder_features, style_vector=None, text_embedding=None):
        """
        Args:
            encoder_features: List of feature maps from encoder
            style_vector: Style embedding [B, style_dim]
            text_embedding: CLIP text embedding [B, text_dim]
        """
        # Start from bottleneck (last encoder feature)
        x = encoder_features[-1]
        
        # Apply residual blocks with style conditioning
        for res_block in self.residual_blocks:
            x = res_block(x, style_vector)
        
        # Upsampling with skip connections
        for i, up_block in enumerate(self.up_blocks):
            # Skip connection from corresponding encoder level
            # We go backwards: -2, -3, -4, ...
            skip_idx = -(i + 2)
            skip_feature = encoder_features[skip_idx]
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_feature], dim=1)
            
            # Upsample
            x = up_block(x)
        
        # Final output
        output = self.final_conv(x)
        return output


class MultimodalGenerator(nn.Module):
    """
    Complete generator with style and text conditioning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract parameters
        gen_config = config['model']['generator']
        base_channels = gen_config['base_channels']
        num_downsampling = gen_config['num_downsampling']
        num_residual_blocks = gen_config['num_residual_blocks']
        use_adain = gen_config.get('use_adain', True)
        
        style_config = config['model']['style_encoder']
        style_dim = style_config['style_dim']
        
        # Components
        self.encoder = UNetEncoder(3, base_channels, num_downsampling)
        self.decoder = UNetDecoder(base_channels, num_downsampling, 
                                  num_residual_blocks, style_dim, use_adain)
        
        # Style projection network (if style and text need to be combined)
        text_dim = config['model']['text_encoder']['embed_dim']
        self.style_projection = nn.Sequential(
            nn.Linear(style_dim + text_dim, style_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim * 2, style_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, content_image, style_vector=None, text_embedding=None):
        """
        Generate stylized image
        
        Args:
            content_image: Input face image [B, 3, H, W]
            style_vector: Style embedding [B, style_dim]
            text_embedding: CLIP text embedding [B, text_dim]
            
        Returns:
            stylized_image: Generated image [B, 3, H, W]
        """
        # Encode content
        encoder_features = self.encoder(content_image)
        
        # Combine style and text embeddings if both provided
        if style_vector is not None and text_embedding is not None:
            # Concatenate and project to combined style vector
            combined = torch.cat([style_vector, text_embedding], dim=1)
            style_vector = self.style_projection(combined)
        elif text_embedding is not None and style_vector is None:
            # Use text as style (text-only mode)
            style_vector = self.style_projection(
                torch.cat([torch.zeros_like(text_embedding), text_embedding], dim=1)
            )
        
        # Decode with style conditioning
        output = self.decoder(encoder_features, style_vector, text_embedding)
        
        return output
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def debug_dimensions(self, image_size=128):
        """Debug function to check all dimensions"""
        print("\n" + "="*60)
        print("Generator Dimension Debug")
        print("="*60)
        
        # Create a dummy input
        dummy_input = torch.randn(1, 3, image_size, image_size)
        print(f"Input shape: {dummy_input.shape}")
        
        # Get encoder features
        encoder_features = self.encoder(dummy_input)
        
        print(f"\nEncoder feature shapes:")
        for i, feat in enumerate(encoder_features):
            print(f"  Level {i}: {feat.shape}")
        
        # Get decoder info
        print(f"\nDecoder info:")
        print(f"  Base channels: {self.decoder.base_channels}")
        print(f"  Style dim: {self.decoder.style_dim}")
        print(f"  Use AdaIN: {self.decoder.use_adain}")
        print(f"  Num downsampling: {self.decoder.num_downsampling}")
        
        # Test forward pass
        style_vec = torch.randn(1, self.decoder.style_dim)
        output = self(dummy_input, style_vec, None)
        print(f"\nOutput shape: {output.shape}")
        
        # Check if output matches input
        if output.shape == dummy_input.shape:
            print("✅ Output shape matches input shape!")
        else:
            print(f"⚠️  Output shape {output.shape} doesn't match input shape {dummy_input.shape}")
        
        print("="*60)


def test_generator():
    """Test generator implementation"""
    print("Testing Generator...")
    
    # Create test config
    test_config = {
        'model': {
            'generator': {
                'base_channels': 32,  # Small for testing
                'num_downsampling': 3,
                'num_residual_blocks': 3,
                'use_adain': True
            },
            'style_encoder': {
                'style_dim': 128
            },
            'text_encoder': {
                'embed_dim': 512
            }
        }
    }
    
    # Create generator
    generator = MultimodalGenerator(test_config)
    
    # Debug dimensions
    generator.debug_dimensions(image_size=128)
    
    # Create test tensors
    batch_size = 2
    image_size = 128
    
    content = torch.randn(batch_size, 3, image_size, image_size)
    style_vector = torch.randn(batch_size, 128)  # style_dim
    text_embedding = torch.randn(batch_size, 512)  # CLIP embedding
    
    print(f"\nGenerator parameters: {generator.get_num_parameters():,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # With both style and text
    output = generator(content, style_vector, text_embedding)
    print(f"Input shape: {content.shape}")
    print(f"Output shape (with style+text): {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}] (should be ~[-1, 1])")
    
    # With style only
    output_style_only = generator(content, style_vector, None)
    print(f"Output shape (style only): {output_style_only.shape}")
    
    # With text only
    output_text_only = generator(content, None, text_embedding)
    print(f"Output shape (text only): {output_text_only.shape}")
    
    # With no conditioning (should still work)
    output_none = generator(content, None, None)
    print(f"Output shape (no conditioning): {output_none.shape}")
    
    # Verify shapes
    assert output.shape == content.shape, "Output shape mismatch"
    assert torch.all(output >= -1.1) and torch.all(output <= 1.1), "Output not in tanh range"
    
    print("\n✅ Generator tests passed!")
    return generator


if __name__ == "__main__":
    test_generator()