"""
Generator with configurable architecture for one-shot face stylization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, num_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # Style modulation
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_bias = nn.Linear(style_dim, num_features)
        
        # Initialize
        nn.init.normal_(self.style_scale.weight, mean=1.0, std=0.02)
        nn.init.constant_(self.style_scale.bias, 0.0)
        nn.init.normal_(self.style_bias.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.style_bias.bias, 0.0)
    
    def forward(self, x, style_vector):
        # Normalize
        normalized = self.norm(x)
        
        # Get style parameters
        scale = self.style_scale(style_vector).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style_vector).unsqueeze(2).unsqueeze(3)
        
        # Apply style
        return scale * normalized + bias


class StyleResBlock(nn.Module):
    """Residual block with style injection"""
    def __init__(self, channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.adain1 = AdaIN(channels, style_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.adain2 = AdaIN(channels, style_dim)
        
        # Initialize
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.02)
    
    def forward(self, x, style_vector):
        residual = x
        
        out = self.conv1(x)
        out = self.adain1(out, style_vector)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.adain2(out, style_vector)
        
        return out + residual


class SimpleGenerator(nn.Module):
    """Generator for one-shot face stylization"""
    def __init__(self, config):
        super().__init__()
        
        gen_config = config['model']['generator']
        style_dim = config['model']['style_encoder']['style_dim']
        
        base_channels = gen_config['base_channels']
        num_res_blocks = gen_config['num_residual_blocks']
        num_downsampling = gen_config.get('num_downsampling', 2)
        use_adain = gen_config.get('use_adain', True)
        
        # Encoder layers
        encoder_layers = []
        
        # Initial layer
        encoder_layers.append(nn.Conv2d(3, base_channels, 7, padding=3, padding_mode='reflect'))
        if use_adain:
            encoder_layers.append(AdaIN(base_channels, style_dim))
        encoder_layers.append(nn.ReLU(inplace=True))
        
        # Downsampling layers
        current_channels = base_channels
        for i in range(num_downsampling):
            next_channels = current_channels * 2 if i < num_downsampling - 1 else current_channels * 2
            encoder_layers.append(nn.Conv2d(current_channels, next_channels, 3, stride=2, padding=1))
            if use_adain:
                encoder_layers.append(AdaIN(next_channels, style_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            StyleResBlock(current_channels, style_dim) for _ in range(num_res_blocks)
        ])
        
        # Upsampling layers
        decoder_layers = []
        for i in range(num_downsampling):
            next_channels = current_channels // 2 if i < num_downsampling - 1 else base_channels
            decoder_layers.append(nn.ConvTranspose2d(
                current_channels, next_channels, 3, stride=2, padding=1, output_padding=1
            ))
            if use_adain:
                decoder_layers.append(AdaIN(next_channels, style_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels
        
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Final layer
        self.final_conv = nn.Conv2d(base_channels, 3, 7, padding=3, padding_mode='reflect')
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, content_image, style_vector=None, text_embedding=None):
        if style_vector is None:
            batch_size = content_image.size(0)
            style_dim = self.encoder[1].style_scale.in_features if isinstance(self.encoder[1], AdaIN) else 256
            style_vector = torch.zeros(batch_size, style_dim).to(content_image.device)
        
        # Encode
        x = content_image
        for layer in self.encoder:
            if isinstance(layer, AdaIN):
                x = layer(x, style_vector)
            else:
                x = layer(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x, style_vector)
        
        # Decode
        for layer in self.decoder:
            if isinstance(layer, AdaIN):
                x = layer(x, style_vector)
            else:
                x = layer(x)
        
        # Output
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x


if __name__ == "__main__":
    # Test the generator
    print("Testing Generator...")
    
    config = {
        'model': {
            'generator': {
                'base_channels': 64,
                'num_residual_blocks': 6,
                'num_downsampling': 2,
                'use_adain': True
            },
            'style_encoder': {
                'style_dim': 256
            }
        }
    }
    
    generator = SimpleGenerator(config)
    
    # Test inputs
    batch_size = 2
    content = torch.randn(batch_size, 3, 256, 256)
    style_vector = torch.randn(batch_size, 256)
    
    # Forward pass
    output = generator(content, style_vector)
    
    print(f"\n📊 Generator Test:")
    print(f"  Content shape: {content.shape}")
    print(f"  Style vector shape: {style_vector.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in generator.parameters()):,}")