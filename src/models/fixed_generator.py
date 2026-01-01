"""
Fixed U-Net Generator that actually works
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedUNetGenerator(nn.Module):
    """Working U-Net generator"""
    def __init__(self, config):
        super().__init__()
        
        gen_config = config['model']['generator']
        base_channels = gen_config['base_channels']
        
        # Encoder
        self.enc1 = self._encoder_block(3, base_channels, normalize=False)  # 64x64 → 64x64
        self.enc2 = self._encoder_block(base_channels, base_channels*2)     # 64x64 → 32x32
        self.enc3 = self._encoder_block(base_channels*2, base_channels*4)   # 32x32 → 16x16
        self.enc4 = self._encoder_block(base_channels*4, base_channels*8)   # 16x16 → 8x8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.InstanceNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.InstanceNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.dec4 = self._decoder_block(base_channels*16, base_channels*4)  # 8x8 → 16x16
        self.dec3 = self._decoder_block(base_channels*8, base_channels*2)   # 16x16 → 32x32
        self.dec2 = self._decoder_block(base_channels*4, base_channels)     # 32x32 → 64x64
        self.dec1 = self._decoder_block(base_channels*2, base_channels)     # 64x64 → 64x64
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1),
            nn.Tanh()
        )
    
    def _encoder_block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels) if normalize else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, style_vector=None, text_embedding=None):
        # Encoder
        e1 = self.enc1(x)      # 64 -> 64
        e2 = self.enc2(e1)     # 64 -> 32
        e3 = self.enc3(e2)     # 32 -> 16
        e4 = self.enc4(e3)     # 16 -> 8
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))  # 8 -> 16
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # 16 -> 32
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # 32 -> 64
        d1 = self.dec1(torch.cat([d2, e1], dim=1)) # 64 -> 64
        
        # Final output
        output = self.final(d1)
        
        return output

def test_fixed_generator():
    """Test the fixed generator"""
    print("Testing Fixed Generator...")
    
    test_config = {
        'model': {
            'generator': {'base_channels': 32}
        }
    }
    
    generator = FixedUNetGenerator(test_config)
    
    # Test with different sizes
    for size in [64, 128]:
        x = torch.randn(2, 3, size, size)
        output = generator(x)
        print(f"Input {size}x{size}: {x.shape} → Output: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch at size {size}"
    
    print(f"\n✅ Fixed generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    return generator

if __name__ == "__main__":
    test_fixed_generator()