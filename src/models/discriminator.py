"""
PatchGAN Discriminator for face stylization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies local image patches as real/fake
    Can be conditioned on content image and/or text
    """
    def __init__(self, config):
        super().__init__()
        
        disc_config = config['model']['discriminator']
        base_channels = disc_config['base_channels']
        num_layers = disc_config['num_layers']
        use_sn = disc_config.get('use_sn', True)  # Spectral normalization
        
        # Whether to use conditional discrimination
        self.conditional = config['model'].get('conditional_discriminator', True)
        
        # Input processing
        if self.conditional:
            # If conditional, input is concatenated [content, generated]
            in_channels = 6  # 3 for content + 3 for generated
        else:
            in_channels = 3  # Just the generated image
        
        layers = []
        
        # First layer
        layers.append(self._conv_block(in_channels, base_channels, use_sn, stride=2))
        
        # Intermediate layers
        current_channels = base_channels
        for i in range(1, num_layers):
            next_channels = min(current_channels * 2, 512)
            layers.append(self._conv_block(current_channels, next_channels, use_sn, stride=2))
            current_channels = next_channels
        
        # Final layer
        layers.append(
            nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _conv_block(self, in_channels, out_channels, use_sn, stride=2):
        """Create a convolutional block"""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                        stride=stride, padding=1)
        
        if use_sn:
            conv = nn.utils.spectral_norm(conv)
        
        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, content_image=None):
        """
        Classify image patches as real/fake
        
        Args:
            image: Tensor [B, 3, H, W] - image to classify
            content_image: Tensor [B, 3, H, W] - conditioning image (optional)
            
        Returns:
            Tensor [B, 1, H', W'] - patch-wise real/fake predictions
        """
        if self.conditional and content_image is not None:
            # Concatenate along channel dimension
            x = torch.cat([content_image, image], dim=1)
        else:
            x = image
        
        return self.model(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultimodalDiscriminator(nn.Module):
    """
    Discriminator with text conditioning (for multimodal training)
    """
    def __init__(self, config):
        super().__init__()
        
        # Base PatchGAN
        self.patch_disc = PatchGANDiscriminator(config)
        
        # Text conditioning projection
        text_dim = config['model']['text_encoder']['embed_dim']
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )
        
        # Project text to spatial dimensions for concatenation
        self.text_to_spatial = nn.Conv2d(256, 64, 1)
    
    def forward(self, image, content_image=None, text_embedding=None):
        """
        Args:
            image: Image to classify [B, 3, H, W]
            content_image: Conditioning image [B, 3, H, W]
            text_embedding: CLIP text embedding [B, text_dim]
        """
        batch_size, _, height, width = image.shape
        
        if text_embedding is not None:
            # Project text embedding
            text_features = self.text_projection(text_embedding)  # [B, 256]
            
            # Reshape to spatial and repeat
            text_features = text_features.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
            text_spatial = self.text_to_spatial(text_features)  # [B, 64, 1, 1]
            text_spatial = F.interpolate(text_spatial, size=(height, width), 
                                        mode='bilinear', align_corners=False)
            
            # Concatenate with image
            image_with_text = torch.cat([image, text_spatial], dim=1)
        else:
            image_with_text = image
        
        # Pass through PatchGAN
        return self.patch_disc(image_with_text, content_image)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_discriminator():
    """Test discriminator"""
    print("Testing Discriminator...")
    
    test_config = {
        'model': {
            'discriminator': {
                'base_channels': 32,
                'num_layers': 3,
                'use_sn': True
            },
            'text_encoder': {
                'embed_dim': 512
            },
            'conditional_discriminator': True
        }
    }
    
    # Test PatchGAN
    patch_disc = PatchGANDiscriminator(test_config)
    
    batch_size = 2
    image_size = 256
    
    # Test images
    fake_image = torch.randn(batch_size, 3, image_size, image_size)
    real_image = torch.randn(batch_size, 3, image_size, image_size)
    
    # Conditional forward
    output_cond = patch_disc(fake_image, real_image)
    print(f"Conditional PatchGAN:")
    print(f"  Input image shape: {fake_image.shape}")
    print(f"  Content image shape: {real_image.shape}")
    print(f"  Output shape: {output_cond.shape}")
    print(f"  Output range: [{output_cond.min():.3f}, {output_cond.max():.3f}]")
    
    # Non-conditional forward
    output_noncond = patch_disc(fake_image, None)
    print(f"\nNon-conditional PatchGAN:")
    print(f"  Output shape: {output_noncond.shape}")
    
    # Test multimodal discriminator
    multi_disc = MultimodalDiscriminator(test_config)
    text_embedding = torch.randn(batch_size, 512)
    
    output_multi = multi_disc(fake_image, real_image, text_embedding)
    print(f"\nMultimodal Discriminator:")
    print(f"  Text embedding shape: {text_embedding.shape}")
    print(f"  Output shape: {output_multi.shape}")
    
    print(f"\nParameters:")
    print(f"  PatchGAN: {patch_disc.get_num_parameters():,}")
    print(f"  Multimodal: {multi_disc.get_num_parameters():,}")
    
    print("\n✅ Discriminator tests passed!")
    return patch_disc, multi_disc


if __name__ == "__main__":
    test_discriminator()