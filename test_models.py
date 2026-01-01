"""
Test all model components
"""
import sys
sys.path.append('src')

print("Testing all model components...\n")

import torch

# Test AdaIN
from models.adain import test_adain
adain, res_block = test_adain()

# Test Generator
test_config = {
    'model': {
        'generator': {
            'base_channels': 32,
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

from models.generator import MultimodalGenerator
generator = MultimodalGenerator(test_config)

# Debug dimensions first
print("\n" + "="*60)
print("DEBUGGING GENERATOR DIMENSIONS")
print("="*60)
generator.debug_dimensions(image_size=128)

# Create test tensors
batch_size = 2
image_size = 128
content = torch.randn(batch_size, 3, image_size, image_size)
style_vector = torch.randn(batch_size, 128)
text_embedding = torch.randn(batch_size, 512)

print(f"\nGenerator parameters: {generator.get_num_parameters():,}")

# Test forward passes
print("\nTesting forward passes...")
try:
    output = generator(content, style_vector, text_embedding)
    print(f"✅ Generator output shape: {output.shape}")
    
    output_style = generator(content, style_vector, None)
    print(f"✅ Style-only output shape: {output_style.shape}")
    
    output_text = generator(content, None, text_embedding)
    print(f"✅ Text-only output shape: {output_text.shape}")
    
    output_none = generator(content, None, None)
    print(f"✅ No conditioning output shape: {output_none.shape}")
    
except Exception as e:
    print(f"❌ Generator failed: {e}")
    import traceback
    traceback.print_exc()

# Test Style Encoder
from models.style_encoder import StyleEncoder
style_encoder = StyleEncoder(test_config)
try:
    style_vec = style_encoder(torch.randn(batch_size, 3, 256, 256))
    print(f"\n✅ Style encoder output shape: {style_vec.shape}")
except Exception as e:
    print(f"❌ Style encoder failed: {e}")

# Test Discriminator
from models.discriminator import PatchGANDiscriminator, MultimodalDiscriminator
test_config['model']['conditional_discriminator'] = True

try:
    patch_disc = PatchGANDiscriminator(test_config)
    multi_disc = MultimodalDiscriminator(test_config)

    fake_image = torch.randn(batch_size, 3, 256, 256)
    real_image = torch.randn(batch_size, 3, 256, 256)

    output_patch = patch_disc(fake_image, real_image)
    output_multi = multi_disc(fake_image, real_image, text_embedding)

    print(f"\n✅ PatchGAN output shape: {output_patch.shape}")
    print(f"✅ Multimodal discriminator output shape: {output_multi.shape}")
    
except Exception as e:
    print(f"❌ Discriminator failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("MODEL TEST SUMMARY")
print("="*60)

# Print model sizes if they were created successfully
if 'generator' in locals():
    print(f"\nModel Parameter Counts:")
    print(f"Generator: {generator.get_num_parameters():,}")
if 'style_encoder' in locals():
    print(f"Style Encoder: {style_encoder.get_num_parameters():,}")
if 'patch_disc' in locals():
    print(f"PatchGAN Discriminator: {patch_disc.get_num_parameters():,}")
if 'multi_disc' in locals():
    print(f"Multimodal Discriminator: {multi_disc.get_num_parameters():,}")

print("\n" + "="*60)
print("✅ Ready to proceed with loss functions and training!")
print("="*60)