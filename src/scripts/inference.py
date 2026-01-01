"""
Inference script for generating stylized faces
"""
import torch
import torch.nn.functional as F
import argparse
import yaml
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.fixed_generator import FixedUNetGenerator
from models.style_encoder import StyleEncoder
from models.discriminator import PatchGANDiscriminator
import clip

class FaceStylizer:
    """Class for loading and running inference with the trained model"""
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        
        # Create models
        self.generator = FixedUNetGenerator(self.config).to(device)
        self.style_encoder = StyleEncoder(self.config).to(device)
        
        # Load CLIP for text encoding if available
        try:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            self.has_clip = True
        except:
            print("Warning: CLIP not available, text prompts disabled")
            self.has_clip = False
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.generator.eval()
        self.style_encoder.eval()
        
        # Define transforms
        self.img_size = self.config['data']['image_size']
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"FaceStylizer initialized on {device}")
        print(f"Image size: {self.img_size}x{self.img_size}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.style_encoder.load_state_dict(checkpoint['style_encoder_state_dict'])
        elif 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
            self.style_encoder.load_state_dict(checkpoint['style_encoder'])
        else:
            # Try direct loading
            self.generator.load_state_dict(checkpoint)
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor, image
    
    def encode_style_from_image(self, style_image_path):
        """Extract style vector from a style image"""
        style_tensor, style_image = self.load_image(style_image_path)
        with torch.no_grad():
            style_vector = self.style_encoder(style_tensor)
        return style_vector, style_tensor, style_image
    
    def encode_style_from_text(self, text_prompt):
        """Create style vector from text prompt using CLIP"""
        if not self.has_clip:
            raise ValueError("CLIP not available for text encoding")
        
        # Tokenize text
        text_tokens = clip.tokenize([text_prompt]).to(self.device)
        
        # Get CLIP text embedding
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Normalize and use as style vector
        style_vector = text_features / text_features.norm(dim=1, keepdim=True)
        
        return style_vector
    
    def stylize(self, content_image_path, style_source, style_source_type='image'):
        """
        Generate stylized image
        
        Args:
            content_image_path: Path to content (face) image
            style_source: Path to style image OR text prompt
            style_source_type: 'image' or 'text'
        """
        # Load content image
        content_tensor, content_image = self.load_image(content_image_path)
        
        # Get style vector
        if style_source_type == 'image':
            style_vector, style_tensor, style_image = self.encode_style_from_image(style_source)
            text_embedding = None
        else:  # 'text'
            style_vector = self.encode_style_from_text(style_source)
            style_tensor = None
            style_image = None
            text_embedding = style_vector  # Use as both style and text
        
        # Generate stylized image
        with torch.no_grad():
            generated = self.generator(content_tensor, style_vector, text_embedding)
        
        # Convert tensors to PIL Images for display
        def tensor_to_image(tensor):
            """Convert tensor [-1, 1] to PIL Image [0, 255]"""
            tensor = tensor.squeeze(0).cpu()
            tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
            return transforms.ToPILImage()(tensor)
        
        content_display = tensor_to_image(content_tensor)
        generated_display = tensor_to_image(generated)
        
        if style_image:
            style_display = tensor_to_image(style_tensor)
        else:
            style_display = None
        
        return {
            'content': content_display,
            'style': style_display,
            'generated': generated_display,
            'style_vector': style_vector,
            'style_text': style_source if style_source_type == 'text' else None
        }
    
    def stylize_batch(self, content_images, style_images):
        """Stylize a batch of images"""
        # Load and preprocess
        content_tensors = []
        style_tensors = []
        
        for content_path, style_path in zip(content_images, style_images):
            content_tensor, _ = self.load_image(content_path)
            style_tensor, _ = self.load_image(style_path)
            content_tensors.append(content_tensor)
            style_tensors.append(style_tensor)
        
        content_tensor = torch.cat(content_tensors, dim=0)
        style_tensor = torch.cat(style_tensors, dim=0)
        
        # Encode styles and generate
        with torch.no_grad():
            style_vectors = self.style_encoder(style_tensor)
            generated = self.generator(content_tensor, style_vectors, None)
        
        return generated
    
    def save_results(self, results, output_dir='outputs'):
        """Save inference results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual images
        results['content'].save(os.path.join(output_dir, 'content.png'))
        results['generated'].save(os.path.join(output_dir, 'generated.png'))
        
        if results['style']:
            results['style'].save(os.path.join(output_dir, 'style.png'))
        
        # Create comparison image
        fig, axes = plt.subplots(1, 3 if results['style'] else 2, figsize=(12, 4))
        
        axes[0].imshow(results['content'])
        axes[0].set_title('Content Face')
        axes[0].axis('off')
        
        if results['style']:
            axes[1].imshow(results['style'])
            axes[1].set_title('Style Reference')
            axes[1].axis('off')
            
            axes[2].imshow(results['generated'])
            axes[2].set_title('Generated Stylization')
            axes[2].axis('off')
        else:
            axes[1].imshow(results['generated'])
            axes[1].set_title(f'Generated: {results["style_text"][:30]}...')
            axes[1].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to {output_dir}/")
        return comparison_path

def main():
    parser = argparse.ArgumentParser(description="Generate stylized faces")
    parser.add_argument("--content", type=str, required=True,
                       help="Path to content face image")
    parser.add_argument("--style", type=str, required=True,
                       help="Path to style image OR text prompt")
    parser.add_argument("--style_type", type=str, choices=['image', 'text'], default='image',
                       help="Type of style source: 'image' or 'text'")
    parser.add_argument("--config", type=str, default="config_proper.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                       help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Initialize stylizer
    stylizer = FaceStylizer(args.config, args.checkpoint, device)
    
    # Generate stylized image
    print(f"\nGenerating stylization...")
    print(f"Content: {args.content}")
    print(f"Style ({args.style_type}): {args.style}")
    
    results = stylizer.stylize(
        content_image_path=args.content,
        style_source=args.style,
        style_source_type=args.style_type
    )
    
    # Save results
    comparison_path = stylizer.save_results(results, args.output_dir)
    
    print(f"\n✅ Inference completed!")
    print(f"Comparison saved: {comparison_path}")
    print(f"\nTo view results:")
    print(f"  open {comparison_path}")

if __name__ == "__main__":
    main()