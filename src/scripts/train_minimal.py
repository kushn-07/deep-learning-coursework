"""
Minimal training script - no pretrained model downloads
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data.dataset import get_dataloaders
from models.simple_generator import SimpleGenerator
from models.style_encoder import StyleEncoder
from models.discriminator import PatchGANDiscriminator

class SimpleLoss(nn.Module):
    """Simple loss for testing"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, original, generated, discriminator_pred=None):
        # Content preservation
        content_loss = self.l1_loss(original, generated)
        
        # Style/color loss
        style_loss = self.mse_loss(
            torch.mean(original, dim=[2, 3]),
            torch.mean(generated, dim=[2, 3])
        )
        
        # Adversarial loss
        if discriminator_pred is not None:
            adv_loss = self.mse_loss(discriminator_pred, torch.ones_like(discriminator_pred))
        else:
            adv_loss = torch.tensor(0.0)
        
        total_loss = content_loss * 10 + style_loss * 5 + adv_loss * 1
        
        return total_loss, {
            'content': content_loss.item(),
            'style': style_loss.item(),
            'adversarial': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            'total': total_loss.item()
        }

class MinimalTrainer:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Create models
        self.generator = SimpleGenerator(config).to(device)
        self.style_encoder = StyleEncoder(config).to(device)
        self.discriminator = PatchGANDiscriminator(config).to(device)
        
        # Simple loss
        self.criterion = SimpleLoss()
        
        # Get optimizer parameters with defaults
        lr_g = config['training'].get('lr_g', 0.0002)
        lr_d = config['training'].get('lr_d', 0.0002)
        beta1 = config['training'].get('beta1', 0.5)
        beta2 = config['training'].get('beta2', 0.999)
        
        # Optimizers
        self.opt_g = optim.Adam(
            list(self.generator.parameters()) + list(self.style_encoder.parameters()),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # Data loaders
        self.train_loader, self.val_loader = get_dataloaders(config)
        
        # Output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"runs/{self.timestamp}/checkpoints"
        self.sample_dir = f"runs/{self.timestamp}/samples"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Save config
        with open(f"runs/{self.timestamp}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        print(f"Minimal training initialized on {device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"Optimizer settings: lr_g={lr_g}, lr_d={lr_d}, beta1={beta1}, beta2={beta2}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.style_encoder.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            content = batch['content_image'].to(self.device)
            style = batch['style_image'].to(self.device)
            
            # ========== Train Discriminator ==========
            self.opt_d.zero_grad()
            
            # Encode style and generate
            with torch.no_grad():
                style_vector = self.style_encoder(style)
                fake_images = self.generator(content, style_vector, None)
            
            # Discriminator predictions
            fake_pred = self.discriminator(fake_images.detach(), content)
            real_pred = self.discriminator(style, content)
            
            # Discriminator loss
            d_loss_fake = self.criterion.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            d_loss_real = self.criterion.mse_loss(real_pred, torch.ones_like(real_pred))
            d_loss = (d_loss_fake + d_loss_real) * 0.5
            
            d_loss.backward()
            self.opt_d.step()
            
            # ========== Train Generator ==========
            self.opt_g.zero_grad()
            
            # Generate
            fake_images = self.generator(content, style_vector, None)
            
            # Discriminator predictions
            fake_pred = self.discriminator(fake_images, content)
            
            # Generator loss
            g_loss, loss_dict = self.criterion(
                original=content,
                generated=fake_images,
                discriminator_pred=fake_pred
            )
            
            g_loss.backward()
            self.opt_g.step()
            
            # Update progress
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'G Loss': f'{g_loss.item():.4f}',
                    'D Loss': f'{d_loss.item():.4f}',
                    'Content': f'{loss_dict["content"]:.4f}'
                })
        
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        
        return avg_g_loss, avg_d_loss
    
    def validate(self, epoch):
        """Quick validation"""
        self.generator.eval()
        self.style_encoder.eval()
        
        with torch.no_grad():
            # Just check first batch
            batch = next(iter(self.val_loader))
            content = batch['content_image'].to(self.device)[:4]  # First 4
            style = batch['style_image'].to(self.device)[:4]
            
            # Generate
            style_vector = self.style_encoder(style)
            fake_images = self.generator(content, style_vector, None)
            
            # Save sample
            self.save_sample_images(content, style, fake_images, epoch)
            
            # Simple validation loss
            val_loss = self.criterion.l1_loss(content, fake_images)
            
            return val_loss.item()
    
    def save_sample_images(self, content, style, generated, epoch):
        """Save sample images"""
        # Denormalize
        content_norm = (content.cpu() + 1) / 2
        style_norm = (style.cpu() + 1) / 2
        generated_norm = (generated.cpu() + 1) / 2
        
        # Create figure
        n_samples = min(4, content.size(0))
        fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 9))
        
        if n_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(n_samples):
            # Content
            axes[0, i].imshow(content_norm[i].permute(1, 2, 0))
            axes[0, i].set_title(f"Content {i}")
            axes[0, i].axis('off')
            
            # Style
            axes[1, i].imshow(style_norm[i].permute(1, 2, 0))
            axes[1, i].set_title(f"Style {i}")
            axes[1, i].axis('off')
            
            # Generated
            axes[2, i].imshow(generated_norm[i].permute(1, 2, 0))
            axes[2, i].set_title(f"Generated {i}")
            axes[2, i].axis('off')
        
        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.sample_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Sample saved: {save_path}")
    
    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }
        
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch:03d}.pth")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_g_loss, train_d_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train G Loss: {train_g_loss:.4f}")
            print(f"  Train D Loss: {train_d_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config['training'].get('save_interval', 1) == 0:
                self.save_checkpoint(epoch)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        
        # Save final model
        self.save_checkpoint('final')

def main():
    parser = argparse.ArgumentParser(description="Minimal training script")
    parser.add_argument("--config", type=str, default="config_super_minimal.yaml", 
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, default=2, 
                       help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to use (cpu, mps)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Create and run trainer
    trainer = MinimalTrainer(config, device)
    trainer.train(args.epochs)

if __name__ == "__main__":
    main()