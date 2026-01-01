"""
Main training script for multimodal face stylization
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
from models.simple_generator import SimpleGenerator  # Use simple version for now
from models.style_encoder import StyleEncoder
from models.discriminator import PatchGANDiscriminator
from models.losses import TotalLoss

class Trainer:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Create models
        self.generator = SimpleGenerator(config).to(device)
        self.style_encoder = StyleEncoder(config).to(device)
        self.discriminator = PatchGANDiscriminator(config).to(device)
        
        # Loss function
        self.criterion = TotalLoss(config, device)
        
        # Optimizers
        self.opt_g = optim.Adam(
            list(self.generator.parameters()) + list(self.style_encoder.parameters()),
            lr=config['training']['lr_g'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['training']['lr_d'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Data loaders
        self.train_loader, self.val_loader = get_dataloaders(config)
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"runs/{self.timestamp}/checkpoints"
        self.sample_dir = f"runs/{self.timestamp}/samples"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Save config
        with open(f"runs/{self.timestamp}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        print(f"Training initialized on {device}")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Samples will be saved to: {self.sample_dir}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.style_encoder.train()
        self.discriminator.train()
        
        total_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            content = batch['content_image'].to(self.device)
            style = batch['style_image'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_prompts = batch['text_prompt']
            
            # ========== Train Discriminator ==========
            self.opt_d.zero_grad()
            
            # Encode style
            with torch.no_grad():
                style_vector = self.style_encoder(style)
            
            # Generate fake images
            fake_images = self.generator(content, style_vector, None)  # No text for now
            
            # Discriminator predictions
            fake_pred = self.discriminator(fake_images.detach(), content)
            real_pred = self.discriminator(style, content)  # Using style as "real" target
            
            # Discriminator loss
            d_loss_fake = self.criterion.adv_loss(fake_pred, False)
            d_loss_real = self.criterion.adv_loss(real_pred, True)
            d_loss = (d_loss_fake + d_loss_real) * 0.5
            
            d_loss.backward()
            self.opt_d.step()
            
            # ========== Train Generator ==========
            self.opt_g.zero_grad()
            
            # Generate fake images again
            fake_images = self.generator(content, style_vector, None)
            
            # Discriminator predictions on fake images
            fake_pred = self.discriminator(fake_images, content)
            
            # Generator total loss
            g_loss, loss_dict = self.criterion(
                original_faces=content,
                generated_faces=fake_images,
                style_images=style,
                text_prompts=text_prompts,
                discriminator_fake_pred=fake_pred
            )
            
            g_loss.backward()
            self.opt_g.step()
            
            # Update progress bar
            total_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'G Loss': f'{g_loss.item():.4f}',
                    'D Loss': f'{d_loss.item():.4f}',
                    'ID Sim': f'{loss_dict.get("identity_similarity", 0):.3f}'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        
        return avg_loss, avg_d_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.generator.eval()
        self.style_encoder.eval()
        
        val_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                content = batch['content_image'].to(self.device)
                style = batch['style_image'].to(self.device)
                text_prompts = batch['text_prompt']
                
                # Encode style and generate
                style_vector = self.style_encoder(style)
                fake_images = self.generator(content, style_vector, None)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    original_faces=content,
                    generated_faces=fake_images,
                    style_images=style,
                    text_prompts=text_prompts
                )
                
                val_loss += loss.item() * content.size(0)
                num_samples += content.size(0)
                
                # Save sample images from first batch
                if num_samples <= 4:
                    self.save_sample_images(content, style, fake_images, epoch, num_samples)
        
        avg_val_loss = val_loss / num_samples
        return avg_val_loss
    
    def save_sample_images(self, content, style, generated, epoch, batch_num):
        """Save sample images for visualization"""
        # Denormalize from [-1, 1] to [0, 1]
        content_norm = (content.cpu() + 1) / 2
        style_norm = (style.cpu() + 1) / 2
        generated_norm = (generated.cpu() + 1) / 2
        
        # Create figure
        fig, axes = plt.subplots(3, min(4, content.size(0)), figsize=(12, 9))
        
        for i in range(min(4, content.size(0))):
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
        
        plt.suptitle(f"Epoch {epoch} - Batch {batch_num}")
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.sample_dir, f"epoch_{epoch:03d}_batch_{batch_num}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }
        
        # Handle both integer and string epoch names
        if isinstance(epoch, int):
            save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch:03d}.pth")
        else:
            save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pth")
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_d_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train D Loss: {train_d_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        
        # Save final model
        self.save_checkpoint('final')
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
    
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.sample_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train multimodal face stylization model")
    parser.add_argument("--config", type=str, default="config_minimal.yaml", 
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to use (cpu, cuda, mps)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Start training
    trainer.train(config['training']['epochs'])


if __name__ == "__main__":
    main()