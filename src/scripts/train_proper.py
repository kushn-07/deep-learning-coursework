"""
Proper training script with all features for MSc project
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import wandb  # Optional: for experiment tracking

# Import our modules
from data.dataset import get_dataloaders
from models.fixed_generator import FixedUNetGenerator
from models.style_encoder import StyleEncoder
from models.discriminator import PatchGANDiscriminator, MultimodalDiscriminator
from models.losses import TotalLoss

class ProperTrainer:
    def __init__(self, config, device='cpu', use_wandb=False):
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize WandB if requested
        if use_wandb:
            wandb.init(project="msc-face-stylization", config=config)
            wandb.run.name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create models
        self.generator = FixedUNetGenerator(config).to(device)
        self.style_encoder = StyleEncoder(config).to(device)
        self.discriminator = MultimodalDiscriminator(config).to(device)
        
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
        
        # Learning rate schedulers
        if config['training']['lr_scheduler']['use_scheduler']:
            self.scheduler_g = StepLR(
                self.opt_g,
                step_size=config['training']['lr_scheduler']['step_size'],
                gamma=config['training']['lr_scheduler']['gamma']
            )
            self.scheduler_d = StepLR(
                self.opt_d,
                step_size=config['training']['lr_scheduler']['step_size'],
                gamma=config['training']['lr_scheduler']['gamma']
            )
        else:
            self.scheduler_g = None
            self.scheduler_d = None
        
        # Data loaders
        self.train_loader, self.val_loader = get_dataloaders(config)
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"runs/{self.timestamp}/checkpoints"
        self.sample_dir = f"runs/{self.timestamp}/samples"
        self.log_dir = f"runs/{self.timestamp}/logs"
        
        for dir_path in [self.checkpoint_dir, self.sample_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save config
        with open(f"runs/{self.timestamp}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.train_history = {
            'g_loss': [], 'd_loss': [], 'val_loss': [],
            'id_sim': [], 'clip_sim': []
        }
        
        print(f"\n{'='*60}")
        print("PROPER TRAINING INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Image size: {config['data']['image_size']}")
        print(f"Batch size: {config['data']['batch_size']}")
        print(f"Epochs: {config['training']['epochs']}")
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Style encoder params: {sum(p.numel() for p in self.style_encoder.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"{'='*60}")
    
    def train_epoch(self, epoch):
        """Train for one epoch with proper loss functions"""
        self.generator.train()
        self.style_encoder.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_id_sim = 0
        total_clip_sim = 0
        
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
            fake_images = self.generator(content, style_vector, text_tokens)
            
            # Discriminator predictions
            fake_pred = self.discriminator(fake_images.detach(), content, text_tokens)
            real_pred = self.discriminator(style, content, text_tokens)
            
            # Discriminator loss
            d_loss_fake = self.criterion.adv_loss(fake_pred, False)
            d_loss_real = self.criterion.adv_loss(real_pred, True)
            d_loss = (d_loss_fake + d_loss_real) * 0.5
            
            # Gradient clipping for discriminator
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            d_loss.backward()
            self.opt_d.step()
            
            # ========== Train Generator ==========
            self.opt_g.zero_grad()
            
            # Generate fake images again
            fake_images = self.generator(content, style_vector, text_tokens)
            
            # Discriminator predictions
            fake_pred = self.discriminator(fake_images, content, text_tokens)
            
            # Generator total loss (ALL losses included)
            g_loss, loss_dict = self.criterion(
                original_faces=content,
                generated_faces=fake_images,
                style_images=style,
                text_prompts=text_prompts,
                discriminator_fake_pred=fake_pred,
                discriminator_real_pred=real_pred
            )
            
            # Gradient clipping for generator
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    list(self.generator.parameters()) + list(self.style_encoder.parameters()),
                    self.config['training']['gradient_clip']
                )
            
            g_loss.backward()
            self.opt_g.step()
            
            # Update metrics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_id_sim += loss_dict.get('identity_similarity', 0).item()
            total_clip_sim += loss_dict.get('clip_similarity', 0).item()
            
            # Update progress bar
            if batch_idx % self.config['training']['log_interval'] == 0:
                pbar.set_postfix({
                    'G': f'{g_loss.item():.4f}',
                    'D': f'{d_loss.item():.4f}',
                    'ID': f'{loss_dict.get("identity_similarity", 0):.3f}',
                    'CLIP': f'{loss_dict.get("clip_similarity", 0):.3f}'
                })
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        'batch_g_loss': g_loss.item(),
                        'batch_d_loss': d_loss.item(),
                        'batch_id_similarity': loss_dict.get('identity_similarity', 0),
                        'batch_clip_similarity': loss_dict.get('clip_similarity', 0),
                        'batch': epoch * len(self.train_loader) + batch_idx
                    })
        
        # Average metrics
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        avg_id_sim = total_id_sim / len(self.train_loader)
        avg_clip_sim = total_clip_sim / len(self.train_loader)
        
        # Update learning rate
        if self.scheduler_g:
            self.scheduler_g.step()
            self.scheduler_d.step()
        
        return avg_g_loss, avg_d_loss, avg_id_sim, avg_clip_sim
    
    def validate(self, epoch):
        """Proper validation with multiple metrics"""
        self.generator.eval()
        self.style_encoder.eval()
        
        val_losses = []
        id_similarities = []
        clip_similarities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx > 10:  # Only validate on first 10 batches for speed
                    break
                
                content = batch['content_image'].to(self.device)
                style = batch['style_image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                text_prompts = batch['text_prompt']
                
                # Encode style and generate
                style_vector = self.style_encoder(style)
                fake_images = self.generator(content, style_vector, text_tokens)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    original_faces=content,
                    generated_faces=fake_images,
                    style_images=style,
                    text_prompts=text_prompts
                )
                
                val_losses.append(loss.item())
                id_similarities.append(loss_dict.get('identity_similarity', 0).item())
                clip_similarities.append(loss_dict.get('clip_similarity', 0).item())
                
                # Save samples from first batch
                if batch_idx == 0:
                    self.save_sample_images(content, style, fake_images, epoch)
        
        avg_val_loss = np.mean(val_losses)
        avg_id_sim = np.mean(id_similarities)
        avg_clip_sim = np.mean(clip_similarities)
        
        return avg_val_loss, avg_id_sim, avg_clip_sim
    
    def save_sample_images(self, content, style, generated, epoch, n_samples=4):
        """Save high-quality sample images"""
        # Denormalize
        content_norm = (content.cpu() + 1) / 2
        style_norm = (style.cpu() + 1) / 2
        generated_norm = (generated.cpu() + 1) / 2
        
        n_samples = min(n_samples, content.size(0))
        
        fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))
        
        if n_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(n_samples):
            # Content
            axes[0, i].imshow(content_norm[i].permute(1, 2, 0))
            axes[0, i].set_title(f"Content {i+1}")
            axes[0, i].axis('off')
            
            # Style
            axes[1, i].imshow(style_norm[i].permute(1, 2, 0))
            axes[1, i].set_title(f"Style {i+1}")
            axes[1, i].axis('off')
            
            # Generated
            axes[2, i].imshow(generated_norm[i].permute(1, 2, 0))
            axes[2, i].set_title(f"Generated {i+1}")
            axes[2, i].axis('off')
        
        plt.suptitle(f"Epoch {epoch} - Validation Samples", fontsize=16)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.sample_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"validation_samples": wandb.Image(save_path)}, step=epoch)
        
        print(f"Sample images saved: {save_path}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save comprehensive checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'style_encoder_state_dict': self.style_encoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Regular checkpoint
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save(checkpoint, save_path)
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        print(f"Checkpoint saved: {save_path}")
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss plots
        epochs = range(1, len(self.train_history['g_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.train_history['g_loss'], 'b-', label='Generator')
        axes[0, 0].plot(epochs, self.train_history['d_loss'], 'r-', label='Discriminator')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, self.train_history['val_loss'], 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].grid(True)
        
        # Similarity plots
        axes[0, 2].plot(epochs, self.train_history['id_sim'], 'b-')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Similarity')
        axes[0, 2].set_title('Identity Similarity (higher is better)')
        axes[0, 2].grid(True)
        axes[0, 2].set_ylim([0, 1])
        
        axes[1, 0].plot(epochs, self.train_history['clip_sim'], 'r-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Similarity')
        axes[1, 0].set_title('CLIP Similarity (higher is better)')
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])
        
        # Learning rate (if scheduler used)
        if self.scheduler_g:
            lr_history = [self.config['training']['lr_g'] * 
                         (self.config['training']['lr_scheduler']['gamma'] ** (i//self.config['training']['lr_scheduler']['step_size']))
                         for i in range(len(epochs))]
            axes[1, 1].plot(epochs, lr_history, 'm-')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Generator Learning Rate Schedule')
            axes[1, 1].grid(True)
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.suptitle('Training History', fontsize=16)
        plt.tight_layout()
        
        # Save
        history_path = os.path.join(self.log_dir, 'training_history.png')
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved: {history_path}")
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({"training_history": wandb.Image(history_path)})
    
    def train(self, num_epochs):
        """Main training loop with early stopping"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_g_loss, train_d_loss, train_id_sim, train_clip_sim = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_id_sim, val_clip_sim = self.validate(epoch)
            
            # Update history
            self.train_history['g_loss'].append(train_g_loss)
            self.train_history['d_loss'].append(train_d_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['id_sim'].append(val_id_sim)
            self.train_history['clip_sim'].append(val_clip_sim)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train G Loss: {train_g_loss:.4f}")
            print(f"  Train D Loss: {train_d_loss:.4f}")
            print(f"  Val Loss:     {val_loss:.4f}")
            print(f"  ID Similarity: {val_id_sim:.3f} (higher is better)")
            print(f"  CLIP Similarity: {val_clip_sim:.3f} (higher is better)")
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_g_loss': train_g_loss,
                    'train_d_loss': train_d_loss,
                    'val_loss': val_loss,
                    'id_similarity': val_id_sim,
                    'clip_similarity': val_clip_sim
                })
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.config['training'].get('min_delta', 0.001):
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                is_best = True
            else:
                self.epochs_no_improve += 1
                is_best = False
            
            # Save checkpoint
            if epoch % self.config['training']['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if self.epochs_no_improve >= self.config['training'].get('patience', 10):
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                print(f"No improvement for {self.epochs_no_improve} epochs.")
                break
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        
        # Plot training history
        self.plot_training_history()
        
        # Save final checkpoint
        self.save_checkpoint('final')
        
        # Close WandB
        if self.use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Proper training for MSc project")
    parser.add_argument("--config", type=str, default="config_proper.yaml",
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases for logging")
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
        print("Using CUDA")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Create and run trainer
    trainer = ProperTrainer(config, device, use_wandb=args.wandb)
    trainer.train(config['training']['epochs'])

if __name__ == "__main__":
    main()