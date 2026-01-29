import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import models
from data.dataset import get_dataloaders
from models.simple_generator import SimpleGenerator
from models.style_encoder import StyleEncoder
from models.discriminator import PatchGANDiscriminator

# ==================== CONFIGURATION ====================
CONFIG_FILE = "config_small.yaml"
EPOCHS = None  # Will use config value
BATCH_SIZE = None
IMAGE_SIZE = None
DEVICE = "mps"
# =======================================================

class GramMatrix(nn.Module):
    """Compute Gram matrix for style loss"""
    def forward(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

class SimpleTrainer:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Seeds
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if device == 'cuda':
            torch.cuda.manual_seed_all(42)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Models
        self.generator = SimpleGenerator(config).to(device)
        self.style_encoder = StyleEncoder(config).to(device)
        self.discriminator = PatchGANDiscriminator(config).to(device)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Style encoder parameters: {sum(p.numel() for p in self.style_encoder.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Initialize feature extractor for style loss
        self.feature_extractor = self._build_feature_extractor().to(device)
        
        # Gram matrix for style loss
        self.gram_matrix = GramMatrix()
        
        # Loss weights from config
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Use config loss weights with sensible defaults
        loss_weights = config['training']['loss_weights']
        self.lambda_content = loss_weights.get('perceptual', 5.0)
        self.lambda_style = loss_weights.get('style', 3.0)
        self.lambda_adv = loss_weights.get('adversarial', 1.0)
        self.lambda_identity = loss_weights.get('identity', 10.0)
        self.lambda_clip = loss_weights.get('clip', 2.0)
        
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
        self.scheduler_g = optim.lr_scheduler.StepLR(self.opt_g, step_size=10, gamma=0.5)
        self.scheduler_d = optim.lr_scheduler.StepLR(self.opt_d, step_size=10, gamma=0.5)
        
        # Data
        self.train_loader, self.val_loader = get_dataloaders(config)
        self.fixed_val_batch = None
        self._get_fixed_val_batch()
        
        # Output
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"runs/{self.timestamp}"
        os.makedirs(f"{self.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.output_dir}/samples", exist_ok=True)
        
        with open(f"{self.output_dir}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # Test forward pass
        self._test_forward_pass()
        
        print(f"\n✅ Training initialized - One-Shot Face Stylization")
        print(f"📁 Output: {self.output_dir}")
        print(f"📊 Train batches: {len(self.train_loader)}")
        print(f"📈 Val batches: {len(self.val_loader)}")
        print(f"⚙️  Loss weights: Content={self.lambda_content}, Style={self.lambda_style}, "
              f"Adv={self.lambda_adv}, Identity={self.lambda_identity}")
    
    def _build_feature_extractor(self):
        """Simple CNN for feature extraction (if VGG is not available)"""
        layers = []
        in_channels = 3
        
        # Block 1
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 2
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 3
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _test_forward_pass(self):
        """Test that forward pass works correctly"""
        print("\n🧪 Testing forward pass...")
        image_size = self.config['data']['image_size']
        dummy_content = torch.randn(1, 3, image_size, image_size).to(self.device)
        dummy_style = torch.randn(1, 3, image_size, image_size).to(self.device)
        
        with torch.no_grad():
            style_vec = self.style_encoder(dummy_style)
            output = self.generator(dummy_content, style_vec)
            
            print(f"  Content shape: {dummy_content.shape}")
            print(f"  Style shape: {dummy_style.shape}")
            print(f"  Style vector shape: {style_vec.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Check for NaNs
            if torch.isnan(style_vec).any():
                print("  ⚠️  WARNING: NaN in style vector!")
            if torch.isnan(output).any():
                print("  ⚠️  WARNING: NaN in output!")
            
            # Test with different style
            different_style = torch.randn(1, 3, image_size, image_size).to(self.device)
            style_vec2 = self.style_encoder(different_style)
            output2 = self.generator(dummy_content, style_vec2)
            diff = (output - output2).abs().mean()
            print(f"  Difference with different styles: {diff:.4f}")
            
            if diff < 0.01:
                print("  ⚠️  WARNING: Different styles produce similar outputs!")
            else:
                print("  ✅ Different styles produce different outputs!")
    
    def _get_fixed_val_batch(self):
        val_iterator = iter(self.val_loader)
        self.fixed_val_batch = next(val_iterator)
        print(f"📊 Stored fixed validation batch with {self.fixed_val_batch['content_image'].shape[0]} images")

    def compute_identity_loss(self, fake, content):
        """
        Preserve facial structure using grayscale
        """
        def to_gray(img):
            r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
            gray = 0.299*r + 0.587*g + 0.114*b
            return gray
        
        fake_gray = to_gray(fake)
        content_gray = to_gray(content)
        
        return self.l1_loss(fake_gray, content_gray)

    def compute_style_loss(self, fake, style):
        """
        Improved style loss using Gram matrices
        """
        # Extract features
        fake_features = self.feature_extractor(fake)
        style_features = self.feature_extractor(style)
        
        # Compute Gram matrices
        gram_fake = self.gram_matrix(fake_features)
        gram_style = self.gram_matrix(style_features)
        
        # Style loss
        style_loss = self.mse_loss(gram_fake, gram_style)
        
        return style_loss

    def compute_content_loss(self, fake, content):
        """Perceptual/content loss"""
        fake_features = self.feature_extractor(fake)
        content_features = self.feature_extractor(content)
        
        return self.l1_loss(fake_features, content_features)

    def train_epoch(self, epoch):
        self.generator.train()
        self.style_encoder.train()
        self.discriminator.train()
        
        total_g_loss, total_d_loss = 0, 0
        total_content, total_identity, total_style, total_adv = 0, 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # Progressive training schedule
        if epoch <= 10:
            # Early: Focus on style transfer and identity preservation
            style_weight = self.lambda_style * 1.2
            identity_weight = self.lambda_identity * 1.0
            content_weight = self.lambda_content * 0.8
            adv_weight = self.lambda_adv * 0.3
        elif epoch <= 30:
            # Mid: Balanced training
            style_weight = self.lambda_style
            identity_weight = self.lambda_identity
            content_weight = self.lambda_content
            adv_weight = self.lambda_adv
        else:
            # Late: Refine realism
            style_weight = self.lambda_style * 0.9
            identity_weight = self.lambda_identity * 0.9
            content_weight = self.lambda_content * 0.9
            adv_weight = self.lambda_adv * 1.5
        
        for batch_idx, batch in enumerate(pbar):
            content = batch['content_image'].to(self.device)
            style = batch['style_image'].to(self.device)
            
            # Debug first batch
            if batch_idx == 0 and epoch == 1:
                print(f"\n🔍 First batch check:")
                print(f"  Content range: [{content.min():.3f}, {content.max():.3f}]")
                print(f"  Style range: [{style.min():.3f}, {style.max():.3f}]")
                print(f"  Batch size: {content.shape[0]}")
            
            # ===== Train Discriminator =====
            self.opt_d.zero_grad()
            
            style_vec = self.style_encoder(style)
            fake = self.generator(content, style_vec, None).detach()
            
            fake_pred = self.discriminator(fake, content)
            real_pred = self.discriminator(style, content)
            
            # Hinge loss
            d_loss_fake = torch.mean(torch.relu(1.0 + fake_pred))
            d_loss_real = torch.mean(torch.relu(1.0 - real_pred))
            d_loss = d_loss_fake + d_loss_real
            
            # Add gradient penalty for stability
            if epoch > 5:
                alpha = torch.rand(content.size(0), 1, 1, 1).to(self.device)
                interpolated = (alpha * content + (1 - alpha) * fake).requires_grad_(True)
                disc_interpolated = self.discriminator(interpolated, content)
                gradients = torch.autograd.grad(
                    outputs=disc_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(disc_interpolated),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                d_loss += 10.0 * gradient_penalty
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.opt_d.step()
            
            # ===== Train Generator =====
            if batch_idx % 2 == 0:  # Train G every other batch
                self.opt_g.zero_grad()
                
                style_vec = self.style_encoder(style)
                fake = self.generator(content, style_vec, None)
                
                fake_pred = self.discriminator(fake, content)
                
                # Compute all losses
                content_loss = self.compute_content_loss(fake, content) * content_weight
                identity_loss = self.compute_identity_loss(fake, content) * identity_weight
                style_loss = self.compute_style_loss(fake, style) * style_weight
                adv_loss = -torch.mean(fake_pred) * adv_weight
                
                g_loss = content_loss + identity_loss + style_loss + adv_loss
                g_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.style_encoder.parameters(), 1.0)
                
                self.opt_g.step()
                
                # Track losses
                total_g_loss += g_loss.item()
                total_content += content_loss.item()
                total_identity += identity_loss.item()
                total_style += style_loss.item()
                total_adv += adv_loss.item()
            
            total_d_loss += d_loss.item()
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'G': f'{g_loss.item() if "g_loss" in locals() else 0:.3f}',
                    'D': f'{d_loss.item():.3f}',
                    'S': f'{style_loss.item() if "style_loss" in locals() else 0:.3f}',
                    'I': f'{identity_loss.item() if "identity_loss" in locals() else 0:.3f}'
                })
        
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        n = len(self.train_loader)
        return (total_g_loss/n, total_d_loss/n, total_content/n, 
                total_identity/n, total_style/n, total_adv/n)
    
    def validate(self, epoch):
        self.generator.eval()
        self.style_encoder.eval()
        
        with torch.no_grad():
            content = self.fixed_val_batch['content_image'].to(self.device)[:4]
            style = self.fixed_val_batch['style_image'].to(self.device)[:4]
            
            if epoch == 1:
                print(f"\n🔍 Validation check:")
                print(f"  Content: [{content.min():.3f}, {content.max():.3f}]")
                print(f"  Style: [{style.min():.3f}, {style.max():.3f}]")
            
            style_vec = self.style_encoder(style)
            fake = self.generator(content, style_vec, None)
            
            if epoch == 1:
                print(f"  Style vec: mean={style_vec.mean():.4f}, std={style_vec.std():.4f}")
                print(f"  Generated: [{fake.min():.3f}, {fake.max():.3f}]")
                print(f"  Generated != Content: {not torch.allclose(fake, content, atol=0.1)}")
            
            self.save_samples(content, style, fake, epoch)
            
            val_identity = self.compute_identity_loss(fake, content)
            val_style = self.compute_style_loss(fake, style)
            val_content = self.compute_content_loss(fake, content)
            val_loss = val_identity + val_style + val_content
            
            return val_loss.item(), val_identity.item(), val_style.item(), val_content.item()
    
    def save_samples(self, content, style, generated, epoch):
        content = (content.cpu() + 1)/2
        style = (style.cpu() + 1)/2
        generated = (generated.cpu() + 1)/2
        
        n = min(4, content.shape[0])
        fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
        if n == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(n):
            axes[0, i].imshow(content[i].permute(1, 2, 0).clip(0, 1))
            axes[0, i].set_title(f'Content {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(style[i].permute(1, 2, 0).clip(0, 1))
            axes[1, i].set_title(f'Style {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(generated[i].permute(1, 2, 0).clip(0, 1))
            axes[2, i].set_title(f'Generated {i+1}')
            axes[2, i].axis('off')
        
        plt.suptitle(f'Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/samples/epoch_{epoch:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.opt_g.state_dict(),
            'optimizer_d': self.opt_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'val_loss': self.best_val_loss,
            'config': self.config,
        }
        save_name = f"epoch_{epoch:03d}.pth" if isinstance(epoch, int) else f"{epoch}.pth"
        save_path = f"{self.output_dir}/checkpoints/{save_name}"
        torch.save(checkpoint, save_path)
        if is_best:
            best_path = f"{self.output_dir}/checkpoints/best.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 BEST MODEL: {save_path} (val_loss: {self.best_val_loss:.4f})")
        else:
            print(f"💾 Checkpoint: {save_path}")
    
    def train(self, num_epochs):
        train_g_losses, train_d_losses, val_losses = [], [], []
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")
            
            g_loss, d_loss, c_loss, i_loss, s_loss, a_loss = self.train_epoch(epoch)
            train_g_losses.append(g_loss)
            train_d_losses.append(d_loss)
            
            # Validate
            if epoch == 1 or epoch % 2 == 0:
                val_loss, val_i, val_s, val_c = self.validate(epoch)
                val_losses.append(val_loss)
            else:
                val_loss = val_losses[-1] if val_losses else 0
                val_i, val_s, val_c = 0, 0, 0
            
            print(f"\n📊 Epoch {epoch}:")
            print(f"  Train G: {g_loss:.4f} (C: {c_loss:.4f}, I: {i_loss:.4f}, S: {s_loss:.4f}, A: {a_loss:.4f})")
            print(f"  Train D: {d_loss:.4f}")
            print(f"  Val: {val_loss:.4f} (I: {val_i:.4f}, S: {val_s:.4f}, C: {val_c:.4f})")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                print(f"🎯 NEW BEST!")
                self.save_checkpoint(epoch, is_best=True)
            else:
                print(f"Best: epoch {self.best_epoch} ({self.best_val_loss:.4f})")
            
            # Save checkpoints
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Log progress
            if epoch % self.config['training']['log_interval'] == 0:
                print(f"\n📈 Progress after {epoch} epochs:")
                print(f"  Learning rate G: {self.scheduler_g.get_last_lr()[0]:.6f}")
                print(f"  Learning rate D: {self.scheduler_d.get_last_lr()[0]:.6f}")
        
        self.plot_training_curves(train_g_losses, train_d_losses, val_losses)
        self.save_checkpoint('final')
    
    def plot_training_curves(self, g_losses, d_losses, val_losses):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(g_losses)+1)
        
        axes[0].plot(epochs, g_losses, 'b-', linewidth=2)
        axes[0].set_title('Generator Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, d_losses, 'r-', linewidth=2)
        axes[1].set_title('Discriminator Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(epochs, val_losses[:len(g_losses)], 'g-', linewidth=2)
        axes[2].set_title('Validation Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    print("\n" + "="*60)
    print("ONE-SHOT FACE STYLIZATION - AdaIN Generator")
    print("="*60)
    print(f"Config: {CONFIG_FILE}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    if BATCH_SIZE is not None:
        config['data']['batch_size'] = BATCH_SIZE
        print(f"Batch size: {BATCH_SIZE}")
    
    if IMAGE_SIZE is not None:
        config['data']['image_size'] = IMAGE_SIZE
        print(f"Image size: {IMAGE_SIZE}")
    
    # Use epochs from config
    epochs = EPOCHS if EPOCHS is not None else config['training']['epochs']
    print(f"Epochs: {epochs}")
    
    if DEVICE == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        print("⚡ Using MPS")
    elif DEVICE == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print("⚡ Using CUDA")
    else:
        device = 'cpu'
        print("⚡ Using CPU")
    
    trainer = SimpleTrainer(config, device)
    trainer.train(epochs)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"Best model: epoch {trainer.best_epoch}")
    print("="*60)


if __name__ == "__main__":
    main()