"""
Face Stylization with StyleGAN - VS Code Single File Version
Run this script locally on your machine with GPU support
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset
    image_size = 256
    batch_size = 8  # Reduced for local training
    epochs = 100    # Reduced for faster training
    learning_rate = 0.0002
    
    # Model
    generator_channels = 64
    style_dim = 256
    num_residual_blocks = 6
    
    # Training
    lambda_content = 10.0
    lambda_adv = 1.0
    lambda_style = 1.0
    lambda_identity = 5.0
    
    # Dataset paths - UPDATE THESE!
    celeba_path = "./data/processed/celeba_small"
    wikiart_path = "./data/processed/wikiart_small"
    
    # Training settings
    save_interval = 10
    warmup_epochs = 10
    num_workers = 4  # For DataLoader
    
    # Output
    output_dir = "./training_output"

# ============================================================================
# OPENCV TRANSFORMS
# ============================================================================

class OpenCVTransforms:
    """Image transformations using OpenCV"""
    
    @staticmethod
    def resize(img, size):
        """Resize image"""
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def to_tensor(img):
        """Convert BGR numpy array to RGB tensor"""
        # Convert BGR to RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        arr = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor
    
    @staticmethod
    def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        """Normalize tensor"""
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    
    @staticmethod
    def random_horizontal_flip(img, p=0.5):
        """Random horizontal flip"""
        if random.random() < p:
            return cv2.flip(img, 1)  # 1 = horizontal flip
        return img
    
    @staticmethod
    def color_jitter(img, brightness=0.1, contrast=0.1):
        """Color jitter using OpenCV"""
        if brightness > 0:
            factor = random.uniform(1 - brightness, 1 + brightness)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
        if contrast > 0:
            factor = random.uniform(1 - contrast, 1 + contrast)
            mean = np.mean(img)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=mean * (1 - factor))
        
        return img

# ============================================================================
# DATASET
# ============================================================================

class FaceStylizationDataset(Dataset):
    def __init__(self, face_dir, style_dir, image_size=256, split='train', augment=True):
        self.image_size = (image_size, image_size)
        self.augment = augment
        
        print(f"\n📁 Loading {split} dataset...")
        
        # Load face image paths (CelebA)
        face_split_dir = os.path.join(face_dir, split)
        print(f"  Looking for faces in: {face_split_dir}")
        
        self.face_paths = []
        if os.path.exists(face_split_dir):
            # Get all image files
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                files = glob.glob(os.path.join(face_split_dir, f'*.{ext}'))
                self.face_paths.extend(files)
            print(f"  Found {len(self.face_paths)} face images")
        else:
            # Try root directory if split subdirectory doesn't exist
            print(f"  Warning: {face_split_dir} not found, trying root directory")
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                files = glob.glob(os.path.join(face_dir, f'*.{ext}'))
                self.face_paths.extend(files)
            print(f"  Found {len(self.face_paths)} face images")
        
        # Load style image paths (WikiArt)
        style_split_dir = os.path.join(style_dir, split)
        print(f"  Looking for styles in: {style_split_dir}")
        
        self.style_paths = []
        if os.path.exists(style_split_dir):
            # Try style subdirectories first
            style_categories = []
            try:
                style_categories = [d for d in os.listdir(style_split_dir)
                                  if os.path.isdir(os.path.join(style_split_dir, d))]
            except:
                pass
            
            if style_categories:
                print(f"  Found {len(style_categories)} style categories")
                for category in style_categories:
                    category_path = os.path.join(style_split_dir, category)
                    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                        files = glob.glob(os.path.join(category_path, f'*.{ext}'))
                        self.style_paths.extend(files)
            else:
                # No subdirectories, look for images directly
                for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                    files = glob.glob(os.path.join(style_split_dir, f'*.{ext}'))
                    self.style_paths.extend(files)
            
            print(f"  Found {len(self.style_paths)} style images")
        else:
            # Try root directory
            print(f"  Warning: {style_split_dir} not found, trying root directory")
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                files = glob.glob(os.path.join(style_dir, f'*.{ext}'))
                self.style_paths.extend(files)
            print(f"  Found {len(self.style_paths)} style images")
        
        # Verify we have images
        if len(self.face_paths) == 0:
            raise ValueError(f"No face images found in {face_dir}")
        if len(self.style_paths) == 0:
            raise ValueError(f"No style images found in {style_dir}")
        
        print(f"✅ {split} dataset loaded: {len(self.face_paths)} faces, {len(self.style_paths)} styles")
    
    def __len__(self):
        return min(len(self.face_paths), len(self.style_paths))
    
    def load_image(self, path):
        """Load image using OpenCV with validation"""
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"OpenCV couldn't read image: {path}")
            
            # Ensure 3 channels
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
        except Exception as e:
            # Create a fallback image
            print(f"⚠️ Error loading {path}: {e}")
            return np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 128
    
    def __getitem__(self, idx):
        # Load face image
        face_path = self.face_paths[idx % len(self.face_paths)]
        face = self.load_image(face_path)
        
        # Load random style image
        style_idx = random.randint(0, len(self.style_paths) - 1)
        style_path = self.style_paths[style_idx]
        style = self.load_image(style_path)
        
        # Apply augmentations for training
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                face = cv2.flip(face, 1)
                style = cv2.flip(style, 1)
            
            # Color jitter (only for faces)
            if random.random() > 0.5:
                face = OpenCVTransforms.color_jitter(face, brightness=0.05, contrast=0.05)
        
        # Resize
        face = cv2.resize(face, self.image_size)
        style = cv2.resize(style, self.image_size)
        
        # Convert to tensor and normalize
        face_tensor = OpenCVTransforms.to_tensor(face)
        style_tensor = OpenCVTransforms.to_tensor(style)
        
        face_tensor = OpenCVTransforms.normalize(face_tensor)
        style_tensor = OpenCVTransforms.normalize(style_tensor)
        
        return {'face': face_tensor, 'style': style_tensor}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, base_channels=64, style_dim=256, num_residual=6):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Style injection
        self.style_fc = nn.Linear(style_dim, base_channels*4*2)
        
        # Residual blocks with adaptive instance normalization (AdaIN)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels*4) for _ in range(num_residual)
        ])
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, 3, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def adain(self, content, style):
        """Adaptive Instance Normalization"""
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + 1e-8
        
        style_mean = style[:, :style.size(1)//2].view(content.size(0), content.size(1), 1, 1)
        style_std = style[:, style.size(1)//2:].view(content.size(0), content.size(1), 1, 1)
        
        normalized = (content - content_mean) / content_std
        return normalized * style_std + style_mean
    
    def forward(self, x, style_vector=None):
        # Encode
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # Apply style if provided
        if style_vector is not None:
            style_params = self.style_fc(style_vector)
            
            # Process through residual blocks with AdaIN
            for block in self.residual_blocks:
                x = block(x)
                x = self.adain(x, style_params)
        else:
            # Just residual blocks
            for block in self.residual_blocks:
                x = block(x)
        
        # Decode
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        
        return x

class StyleEncoder(nn.Module):
    def __init__(self, style_dim=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, style_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*4, base_channels*8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple feature extractor (lightweight alternative to VGG)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return F.l1_loss(x_features, y_features)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_samples(face, style, generated, epoch, save_dir):
    """Save sample images for visualization"""
    # Denormalize from [-1, 1] to [0, 1]
    face = (face.cpu() + 1) / 2
    style = (style.cpu() + 1) / 2
    generated = (generated.cpu() + 1) / 2
    
    n = min(4, face.shape[0])
    
    fig, axes = plt.subplots(3, n, figsize=(n*3, 9))
    
    if n == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(n):
        # Convert to numpy and ensure correct range
        face_img = np.clip(face[i].permute(1, 2, 0).numpy(), 0, 1)
        style_img = np.clip(style[i].permute(1, 2, 0).numpy(), 0, 1)
        gen_img = np.clip(generated[i].permute(1, 2, 0).numpy(), 0, 1)
        
        axes[0, i].imshow(face_img)
        axes[0, i].set_title(f'Face {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(style_img)
        axes[1, i].set_title(f'Style {i+1}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(gen_img)
        axes[2, i].set_title(f'Generated {i+1}')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch}', fontsize=14, y=0.95)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_losses(loss_history, save_dir):
    """Plot training losses"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Generator loss
    if 'g_loss' in loss_history:
        axes[0, 0].plot(loss_history['g_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator loss
    if 'd_loss' in loss_history:
        axes[0, 1].plot(loss_history['d_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Content loss
    if 'content_loss' in loss_history:
        axes[1, 0].plot(loss_history['content_loss'])
        axes[1, 0].set_title('Content Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Validation loss
    if 'val_loss' in loss_history:
        axes[1, 1].plot(loss_history['val_loss'])
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_losses.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return save_path

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(config):
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create datasets
    print("\n📁 Creating datasets...")
    try:
        train_dataset = FaceStylizationDataset(
            face_dir=config.celeba_path,
            style_dir=config.wikiart_path,
            image_size=config.image_size,
            split='train',
            augment=True
        )
        
        val_dataset = FaceStylizationDataset(
            face_dir=config.celeba_path,
            style_dir=config.wikiart_path,
            image_size=config.image_size,
            split='val',
            augment=False
        )
        
        print(f"✅ Datasets created:")
        print(f"   Training samples: {len(train_dataset):,}")
        print(f"   Validation samples: {len(val_dataset):,}")
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        print("\nDebugging paths...")
        print(f"CelebA path exists: {os.path.exists(config.celeba_path)}")
        print(f"WikiArt path exists: {os.path.exists(config.wikiart_path)}")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(4, config.batch_size),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Initialize models
    print("\n🏗️ Initializing models...")
    generator = Generator(
        base_channels=config.generator_channels,
        style_dim=config.style_dim,
        num_residual=config.num_residual_blocks
    ).to(device)
    
    style_encoder = StyleEncoder(style_dim=config.style_dim).to(device)
    discriminator = Discriminator(base_channels=config.generator_channels).to(device)
    
    print(f"✅ Models initialized:")
    print(f"  Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Style encoder parameters: {sum(p.numel() for p in style_encoder.parameters()):,}")
    print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(
        list(generator.parameters()) + list(style_encoder.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )
    
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )
    
    # Learning rate schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )
    
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"training_{timestamp}")
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Loss history
    loss_history = {
        'g_loss': [],
        'd_loss': [],
        'content_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print("="*60)
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")
        
        # Training phase
        generator.train()
        style_encoder.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_content_loss = 0
        batches_processed = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move data to device
            face = batch['face'].to(device)
            style = batch['style'].to(device)
            
            # ===== Train Discriminator =====
            d_optimizer.zero_grad()
            
            with torch.no_grad():
                style_vector = style_encoder(style)
                generated = generator(face, style_vector)
            
            # Real images
            real_pred = discriminator(style)
            real_loss = mse_loss(real_pred, torch.ones_like(real_pred))
            
            # Fake images
            fake_pred = discriminator(generated.detach())
            fake_loss = mse_loss(fake_pred, torch.zeros_like(fake_pred))
            
            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()
            
            # ===== Train Generator =====
            g_optimizer.zero_grad()
            
            # Encode style and generate
            style_vector = style_encoder(style)
            generated = generator(face, style_vector)
            
            # Adversarial loss
            fake_pred = discriminator(generated)
            adv_loss = mse_loss(fake_pred, torch.ones_like(fake_pred)) * config.lambda_adv
            
            # Content loss
            content_loss = l1_loss(face, generated) * config.lambda_content
            
            # Perceptual loss
            perc_loss = perceptual_loss(face, generated) * 0.5
            
            # Style consistency loss
            style_recon = style_encoder(generated)
            style_loss = mse_loss(style_vector, style_recon) * config.lambda_style
            
            # Identity loss
            face_style = style_encoder(face)
            face_recon = generator(face, face_style)
            identity_loss = l1_loss(face, face_recon) * config.lambda_identity
            
            # Total generator loss
            g_loss = adv_loss + content_loss + perc_loss + style_loss + identity_loss
            g_loss.backward()
            g_optimizer.step()
            
            # Update statistics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_content_loss += content_loss.item()
            batches_processed += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item():.3f}',
                'C': f'{content_loss.item():.3f}'
            })
        
        # Calculate epoch averages
        avg_g_loss = epoch_g_loss / batches_processed
        avg_d_loss = epoch_d_loss / batches_processed
        avg_content_loss = epoch_content_loss / batches_processed
        
        # Validation
        generator.eval()
        style_encoder.eval()
        
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_face = val_batch['face'].to(device)[:4]
            val_style = val_batch['style'].to(device)[:4]
            
            val_style_vector = style_encoder(val_style)
            val_generated = generator(val_face, val_style_vector)
            
            # Validation loss
            val_content_loss = l1_loss(val_face, val_generated).item()
            
            # Save samples
            sample_path = save_samples(val_face, val_style, val_generated, epoch,
                                     f"{output_dir}/samples")
            print(f"📸 Samples saved: {sample_path}")
        
        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
        # Record losses
        loss_history['g_loss'].append(avg_g_loss)
        loss_history['d_loss'].append(avg_d_loss)
        loss_history['content_loss'].append(avg_content_loss)
        loss_history['val_loss'].append(val_content_loss)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"  Generator Loss: {avg_g_loss:.4f}")
        print(f"  Discriminator Loss: {avg_d_loss:.4f}")
        print(f"  Content Loss: {avg_content_loss:.4f}")
        print(f"  Validation Loss: {val_content_loss:.4f}")
        print(f"  Learning Rate: {g_optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_content_loss < best_val_loss:
            best_val_loss = val_content_loss
            best_epoch = epoch
            print(f"  🎯 NEW BEST MODEL!")
            
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'style_encoder_state_dict': style_encoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'val_loss': val_content_loss,
                'config': vars(config)
            }
            
            torch.save(checkpoint, f"{output_dir}/checkpoints/best_model.pth")
            print(f"  💾 Best model saved")
        
        # Save periodic checkpoint
        if epoch % config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'style_encoder_state_dict': style_encoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'val_loss': val_content_loss
            }
            
            torch.save(checkpoint, f"{output_dir}/checkpoints/epoch_{epoch:03d}.pth")
            print(f"  💾 Checkpoint saved")
    
    # Training complete
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"🏆 Best model: epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    # Save final model
    final_checkpoint = {
        'epoch': config.epochs,
        'generator_state_dict': generator.state_dict(),
        'style_encoder_state_dict': style_encoder.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'config': vars(config)
    }
    
    torch.save(final_checkpoint, f"{output_dir}/checkpoints/final_model.pth")
    print(f"💾 Final model saved")
    
    # Plot losses
    loss_plot_path = plot_losses(loss_history, output_dir)
    print(f"📈 Loss plot saved: {loss_plot_path}")
    
    return output_dir

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_model(checkpoint_path, config, num_samples=4):
    """Test a trained model"""
    print(f"\n🔍 Testing model: {checkpoint_path}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    generator = Generator(
        base_channels=config.generator_channels,
        style_dim=config.style_dim,
        num_residual=config.num_residual_blocks
    ).to(device)
    
    style_encoder = StyleEncoder(style_dim=config.style_dim).to(device)
    
    # Load model weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    style_encoder.load_state_dict(checkpoint['style_encoder_state_dict'])
    
    generator.eval()
    style_encoder.eval()
    
    # Create test dataset
    test_dataset = FaceStylizationDataset(
        face_dir=config.celeba_path,
        style_dir=config.wikiart_path,
        image_size=config.image_size,
        split='val',
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=num_samples,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    # Get test batch
    val_batch = next(iter(test_loader))
    test_face = val_batch['face'].to(device)
    test_style = val_batch['style'].to(device)
    
    with torch.no_grad():
        style_vector = style_encoder(test_style)
        generated = generator(test_face, style_vector)
    
    # Display results
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*3, 9))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_samples):
        # Denormalize
        face_img = (test_face[i].cpu().permute(1, 2, 0) + 1) / 2
        style_img = (test_style[i].cpu().permute(1, 2, 0) + 1) / 2
        gen_img = (generated[i].cpu().permute(1, 2, 0) + 1) / 2
        
        axes[0, i].imshow(np.clip(face_img.numpy(), 0, 1))
        axes[0, i].set_title(f'Face {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.clip(style_img.numpy(), 0, 1))
        axes[1, i].set_title(f'Style {i+1}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(np.clip(gen_img.numpy(), 0, 1))
        axes[2, i].set_title(f'Generated {i+1}')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Test Results - Epoch {checkpoint["epoch"]}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return generated

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Face Stylization Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Checkpoint path for testing')
    parser.add_argument('--celeba_path', type=str, default='./data/processed/celeba_small',
                      help='Path to CelebA dataset')
    parser.add_argument('--wikiart_path', type=str, default='./data/processed/wikiart_small',
                      help='Path to WikiArt dataset')
    parser.add_argument('--output_dir', type=str, default='./training_output',
                      help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.celeba_path = args.celeba_path
    config.wikiart_path = args.wikiart_path
    config.output_dir = args.output_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
    print("="*60)
    print("Face Stylization with StyleGAN")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"CelebA path: {config.celeba_path}")
    print(f"WikiArt path: {config.wikiart_path}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print("="*60)
    
    if args.mode == 'train':
        # Create data directories if they don't exist
        os.makedirs(config.celeba_path, exist_ok=True)
        os.makedirs(config.wikiart_path, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Check if data exists
        if not os.path.exists(config.celeba_path) or not os.listdir(config.celeba_path):
            print(f"\n⚠️ Warning: No data found in {config.celeba_path}")
            print("Please download and prepare the CelebA dataset:")
            print("1. Download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            print("2. Extract to: ./data/celeba_small/")
            print("3. Organize with train/val/test subdirectories")
            return
        
        if not os.path.exists(config.wikiart_path) or not os.listdir(config.wikiart_path):
            print(f"\n⚠️ Warning: No data found in {config.wikiart_path}")
            print("Please download and prepare the WikiArt dataset:")
            print("1. Download from: https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan")
            print("2. Extract to: ./data/wikiart_small/")
            print("3. Organize with train/val/test subdirectories")
            return
        
        # Start training
        output_dir = train_model(config)
        print(f"\n✅ Training complete!")
        print(f"📁 All outputs saved to: {output_dir}")
    
    elif args.mode == 'test':
        if not args.checkpoint:
            print("❌ Error: Please provide a checkpoint path with --checkpoint")
            return
        
        if not os.path.exists(args.checkpoint):
            print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
            return
        
        test_model(args.checkpoint, config)

if __name__ == "__main__":
    main()