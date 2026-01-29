"""
SIMPLE Preprocessing - Creates small dataset for Colab (30K CelebA + 8K WikiArt)
"""
import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

# ================= CONFIGURATION =================
# Everything is hardcoded here - NO command line arguments
CONFIG = {
    'celeba': {
        'raw_path': "data/raw/celeba",
        'output_path': "data/processed/celeba_small",
        'max_images': 30000,  # Total images: 25.5K train + 1.5K val
        'image_size': 256
    },
    'wikiart': {
        'raw_path': "data/raw/wikiart",
        'output_path': "data/processed/wikiart_small",
        'max_images': 8000,   # Total images: ~7.6K train + 0.4K val
        'image_size': 256,
        'max_per_style': 200
    }
}

# ================= CELEBA PROCESSING =================
def process_celeba():
    """Process 30K CelebA images"""
    print("\n" + "="*60)
    print("PROCESSING CELEBA (30K images)")
    print("="*60)
    
    cfg = CONFIG['celeba']
    raw_path = Path(cfg['raw_path'])
    output_path = Path(cfg['output_path'])
    img_size = cfg['image_size']
    
    # Create output directories
    train_path = output_path / "train"
    val_path = output_path / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    img_dir = raw_path / "img_align_celeba"
    if not img_dir.exists():
        print(f"❌ ERROR: CelebA images not found at {img_dir}")
        print("Make sure you downloaded CelebA to data/raw/celeba/")
        return 0, 0
    
    all_images = list(img_dir.glob("*.jpg"))
    print(f"Found {len(all_images)} CelebA images")
    
    # Randomly select 30K images
    random.shuffle(all_images)
    selected = all_images[:cfg['max_images']]
    
    # Split: 85% train, 15% val
    split_idx = int(len(selected) * 0.85)
    train_files = selected[:split_idx]
    val_files = selected[split_idx:]
    
    print(f"Selected: {len(train_files)} train, {len(val_files)} val")
    
    # Process train images
    print("\nProcessing train images...")
    for img_path in tqdm(train_files):
        try:
            img = Image.open(img_path)
            img = img.resize((img_size, img_size), Image.BICUBIC)
            img.save(train_path / img_path.name)
        except:
            pass
    
    # Process val images
    print("Processing val images...")
    for img_path in tqdm(val_files):
        try:
            img = Image.open(img_path)
            img = img.resize((img_size, img_size), Image.BICUBIC)
            img.save(val_path / img_path.name)
        except:
            pass
    
    print(f"✅ CelebA done: {len(train_files)} train, {len(val_files)} val")
    return len(train_files), len(val_files)

# ================= WIKIART PROCESSING =================
def process_wikiart():
    """Process 8K WikiArt images"""
    print("\n" + "="*60)
    print("PROCESSING WIKIART (8K images)")
    print("="*60)
    
    cfg = CONFIG['wikiart']
    raw_path = Path(cfg['raw_path'])
    output_path = Path(cfg['output_path'])
    img_size = cfg['image_size']
    
    # Create output directories
    train_path = output_path / "train"
    val_path = output_path / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Check if WikiArt exists
    if not raw_path.exists():
        print(f"❌ ERROR: WikiArt not found at {raw_path}")
        print("Make sure you downloaded WikiArt to data/raw/wikiart/")
        return 0, 0
    
    # Get all style folders
    style_folders = [f for f in raw_path.iterdir() if f.is_dir()]
    print(f"Found {len(style_folders)} art styles")
    
    # Common styles to use (for diversity)
    common_styles = [
        'impressionism', 'realism', 'surrealism', 'expressionism',
        'abstract', 'cubism', 'renaissance', 'baroque', 'pop_art',
        'romanticism', 'post_impressionism', 'abstract_expressionism',
        'art_nouveau', 'contemporary', 'modern', 'naive_art'
    ]
    
    # Use available common styles
    available_styles = []
    for style in common_styles:
        if (raw_path / style).exists():
            available_styles.append(style)
    
    # If not enough common styles, use whatever we have
    if len(available_styles) < 8:
        available_styles = [f.name for f in style_folders[:12]]
    
    print(f"Using {len(available_styles)} styles")
    
    # Calculate how many images per style
    images_per_style = cfg['max_images'] // len(available_styles)
    images_per_style = min(images_per_style, cfg['max_per_style'])
    
    total_train = 0
    total_val = 0
    
    # Process each style
    for style_name in tqdm(available_styles, desc="Processing styles"):
        style_dir = raw_path / style_name
        
        # Get all images in this style
        images = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
        if len(images) < 10:  # Skip styles with too few images
            continue
        
        # Randomly select images from this style
        random.shuffle(images)
        selected = images[:images_per_style]
        
        # Split: 95% train, 5% val
        split_idx = int(len(selected) * 0.95)
        train_images = selected[:split_idx]
        val_images = selected[split_idx:]
        
        # Create style directories
        style_train = train_path / style_name
        style_val = val_path / style_name
        style_train.mkdir(exist_ok=True)
        style_val.mkdir(exist_ok=True)
        
        # Process train images
        for img_path in train_images:
            try:
                img = Image.open(img_path)
                # Resize maintaining aspect ratio
                img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
                
                # Create square canvas
                square = Image.new('RGB', (img_size, img_size), (255, 255, 255))
                offset = ((img_size - img.width) // 2, (img_size - img.height) // 2)
                square.paste(img, offset)
                
                square.save(style_train / img_path.name)
            except:
                pass
        
        # Process val images
        for img_path in val_images:
            try:
                img = Image.open(img_path)
                img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
                
                square = Image.new('RGB', (img_size, img_size), (255, 255, 255))
                offset = ((img_size - img.width) // 2, (img_size - img.height) // 2)
                square.paste(img, offset)
                
                square.save(style_val / img_path.name)
            except:
                pass
        
        total_train += len(train_images)
        total_val += len(val_images)
    
    print(f"✅ WikiArt done: {total_train} train, {total_val} val")
    return total_train, total_val

# ================= CREATE CONFIG =================
def create_config(celeba_train, celeba_val, wikiart_train, wikiart_val):
    """Create config file for the small dataset"""
    config = {
        'project': {
            'name': 'msc-face-stylization-small',
            'description': 'Small dataset for Colab training'
        },
        'data': {
            'celeba_path': 'data/processed/celeba_small',
            'wikiart_path': 'data/processed/wikiart_small',
            'image_size': 256,
            'batch_size': 16,
            'num_workers': 2,
            'train_split': 0.95
        },
        'model': {
            'generator': {
                'base_channels': 64,
                'num_downsampling': 4,
                'num_residual_blocks': 6,
                'use_adain': True
            },
            'style_encoder': {
                'base_channels': 32,
                'style_dim': 256
            },
            'discriminator': {
                'base_channels': 64,
                'num_layers': 3
            },
            'text_encoder': {
                'embed_dim': 512
            }
        },
        'training': {
            'epochs': 50,
            'lr_g': 0.0002,
            'lr_d': 0.0002,
            'beta1': 0.5,
            'beta2': 0.999,
            'loss_weights': {
                'adversarial': 1.0,
                'identity': 10.0,
                'perceptual': 5.0,
                'style': 3.0,
                'clip': 2.0
            },
            'save_interval': 5,
            'log_interval': 100
        }
    }
    
    # Save config
    with open('config_small.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n📁 Config saved: config_small.yaml")

# ================= CALCULATE SIZE =================
def calculate_size(celeba_train, celeba_val, wikiart_train, wikiart_val):
    """Calculate total dataset size"""
    total_images = celeba_train + celeba_val + wikiart_train + wikiart_val
    
    # Rough size calculation
    img_size = 256
    bytes_per_image = img_size * img_size * 3  # RGB
    total_bytes = total_images * bytes_per_image
    total_gb = total_bytes / (1024 ** 3)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"CelebA: {celeba_train:,} train + {celeba_val:,} val")
    print(f"WikiArt: {wikiart_train:,} train + {wikiart_val:,} val")
    print(f"TOTAL: {total_images:,} images")
    print(f"SIZE: {total_gb:.1f} GB")
    print("="*60)
    
    return total_gb

# ================= MAIN =================
def main():
    print("\n" + "="*60)
    print("CREATING SMALL DATASET FOR COLAB")
    print("="*60)
    print("Target: 30K CelebA + 8K WikiArt")
    print("Size: ~10GB total")
    print("="*60)
    
    # Process datasets
    celeba_train, celeba_val = process_celeba()
    wikiart_train, wikiart_val = process_wikiart()
    
    # Calculate size
    total_gb = calculate_size(celeba_train, celeba_val, wikiart_train, wikiart_val)
    
    # Create config
    create_config(celeba_train, celeba_val, wikiart_train, wikiart_val)
    
    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)
    print(f"\nTo train with this dataset:")
    print("  python src/scripts/train.py --config config_small.yaml")
    print(f"\nDataset fits in Colab: {total_gb:.1f} GB < 15GB (Colab free tier)")
    print("="*60)

if __name__ == "__main__":
    main()