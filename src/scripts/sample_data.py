"""
Intelligently sample subsets from CelebA and WikiArt for efficient training
"""
import os
import random
import shutil
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import numpy as np

def sample_celeba(input_path, output_path, target_size=50000):
    """
    Sample CelebA images with diversity considerations
    """
    print(f"Sampling {target_size} images from CelebA...")
    
    # Get all image paths
    train_dir = Path(input_path) / "train"
    val_dir = Path(input_path) / "val"
    
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    
    all_images = train_images + val_images
    print(f"  Total available: {len(all_images)} images")
    
    # If we have less than target, use all
    if len(all_images) <= target_size:
        print(f"  Using all {len(all_images)} images")
        # Copy all
        (Path(output_path) / "train").mkdir(parents=True, exist_ok=True)
        (Path(output_path) / "val").mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(train_images, desc="Copying train"):
            shutil.copy(img_path, Path(output_path) / "train" / img_path.name)
        for img_path in tqdm(val_images, desc="Copying val"):
            shutil.copy(img_path, Path(output_path) / "val" / img_path.name)
        return
    
    # Calculate split (95/5)
    train_size = int(target_size * 0.95)
    val_size = target_size - train_size
    
    # Random sample
    sampled_images = random.sample(all_images, target_size)
    sampled_train = sampled_images[:train_size]
    sampled_val = sampled_images[train_size:]
    
    # Create output directories
    train_output = Path(output_path) / "train"
    val_output = Path(output_path) / "val"
    train_output.mkdir(parents=True, exist_ok=True)
    val_output.mkdir(parents=True, exist_ok=True)
    
    # Copy sampled images
    for img_path in tqdm(sampled_train, desc="Copying sampled train"):
        shutil.copy(img_path, train_output / img_path.name)
    
    for img_path in tqdm(sampled_val, desc="Copying sampled val"):
        shutil.copy(img_path, val_output / img_path.name)
    
    print(f"  Sampled: {len(sampled_train)} train, {len(sampled_val)} val images")

def sample_wikiart(input_path, output_path, target_size=12000, min_per_style=50):
    """
    Sample WikiArt images ensuring coverage of all styles
    """
    print(f"Sampling ~{target_size} images from WikiArt...")
    
    # Get all style directories
    train_dir = Path(input_path) / "train"
    val_dir = Path(input_path) / "val"
    
    style_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    
    # Count total images per style
    style_counts = {}
    for style_dir in style_dirs:
        style_name = style_dir.name
        train_images = list(style_dir.glob("*.*"))
        val_images = list((val_dir / style_name).glob("*.*")) if (val_dir / style_name).exists() else []
        style_counts[style_name] = len(train_images) + len(val_images)
    
    # Sort styles by count
    sorted_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate sampling strategy
    sampled_count = 0
    sampling_plan = {}
    
    for style_name, count in sorted_styles:
        # Calculate proportional allocation
        proportion = count / sum(style_counts.values())
        allocated = max(min_per_style, int(target_size * proportion))
        allocated = min(allocated, count)  # Don't allocate more than available
        
        sampling_plan[style_name] = allocated
        sampled_count += allocated
    
    # Adjust if we overshot
    if sampled_count > target_size:
        # Reduce from largest styles
        excess = sampled_count - target_size
        for style_name, _ in sorted_styles:
            if sampling_plan[style_name] > min_per_style:
                reduction = min(excess, sampling_plan[style_name] - min_per_style)
                sampling_plan[style_name] -= reduction
                excess -= reduction
                if excess == 0:
                    break
    
    # Create output directories
    train_output = Path(output_path) / "train"
    val_output = Path(output_path) / "val"
    train_output.mkdir(parents=True, exist_ok=True)
    val_output.mkdir(parents=True, exist_ok=True)
    
    total_sampled = 0
    
    # Sample from each style
    for style_name, target_count in tqdm(sampling_plan.items(), desc="Sampling styles"):
        # Skip if no target
        if target_count == 0:
            continue
            
        # Create style directories
        style_train_out = train_output / style_name
        style_val_out = val_output / style_name
        style_train_out.mkdir(exist_ok=True)
        style_val_out.mkdir(exist_ok=True)
        
        # Get all images for this style
        train_images = list((train_dir / style_name).glob("*.*")) if (train_dir / style_name).exists() else []
        val_images = list((val_dir / style_name).glob("*.*")) if (val_dir / style_name).exists() else []
        
        all_images = train_images + val_images
        
        if len(all_images) == 0:
            print(f"  Warning: No images found for style {style_name}")
            continue
        
        # If we have fewer than target, use all
        if len(all_images) <= target_count:
            sampled = all_images
        else:
            sampled = random.sample(all_images, target_count)
        
        # Copy to output
        for img_path in sampled:
            # Determine if it was originally train or val
            if img_path in train_images:
                output_dir = style_train_out
            else:
                output_dir = style_val_out
            shutil.copy(img_path, output_dir / img_path.name)
        
        total_sampled += len(sampled)
        print(f"    {style_name}: {len(sampled)} images")
    
    print(f"  Total sampled: {total_sampled} images across {len(sampling_plan)} styles")

def create_minimal_config():
    """Create a minimal training config for the sampled dataset"""
    config = {
        'data': {
            'image_size': 256,
            'batch_size': 16,
            'num_workers': 4,
            'train_split': 0.95,
            'style_augmentation': True
        },
        'model': {
            'generator': {'base_channels': 64, 'num_residual_blocks': 6},  # Reduced blocks
            'style_encoder': {'base_channels': 32, 'style_dim': 256},      # Smaller style dim
            'discriminator': {'base_channels': 64, 'num_layers': 3}
        },
        'training': {
            'epochs': 60,  # Reduced epochs
            'lr_g': 0.0002,
            'lr_d': 0.0002,
            'loss_weights': {
                'adversarial': 1.0,
                'identity': 10.0,
                'perceptual': 10.0,
                'style': 5.0,
                'clip': 2.0
            },
            'save_interval': 5
        }
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Sample subsets from datasets")
    parser.add_argument("--celeba_size", type=int, default=50000, help="Target CelebA size")
    parser.add_argument("--wikiart_size", type=int, default=12000, help="Target WikiArt size")
    parser.add_argument("--output_suffix", type=str, default="_subset", help="Suffix for output folders")
    args = parser.parse_args()
    
    # Load main config
    with open("config.yaml", 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Define paths
    celeba_input = main_config['data']['celeba_path']
    wikiart_input = main_config['data']['wikiart_path']
    
    celeba_output = celeba_input + args.output_suffix
    wikiart_output = wikiart_input + args.output_suffix
    
    print("=" * 60)
    print("SAMPLING DATASETS FOR EFFICIENT TRAINING")
    print(f"CelebA: {args.celeba_size} images")
    print(f"WikiArt: {args.wikiart_size} images")
    print("=" * 60)
    
    # Sample datasets
    sample_celeba(celeba_input, celeba_output, args.celeba_size)
    sample_wikiart(wikiart_input, wikiart_output, args.wikiart_size)
    
    # Update config with sampled paths
    main_config['data']['celeba_path'] = celeba_output
    main_config['data']['wikiart_path'] = wikiart_output
    main_config['training']['epochs'] = 60  # Update epochs
    
    # Save updated config
    subset_config_path = f"config_subset{args.output_suffix}.yaml"
    with open(subset_config_path, 'w') as f:
        yaml.dump(main_config, f, default_flow_style=False)
    
    # Create minimal config for quick experiments
    minimal_config = create_minimal_config()
    minimal_config['data']['celeba_path'] = celeba_output
    minimal_config['data']['wikiart_path'] = wikiart_output
    
    with open("config_minimal.yaml", 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False)
    
    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE!")
    print(f"CelebA subset saved to: {celeba_output}")
    print(f"WikiArt subset saved to: {wikiart_output}")
    print(f"\nConfig files created:")
    print(f"  - {subset_config_path} (full config with updated paths)")
    print(f"  - config_minimal.yaml (minimal config for quick experiments)")
    print("\nTo use sampled datasets, run:")
    print(f"  python train.py --config {subset_config_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()