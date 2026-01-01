"""
Preprocess CelebA and WikiArt datasets
"""
import os
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml

def preprocess_celeba(raw_path, output_path, img_size=256):
    """Preprocess CelebA face images"""
    print(f"Preprocessing CelebA from {raw_path} to {output_path}")
    
    # Create output directories
    train_path = Path(output_path) / "train"
    val_path = Path(output_path) / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Read partition file
    partition_file = Path(raw_path) / "list_eval_partition.txt"
    if not partition_file.exists():
        print("Warning: No partition file found. Creating 95/5 split.")
        # We'll create our own split
        image_files = list(Path(raw_path).glob("img_align_celeba/*.jpg"))
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.95)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
    else:
        # Use existing partition
        with open(partition_file, 'r') as f:
            lines = f.readlines()
        
        train_files = []
        val_files = []
        
        for line in lines:
            filename, partition = line.strip().split()
            src_path = Path(raw_path) / "img_align_celeba" / filename
            
            if partition == '0':  # Training
                train_files.append(src_path)
            elif partition == '2':  # Validation
                val_files.append(src_path)
    
    # Process and save images
    def process_files(file_list, save_dir):
        for img_path in tqdm(file_list, desc=f"Processing {save_dir.name}"):
            try:
                # Load and resize
                img = Image.open(img_path)
                img = img.resize((img_size, img_size), Image.BICUBIC)
                
                # Save
                save_path = save_dir / img_path.name
                img.save(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    process_files(train_files, train_path)
    process_files(val_files, val_path)
    
    print(f"CelebA preprocessing complete: {len(train_files)} train, {len(val_files)} val")

def preprocess_wikiart(raw_path, output_path, img_size=256):
    """Preprocess WikiArt style images"""
    print(f"Preprocessing WikiArt from {raw_path} to {output_path}")
    
    # Create output directories
    train_path = Path(output_path) / "train"
    val_path = Path(output_path) / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Get all style directories
    style_dirs = [d for d in Path(raw_path).iterdir() if d.is_dir()]
    
    total_train = 0
    total_val = 0
    
    for style_dir in style_dirs:
        style_name = style_dir.name
        print(f"  Processing {style_name}...")
        
        # Get all images
        image_files = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"    Warning: No images found in {style_name}")
            continue
        
        # Create style subdirectories
        style_train_path = train_path / style_name
        style_val_path = val_path / style_name
        style_train_path.mkdir(exist_ok=True)
        style_val_path.mkdir(exist_ok=True)
        
        # Shuffle and split
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.95)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process images
        for img_path in tqdm(train_files, desc=f"    Train"):
            try:
                img = Image.open(img_path)
                # Resize maintaining aspect ratio
                img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
                
                # Create square canvas
                square_img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
                offset = ((img_size - img.width) // 2, (img_size - img.height) // 2)
                square_img.paste(img, offset)
                
                # Save
                save_path = style_train_path / img_path.name
                square_img.save(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        for img_path in tqdm(val_files, desc=f"    Val"):
            try:
                img = Image.open(img_path)
                img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
                
                square_img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
                offset = ((img_size - img.width) // 2, (img_size - img.height) // 2)
                square_img.paste(img, offset)
                
                save_path = style_val_path / img_path.name
                square_img.save(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        total_train += len(train_files)
        total_val += len(val_files)
    
    print(f"WikiArt preprocessing complete: {total_train} train, {total_val} val images")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    img_size = config['data']['image_size']
    
    # Preprocess CelebA
    raw_celeba = "data/raw/celeba"
    processed_celeba = config['data']['celeba_path']
    preprocess_celeba(raw_celeba, processed_celeba, img_size)
    
    # Preprocess WikiArt
    raw_wikiart = "data/raw/wikiart"
    processed_wikiart = config['data']['wikiart_path']
    preprocess_wikiart(raw_wikiart, processed_wikiart, img_size)
    
    print("=" * 50)
    print("All preprocessing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()