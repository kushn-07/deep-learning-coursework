"""
Dataset class for multimodal face stylization
Handles loading face images, style images, and text prompts
"""
import os
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import clip
import json

class MultimodalFaceStylizationDataset(Dataset):
    """
    Dataset for one-shot multimodal face stylization
    
    For each sample returns:
        - content_image: Face to stylize
        - style_image: Reference style image
        - text_prompt: Optional text description
        - style_label: Style category (for evaluation)
    """
    
    def __init__(
        self,
        celeba_root: str,
        wikiart_root: str,
        image_size: int = 256,
        split: str = "train",
        use_text_prompts: bool = True,
        text_prompt_file: Optional[str] = None,
        augment: bool = True,
        style_mixing_prob: float = 0.3,
    ):
        """
        Args:
            celeba_root: Path to CelebA dataset
            wikiart_root: Path to WikiArt dataset
            image_size: Size to resize images to
            split: 'train' or 'val'
            use_text_prompts: Whether to use text conditioning
            text_prompt_file: JSON file with style->prompt mappings
            augment: Whether to apply data augmentation
            style_mixing_prob: Probability to mix styles from different categories
        """
        self.celeba_root = Path(celeba_root) / split
        self.wikiart_root = Path(wikiart_root) / split
        self.image_size = image_size
        self.split = split
        self.use_text_prompts = use_text_prompts
        self.augment = augment and (split == "train")
        self.style_mixing_prob = style_mixing_prob
        
        # Load CLIP tokenizer (for text prompts)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.tokenizer = clip.tokenize
        
        # Load image paths
        self._load_image_paths()
        
        # Load text prompts if provided
        self.style_to_prompts = self._load_text_prompts(text_prompt_file)
        
        # Define transforms
        self.content_transform = self._get_content_transform()
        self.style_transform = self._get_style_transform()
        
        # Cache for loaded style images (to speed up training)
        self.style_cache = {}
        self.cache_size = 1000 if split == "train" else 100
        
        print(f"Dataset initialized: {len(self.face_paths)} faces, "
              f"{len(self.style_paths)} styles, {len(self.style_categories)} categories")
    
    def _load_image_paths(self):
        """Load paths to face and style images"""
        # Load CelebA face images
        self.face_paths = list(self.celeba_root.glob("*.*"))
        self.face_paths = [p for p in self.face_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not self.face_paths:
            raise ValueError(f"No face images found in {self.celeba_root}")
        
        # Load WikiArt style images
        self.style_paths = []
        self.style_categories = []
        
        # Get all style categories
        style_dirs = [d for d in self.wikiart_root.iterdir() if d.is_dir()]
        
        for style_dir in style_dirs:
            category = style_dir.name
            image_paths = list(style_dir.glob("*.*"))
            image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            if image_paths:
                self.style_paths.extend(image_paths)
                self.style_categories.extend([category] * len(image_paths))
        
        if not self.style_paths:
            raise ValueError(f"No style images found in {self.wikiart_root}")
        
        # Create mapping from category to indices
        self.category_to_indices = {}
        for idx, category in enumerate(self.style_categories):
            if category not in self.category_to_indices:
                self.category_to_indices[category] = []
            self.category_to_indices[category].append(idx)
        
        # Available categories
        self.available_categories = list(self.category_to_indices.keys())
        
    def _load_text_prompts(self, prompt_file: Optional[str]) -> Dict[str, list]:
        """
        Load text prompts for style categories
        If no file provided, generate generic prompts
        """
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompts = json.load(f)
            print(f"Loaded text prompts from {prompt_file}")
            return prompts
        
        # Default prompts for common art styles
        default_prompts = {
            "impressionism": [
                "an impressionist painting with visible brush strokes",
                "soft edges and vibrant colors like Monet",
                "capturing the feeling of a moment",
                "loose brushwork and emphasis on light"
            ],
            "realism": [
                "a realistic painting with fine details",
                "photorealistic style with accurate colors",
                "detailed and precise representation",
                "like a high-resolution photograph"
            ],
            "surrealism": [
                "a surreal painting with dreamlike elements",
                "fantastical and imaginative composition",
                "unexpected juxtapositions and symbolism",
                "like a painting by Salvador Dali"
            ],
            "expressionism": [
                "an expressionist painting with emotional intensity",
                "bold colors and distorted forms",
                "conveying strong feelings and moods",
                "like a painting by Edvard Munch"
            ],
            "abstract": [
                "an abstract painting with geometric shapes",
                "non-representational art with bold colors",
                "focused on form and color rather than realism",
                "like a painting by Kandinsky"
            ],
            "cubism": [
                "a cubist painting with fragmented forms",
                "geometric shapes and multiple viewpoints",
                "like a painting by Picasso",
                "deconstructed and reassembled forms"
            ],
            "renaissance": [
                "a renaissance painting with classical beauty",
                "balanced composition and realistic figures",
                "like a painting by Leonardo da Vinci",
                "attention to perspective and human form"
            ],
            "baroque": [
                "a baroque painting with dramatic lighting",
                "rich colors and intense emotions",
                "grand and theatrical composition",
                "like a painting by Caravaggio"
            ]
        }
        
        # Add generic prompts for any missing categories
        for category in self.available_categories:
            if category not in default_prompts:
                default_prompts[category] = [
                    f"a painting in the {category} style",
                    f"artistic style of {category}",
                    f"inspired by {category} art",
                    f"{category} artistic interpretation"
                ]
        
        return default_prompts
    
    def _get_content_transform(self):
        """Transforms for content (face) images"""
        if self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _get_style_transform(self):
        """Transforms for style images"""
        # Style images can have more aggressive augmentation
        if self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _load_and_cache_style_image(self, idx: int) -> torch.Tensor:
        """Load style image with caching for efficiency"""
        if idx in self.style_cache:
            return self.style_cache[idx]
        
        style_path = self.style_paths[idx]
        style_image = Image.open(style_path).convert('RGB')
        style_tensor = self.style_transform(style_image)
        
        # Add to cache
        if len(self.style_cache) < self.cache_size:
            self.style_cache[idx] = style_tensor
        
        return style_tensor
    
    def _get_text_prompt(self, style_category: str) -> str:
        """Get a text prompt for a given style category"""
        if not self.use_text_prompts:
            return ""
        
        if style_category in self.style_to_prompts:
            prompts = self.style_to_prompts[style_category]
            return random.choice(prompts)
        
        # Fallback generic prompt
        generic_prompts = [
            "an artistic rendering",
            "a stylized portrait",
            "creative visual interpretation",
            "artistic transformation"
        ]
        return random.choice(generic_prompts)
    
    def _get_tokenized_prompt(self, text: str) -> torch.Tensor:
        """Tokenize text prompt for CLIP"""
        if not text:
            # Return empty tensor if no text
            return torch.zeros((1, 77), dtype=torch.long)
        
        tokens = self.tokenizer(text, truncate=True).squeeze(0)
        return tokens
    
    def __len__(self) -> int:
        """Number of samples in dataset"""
        return len(self.face_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample
        
        Returns dictionary with:
            - content_image: Tensor [3, H, W]
            - style_image: Tensor [3, H, W]
            - text_prompt: String
            - text_tokens: Tensor [77] (CLIP tokens)
            - style_category: String
            - content_path: String (for debugging)
            - style_path: String (for debugging)
        """
        # Load content (face) image
        content_path = self.face_paths[idx]
        content_image = Image.open(content_path).convert('RGB')
        content_tensor = self.content_transform(content_image)
        
        # Select style image
        if random.random() < self.style_mixing_prob and self.augment:
            # Sometimes mix styles from different categories
            style_idx = random.randint(0, len(self.style_paths) - 1)
        else:
            # Usually select from same or random category
            if random.random() < 0.7 and self.augment:
                # Select random style
                style_idx = random.randint(0, len(self.style_paths) - 1)
            else:
                # Select from random category (not necessarily same as any specific face)
                random_category = random.choice(self.available_categories)
                style_idx = random.choice(self.category_to_indices[random_category])
        
        # Load style image (with caching)
        style_tensor = self._load_and_cache_style_image(style_idx)
        style_category = self.style_categories[style_idx]
        style_path = str(self.style_paths[style_idx])
        
        # Get text prompt
        text_prompt = self._get_text_prompt(style_category)
        text_tokens = self._get_tokenized_prompt(text_prompt)
        
        return {
            'content_image': content_tensor,
            'style_image': style_tensor,
            'text_prompt': text_prompt,
            'text_tokens': text_tokens,
            'style_category': style_category,
            'content_path': str(content_path),
            'style_path': style_path,
        }


class PairedDataset(MultimodalFaceStylizationDataset):
    """
    Dataset with fixed (face, style) pairs for validation
    Useful for consistent evaluation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create deterministic pairs for validation
        if self.split == "val":
            self.pairs = self._create_fixed_pairs()
    
    def _create_fixed_pairs(self):
        """Create fixed face-style pairs for consistent validation"""
        pairs = []
        num_pairs = min(1000, len(self.face_paths))  # Fixed number of validation pairs
        
        for i in range(num_pairs):
            face_idx = i % len(self.face_paths)
            style_idx = i % len(self.style_paths)
            pairs.append((face_idx, style_idx))
        
        return pairs
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == "val" and hasattr(self, 'pairs'):
            # Use fixed pairs for validation
            face_idx, style_idx = self.pairs[idx % len(self.pairs)]
            
            # Load content image
            content_path = self.face_paths[face_idx]
            content_image = Image.open(content_path).convert('RGB')
            content_tensor = self.content_transform(content_image)
            
            # Load style image
            style_tensor = self._load_and_cache_style_image(style_idx)
            style_category = self.style_categories[style_idx]
            style_path = str(self.style_paths[style_idx])
            
            # Get text prompt
            text_prompt = self._get_text_prompt(style_category)
            text_tokens = self._get_tokenized_prompt(text_prompt)
            
            return {
                'content_image': content_tensor,
                'style_image': style_tensor,
                'text_prompt': text_prompt,
                'text_tokens': text_tokens,
                'style_category': style_category,
                'content_path': str(content_path),
                'style_path': style_path,
            }
        else:
            # For training, use random pairing
            return super().__getitem__(idx)


def get_dataloaders(config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders from config
    
    Returns:
        train_loader, val_loader
    """
    # Extract config parameters
    data_config = config['data']
    training_config = config['training']
    
    # Create datasets
    train_dataset = MultimodalFaceStylizationDataset(
        celeba_root=data_config['celeba_path'],
        wikiart_root=data_config['wikiart_path'],
        image_size=data_config['image_size'],
        split='train',
        use_text_prompts=True,
        augment=True,
        style_mixing_prob=0.3,
    )
    
    val_dataset = PairedDataset(
        celeba_root=data_config['celeba_path'],
        wikiart_root=data_config['wikiart_path'],
        image_size=data_config['image_size'],
        split='val',
        use_text_prompts=True,
        augment=False,  # No augmentation for validation
        style_mixing_prob=0.0,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(8, data_config['batch_size']),  # Smaller batch for validation
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Train dataloader: {len(train_loader)} batches")
    print(f"Val dataloader: {len(val_loader)} batches")
    
    return train_loader, val_loader


def test_dataset():
    """Quick test function"""
    import matplotlib.pyplot as plt
    
    # Test with minimal config
    test_config = {
        'data': {
            'celeba_path': 'data/processed/celeba_subset',
            'wikiart_path': 'data/processed/wikiart_subset',
            'image_size': 128,  # Smaller for testing
            'batch_size': 4,
            'num_workers': 0,
        },
        'training': {}
    }
    
    print("Testing dataset...")
    
    # Create dataset
    dataset = MultimodalFaceStylizationDataset(
        celeba_root=test_config['data']['celeba_path'],
        wikiart_root=test_config['data']['wikiart_path'],
        image_size=test_config['data']['image_size'],
        split='train',
        use_text_prompts=True,
        augment=True,
    )
    
    # Get a sample
    sample = dataset[0]
    
    print(f"Content image shape: {sample['content_image'].shape}")
    print(f"Style image shape: {sample['style_image'].shape}")
    print(f"Text prompt: {sample['text_prompt']}")
    print(f"Text tokens shape: {sample['text_tokens'].shape}")
    print(f"Style category: {sample['style_category']}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Denormalize images
    content_img = sample['content_image'] * 0.5 + 0.5
    style_img = sample['style_image'] * 0.5 + 0.5
    
    axes[0].imshow(content_img.permute(1, 2, 0))
    axes[0].set_title("Content (Face)")
    axes[0].axis('off')
    
    axes[1].imshow(style_img.permute(1, 2, 0))
    axes[1].set_title(f"Style: {sample['style_category']}")
    axes[1].axis('off')
    
    plt.suptitle(f"Text: {sample['text_prompt'][:50]}...")
    plt.tight_layout()
    plt.savefig('dataset_test.png')
    print("Saved visualization to dataset_test.png")
    
    # Test dataloader
    train_loader, val_loader = get_dataloaders(test_config)
    
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Content images: {batch['content_image'].shape}")
    print(f"  Style images: {batch['style_image'].shape}")
    print(f"  Text tokens: {batch['text_tokens'].shape}")
    
    return dataset, train_loader


if __name__ == "__main__":
    # Run test
    test_dataset()