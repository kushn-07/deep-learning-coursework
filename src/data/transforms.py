"""
Custom transforms for face stylization dataset
"""
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from PIL import Image

class RandomStyleAugmentation:
    """
    Apply random augmentations specifically for style images
    Helps with one-shot generalization
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # Apply random color adjustments
        img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
        img = F.adjust_saturation(img, random.uniform(0.8, 1.2))
        
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = F.rotate(img, angle)
            
        return img

class ContentPreservingAugmentation:
    """
    Augmentations for content (face) images that preserve identity
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # Mild augmentations that don't affect identity
        img = F.adjust_brightness(img, random.uniform(0.9, 1.1))
        img = F.adjust_contrast(img, random.uniform(0.9, 1.1))
        
        # Horizontal flip (preserves identity)
        if random.random() < 0.5:
            img = F.hflip(img)
            
        return img

def get_content_transform(image_size=256, augment=True):
    """Get transforms for content images"""
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ContentPreservingAugmentation(p=0.3),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def get_style_transform(image_size=256, augment=True):
    """Get transforms for style images"""
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RandomStyleAugmentation(p=0.7),
            transforms.RandomResizedCrop(
                image_size, 
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])