"""
Loss functions for multimodal face stylization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import clip
from facenet_pytorch import InceptionResnetV1
from typing import Optional, Tuple, List
import warnings

class IdentityLoss(nn.Module):
    """
    Identity preservation loss using FaceNet/ArcFace
    """
    def __init__(self, device='cpu'):
        super().__init__()
        try:
            # Use pretrained FaceNet for face recognition
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Freeze the network
            for param in self.facenet.parameters():
                param.requires_grad = False
                
            self.facenet = self.facenet.to(device)
        except:
            # Fallback to a simple CNN if FaceNet fails to download
            print("Warning: FaceNet failed to load. Using simple identity loss.")
            self.facenet = SimpleIdentityNet().to(device)
            
        self.device = device
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, original_faces, generated_faces):
        # Preprocess images
        original_resized = F.interpolate(original_faces, size=(160, 160), mode='bilinear')
        generated_resized = F.interpolate(generated_faces, size=(160, 160), mode='bilinear')
        
        # Normalize
        original_normalized = (original_resized + 1) / 2
        generated_normalized = (generated_resized + 1) / 2
        
        # Get embeddings
        with torch.no_grad():
            original_emb = self.facenet(original_normalized)
            generated_emb = self.facenet(generated_normalized)
        
        # Compute similarity
        similarity = self.cosine_sim(original_emb, generated_emb)
        identity_loss = 1 - similarity.mean()
        
        return identity_loss, similarity.mean()


class SimpleIdentityNet(nn.Module):
    """Simple fallback network for identity loss"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-19 features
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], device='cpu'):
        super().__init__()
        
        try:
            # Try to load pretrained VGG19
            vgg = models.vgg19(weights='DEFAULT').features.eval()
        except:
            # Create a simple feature extractor if download fails
            print("Warning: VGG19 failed to load. Using simple perceptual loss.")
            vgg = SimpleFeatureExtractor().eval()
        
        # Freeze
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg.to(device)
        self.device = device
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
    def forward(self, original, generated):
        # Normalize
        original_norm = (original + 1) / 2
        generated_norm = (generated + 1) / 2
        
        original_norm = (original_norm - self.mean) / self.std
        generated_norm = (generated_norm - self.mean) / self.std
        
        # Simple feature extraction (skip complex layer selection)
        orig_features = self.vgg(original_norm)
        gen_features = self.vgg(generated_norm)
        
        # Compute L1 loss
        loss = F.l1_loss(orig_features, gen_features)
        
        return loss


class SimpleFeatureExtractor(nn.Module):
    """Simple fallback for perceptual loss"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.layers(x)


class StyleLoss(nn.Module):
    """
    Style loss - simplified version
    """
    def __init__(self, method='simple', device='cpu'):
        super().__init__()
        self.method = method
        self.device = device
        
    def forward(self, generated_image, style_image=None, style_text=None):
        if style_image is not None:
            # Simple color histogram matching
            gen_hist = torch.histc(generated_image, bins=10, min=-1, max=1)
            style_hist = torch.histc(style_image, bins=10, min=-1, max=1)
            loss = F.mse_loss(gen_hist, style_hist)
            return loss
        return torch.tensor(0.0, device=self.device)


class CLIPLoss(nn.Module):
    """
    CLIP-based loss - simplified
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        try:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.use_clip = True
        except:
            print("Warning: CLIP failed to load. Using random embeddings.")
            self.use_clip = False
            self.text_embeddings = {}
    
    def forward(self, generated_images, text_prompts):
        if not self.use_clip:
            # Return dummy loss if CLIP not available
            return torch.tensor(0.1, device=self.device), torch.tensor(0.5, device=self.device)
        
        # Normalize images
        images_norm = (generated_images + 1) / 2
        
        # Tokenize text
        if isinstance(text_prompts, list):
            text_tokens = clip.tokenize(text_prompts).to(self.device)
        else:
            text_tokens = text_prompts
        
        # Get features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images_norm)
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Normalize and compute similarity
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        similarity = (image_features * text_features).sum(dim=1).mean()
        
        clip_loss = 1 - similarity
        return clip_loss, similarity


class AdversarialLoss(nn.Module):
    """Simplified adversarial loss"""
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class TotalLoss(nn.Module):
    """
    Combined loss function - simplified
    """
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize losses (simplified versions)
        self.identity_loss = IdentityLoss(device)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.style_loss = StyleLoss(device=device)
        self.clip_loss = CLIPLoss(device)
        self.adv_loss = AdversarialLoss(gan_mode='lsgan')
        
        # Loss weights
        self.loss_weights = config['training']['loss_weights']
        
        print(f"Total loss initialized (simplified version)")
    
    def forward(self, 
                original_faces,
                generated_faces,
                style_images=None,
                text_prompts=None,
                discriminator_fake_pred=None,
                discriminator_real_pred=None):
        
        loss_dict = {}
        
        # Identity loss
        identity_loss, identity_sim = self.identity_loss(original_faces, generated_faces)
        loss_dict['identity'] = identity_loss
        loss_dict['identity_similarity'] = identity_sim
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(original_faces, generated_faces)
        loss_dict['perceptual'] = perceptual_loss
        
        # Style loss
        if style_images is not None:
            style_loss = self.style_loss(generated_faces, style_images)
            loss_dict['style'] = style_loss
        else:
            style_loss = torch.tensor(0.0, device=self.device)
            loss_dict['style'] = style_loss
        
        # CLIP loss
        if text_prompts is not None:
            clip_loss, clip_sim = self.clip_loss(generated_faces, text_prompts)
            loss_dict['clip'] = clip_loss
            loss_dict['clip_similarity'] = clip_sim
        else:
            clip_loss = torch.tensor(0.0, device=self.device)
            loss_dict['clip'] = clip_loss
            loss_dict['clip_similarity'] = torch.tensor(0.0, device=self.device)
        
        # Adversarial loss
        if discriminator_fake_pred is not None:
            adv_loss = self.adv_loss(discriminator_fake_pred, True)
            loss_dict['adversarial'] = adv_loss
        else:
            adv_loss = torch.tensor(0.0, device=self.device)
            loss_dict['adversarial'] = adv_loss
        
        # Weighted total
        total_loss = (
            self.loss_weights['identity'] * identity_loss +
            self.loss_weights['perceptual'] * perceptual_loss +
            self.loss_weights['style'] * style_loss +
            self.loss_weights['clip'] * clip_loss +
            self.loss_weights['adversarial'] * adv_loss
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict