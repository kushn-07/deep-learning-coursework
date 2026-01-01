"""
SIMPLE INFERENCE - One file solution
"""
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import glob

# ========== 1. LOAD YOUR TRAINED MODEL ==========
print("Loading your trained model...")

# Load checkpoint
checkpoint_path = "runs/20260101_200415/checkpoints/checkpoint_001.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"✓ Checkpoint loaded from {checkpoint_path}")

# Create a simple generator (same as your training)
class SimpleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def forward(self, x, style=None, text=None):
        return self.model(x)

# Create model and load weights
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = SimpleGenerator().to(device)
model.load_state_dict(checkpoint['generator'])
model.eval()
print(f"✓ Model loaded on {device}")

# ========== 2. FIND TEST IMAGES ==========
print("\nFinding test images...")

# Find a face image
face_files = glob.glob("data/processed/celeba_subset/train/*.jpg") + \
             glob.glob("data/processed/celeba_subset/train/*.png")
content_path = face_files[0] if face_files else None

# Find a style image
style_files = glob.glob("data/processed/wikiart_subset/train/*/*.jpg") + \
              glob.glob("data/processed/wikiart_subset/train/*/*.png")
style_path = style_files[0] if style_files else None

if not content_path or not style_path:
    print("❌ Error: No test images found!")
    print("Make sure you ran the sampling script:")
    print("  python src/scripts/sample_data.py --celeba_size 50000 --wikiart_size 12000")
    exit()

print(f"✓ Content face: {content_path}")
print(f"✓ Style image: {style_path}")

# ========== 3. LOAD AND PROCESS IMAGES ==========
def load_image(path, size=128):
    """Load and preprocess image"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

content_tensor = load_image(content_path).to(device)
style_tensor = load_image(style_path).to(device)

print(f"✓ Images loaded: {content_tensor.shape}, {style_tensor.shape}")

# ========== 4. RUN INFERENCE ==========
print("\nGenerating stylized face...")
with torch.no_grad():
    # Note: We're only using content image since our simple model ignores style
    output_tensor = model(content_tensor)
print("✓ Generation complete!")

# ========== 5. VISUALIZE RESULTS ==========
def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
    return transforms.ToPILImage()(tensor)

# Convert tensors to images
content_img = tensor_to_image(content_tensor)
style_img = tensor_to_image(style_tensor)
output_img = tensor_to_image(output_tensor)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(content_img)
axes[0].set_title('Input Face', fontsize=14)
axes[0].axis('off')

axes[1].imshow(style_img)
axes[1].set_title('Style Reference', fontsize=14)
axes[1].axis('off')

axes[2].imshow(output_img)
axes[2].set_title('Generated Stylization', fontsize=14)
axes[2].axis('off')

plt.suptitle('One-Shot Face Stylization - Your MSc Project', fontsize=16)
plt.tight_layout()

# Save result
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "first_inference.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\n{'='*60}")
print("✅ INFERENCE SUCCESSFUL!")
print(f"{'='*60}")
print(f"Results saved to: {output_path}")
print("\nYour model has generated its first stylized face!")
print("\nNext steps for your MSc project:")
print("1. View the generated image")
print("2. Train for more epochs (10-20)")
print("3. Increase image resolution to 256x256")
print("4. Add proper style conditioning")
print(f"{'='*60}")

# Show the plot
plt.show()