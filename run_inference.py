"""
FACE STYLIZATION - INFERENCE
Uses the SAME models as training (Generator + StyleEncoder)
"""

import os
import glob
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)


# ==================== CONFIGURATION ====================
CONFIG_FILE = "config_small.yaml"
DEVICE = "mps"   # "cpu", "mps", or "cuda"
# =======================================================

# -------- Import trained models (VERY IMPORTANT) --------
from models.simple_generator import SimpleGenerator
from models.style_encoder import StyleEncoder



def find_latest_checkpoint():
    """Automatically find the latest trained checkpoint"""
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return None

    runs = sorted(
        [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    )
    if not runs:
        return None

    latest_run = runs[-1]
    checkpoint_dir = os.path.join(runs_dir, latest_run, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    )

    # Prefer best or final
    for f in checkpoints:
        if "best" in f.lower() or "final" in f.lower():
            return os.path.join(checkpoint_dir, f)

    # Otherwise take latest epoch
    return os.path.join(checkpoint_dir, checkpoints[-1]) if checkpoints else None


def load_image(path, size):
    """Load and normalize image"""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)


def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor)


def main():
    print("\n" + "=" * 60)
    print("FACE STYLIZATION - INFERENCE")
    print("=" * 60)

    # -------- Device --------
    if DEVICE == "mps" and torch.backends.mps.is_available():
        device = "mps"
        print("⚡ Using Apple MPS")
    elif DEVICE == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print("⚡ Using CUDA")
    else:
        device = "cpu"
        print("⚡ Using CPU")

    # -------- Load config --------
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    image_size = config["data"]["image_size"]

    # -------- Load checkpoint --------
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        print("❌ No trained checkpoint found")
        return

    print(f"✅ Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # -------- Create models --------
    generator = SimpleGenerator(config).to(device)
    style_encoder = StyleEncoder(config).to(device)

    generator.load_state_dict(checkpoint["generator"])
    style_encoder.load_state_dict(checkpoint["style_encoder"])

    generator.eval()
    style_encoder.eval()

    print("✅ Models loaded successfully")

    # -------- Find test images --------
    def find_image(patterns):
        for p in patterns:
            files = glob.glob(p)
            if files:
                return files[0]
        return None

    content_path = find_image([
        "data/processed/celeba_small/train/000072.jpg",
        "data/processed/celeba_small/train/000072.png"
    ])

    style_path = find_image([
        "data/processed/wikiart_small/train/baroque/*.jpg",
        "data/processed/wikiart_small/train/baroque/*.png"
    ])

    if not content_path or not style_path:
        print("❌ Could not find content/style images")
        return

    print(f"🧑 Content image: {os.path.basename(content_path)}")
    print(f"🎨 Style image: {os.path.basename(style_path)}")

    # -------- Load images --------
    content = load_image(content_path, image_size).to(device)
    style = load_image(style_path, image_size).to(device)

    # -------- Inference --------
    print("\n🎨 Generating stylized face...")
    with torch.no_grad():
        style_vec = style_encoder(style)
        output = generator(content, style_vec, None)

    print("✅ Generation complete")

    # -------- Convert to images --------
    content_img = tensor_to_pil(content)
    style_img = tensor_to_pil(style)
    output_img = tensor_to_pil(output)

    # -------- Display --------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(content_img)
    axes[0].set_title("Input Face")
    axes[0].axis("off")

    axes[1].imshow(style_img)
    axes[1].set_title("Style Image")
    axes[1].axis("off")

    axes[2].imshow(output_img)
    axes[2].set_title("Generated Output")
    axes[2].axis("off")

    plt.tight_layout()

    # -------- Save result --------
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print("🎉 INFERENCE COMPLETE")
    print(f"📁 Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
