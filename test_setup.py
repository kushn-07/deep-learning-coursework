import sys
print("Python version:", sys.version)

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import clip
print("CLIP imported successfully")

from facenet_pytorch import InceptionResnetV1
print("FaceNet imported successfully")

print("\n✅ All imports successful!")