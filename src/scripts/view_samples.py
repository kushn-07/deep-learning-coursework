# view_samples.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Find the latest run
runs_dir = "runs"
latest_run = sorted(os.listdir(runs_dir))[-1]
sample_dir = os.path.join(runs_dir, latest_run, "samples")

print(f"Viewing samples from: {sample_dir}")

# Display all sample images
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])

for idx, sample_file in enumerate(sample_files[:4]):
    img = mpimg.imread(os.path.join(sample_dir, sample_file))
    row = idx // 2
    col = idx % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(sample_file)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Also print training stats
print("\nTraining completed successfully!")
print(f"Checkpoints saved in: {os.path.join(runs_dir, latest_run, 'checkpoints')}")
print("\nNext steps:")
print("1. View generated images above")
print("2. Increase image size to 128x128")
print("3. Train for more epochs (10-20)")
print("4. Add proper loss functions")