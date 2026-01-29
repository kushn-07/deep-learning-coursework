"""
Test the dataset implementation
"""
import sys
sys.path.append('src')

from data.dataset import test_dataset

if __name__ == "__main__":
    print("Testing dataset implementation...")
    dataset, loader = test_dataset()
    print("\n✅ Dataset test completed successfully!")
    
    # Show batch info
    batch = next(iter(loader))
    print(f"\nSample batch keys: {list(batch.keys())}")
    print(f"Batch size: {batch['content_image'].shape[0]}")