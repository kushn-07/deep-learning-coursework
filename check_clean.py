# check_clean.py
import os

print("Checking project structure...")

# Essential files that MUST exist
essential_files = [
    "config.yaml",
    "run_inference.py",
    "src/scripts/preprocess_data.py",
    "src/scripts/sample_data.py",
    "src/scripts/train.py",
    "src/models/simple_generator.py",
    "src/models/style_encoder.py",
    "src/models/discriminator.py",
]

print("\n✅ ESSENTIAL FILES:")
for file in essential_files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} (MISSING!)")

# Files that should NOT exist (deleted)
deleted_files = [
    "src/scripts/train_minimal.py",
    "src/scripts/view_sample.py",
    "src/scripts/minimal_inference.py",
    "src/models/generator.py",
    "src/models/fixed_generator.py",
]

print("\n🚫 DELETED FILES (should not exist):")
for file in deleted_files:
    if os.path.exists(file):
        print(f"  ✗ {file} (STILL EXISTS - delete it!)")
    else:
        print(f"  ✓ {file} (properly deleted)")

print("\n📁 Project structure is CLEAN!" if all(not os.path.exists(f) for f in deleted_files) else "\n⚠️  Some files need cleanup!")