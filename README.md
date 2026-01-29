First put both the downloaded dataset inside data/raw/ [make folder in root directory]
Also make the folder called processed inside data folder
then...

Step 1: run preprocess_data.py in scripts folder
    This will:
        Resize all CelebA faces to 256x256
        Split them into train/val (95%/5%)
        Process WikiArt images (resize to fit in 256x256 squares)
        Organize by style category

Step 2: run train.py of scripts folder 
    using the command:
        python src/scripts/train.py

    This will create models

Step 3: run: python run_inference.py