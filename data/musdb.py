import os
import shutil

### MUSDB18-HQ dataset preprocessing
source_dir = "/home/hsun/Datasets/Musdb18HQ/train"  # main folder
dest_dir = "/home/hsun/Datasets/Musdb18HQ/wavfiles"  # destination folder

# Loop through all subdirectories in aa
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # Path to mixture.wav in current subdirectory
        mixture_path = os.path.join(subdir_path, 'mixture.wav')

        # Check if mixture.wav exists
        if os.path.exists(mixture_path):
            # Create new filename (subfolder name + .wav)
            new_filename = f"{subdir}.wav"
            # Path for the new file in cc folder
            dest_path = os.path.join(dest_dir, new_filename)

            # Copy the file with new name
            shutil.copy2(mixture_path, dest_path)
            print(f"Copied {mixture_path} to {dest_path}")

print("Done!")