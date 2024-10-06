import os
import torch
import pandas as pd
import numpy as np
import nibabel as nib
import re

# Function to load age data from CSV file
def load_age_data(csv_file, participant_id_column, age_column):
    age_data = pd.read_csv(csv_file)
    return age_data.set_index(participant_id_column)[age_column].to_dict()

# Function to add noise to image
def add_noise(image, age_value):
    noise = torch.randint_like(image, low=-2, high=3)
    noised_image = torch.add(image, noise)
    noised_image[noised_image != age_value] = age_value  # Make sure all voxel values contain the same age number
    return noised_image

# Custom function to load images using nibabel
def load_image(image_path):
    try:
        print(f"Attempting to load image from: {image_path}")
        img = nib.load(image_path)
        print(f"Image loaded successfully with shape: {img.shape}")
        img_data = img.get_fdata()
        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        return img_tensor
    except Exception as e:
        print(f"Failed to load image from {image_path}: {e}")
        return None

# Paths for image and CSV file
image_directory = "/work/souza_lab/tasneem/CNSsynthstripy"
output_directory = "/work/souza_lab/tasneem/CNSsynthstripy/age_images"
csv_file = "/work/souza_lab/tasneem/CNSsynthstripy/output.csv"
participant_id_column = "ID"
age_column = "Age"

# Load age data from CSV
age_data = load_age_data(csv_file, participant_id_column, age_column)

# Recursive function to process images
def process_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz"):
                image_path = os.path.join(root, file)
                print("Processing:", image_path)
                
                # Updated regex to handle a broader range of filenames
                match = re.match(r'^([A-Za-z0-9_\-]+)_(\d+\.?\d*)_([MF])', file)
                if match:
                    participant_id = match.group(1)
                    print("Participant ID extracted from filename:", participant_id)  # Check participant ID extracted from filename
                    
                    # Get age value for current participant
                    age_value = age_data.get(participant_id)
                    print("Age value:", age_value)  # Check age value
                    if age_value is not None:
                        # Load image using nibabel
                        image = load_image(image_path)
                        if image is None:
                            print("Skipping file due to loading error:", image_path)
                            continue
                        
                        # Add noise and assign age values
                        noised_image = add_noise(image, age_value)
                        # Save noised image with the original filename
                        output_path = os.path.join(output_directory, file)
                        print("Output path:", output_path)  # Check output path
                        # Convert tensor to NIfTI image
                        nii = nib.Nifti1Image(noised_image.numpy(), np.eye(4))
                        nib.save(nii, output_path)
                        print("Saved:", output_path)
                else:
                    print(f"Filename '{file}' does not match the expected pattern.")

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Start processing images
process_images(image_directory)

