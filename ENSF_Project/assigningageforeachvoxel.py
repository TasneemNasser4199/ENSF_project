import os
import torch
import pandas as pd
from monai.transforms import LoadImage
import numpy as np
import nibabel as nib

# Function to load age data from CSV file
def load_age_data(csv_file, participant_id_column, age_column):
    age_data = pd.read_csv(csv_file)
    return age_data.set_index(participant_id_column)[age_column].to_dict()

# Function to add noise to image
def add_noise(image, age_value):
    noise = torch.randint_like(image, low=-2, high=3)
    noised_image = torch.add(image, noise)
    noised_image[noised_image != age_value] = age_value # Make sure all voxel values contain the same age number
    return noised_image

# Paths for image and CSV file
image_directory = "/home/tasneem.nasser/camcan/stripped"
output_directory = "/home/tasneem.nasser/camcan/noisy_images"
csv_file = "/home/tasneem.nasser/camcan/CAMCAN.csv"
participant_id_column = "participant_id"
age_column = "age"

# Load age data from CSV
age_data = load_age_data(csv_file, participant_id_column, age_column)

# Initialize MONAI's LoadImage function
loader = LoadImage(image_only=True)

# Recursive function to process images
def process_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_T1w.nii.gz"):
                image_path = os.path.join(root, file)
                print("Processing:", image_path)
                # Load image
                image = loader(image_path)
                # Extract participant ID from filename
                participant_id = file.split("_")[0]
                print("Participant ID extracted from filename:", participant_id)  # Check participant ID extracted from filename
                # Get age value for current participant
                age_value = age_data.get(participant_id)
                print("Age value:", age_value)  # Check age value
                if age_value is not None:
                    # Add noise and assign age values
                    noised_image = add_noise(image, age_value)
                    # Save noised image
                    output_filename = f"{participant_id}_noisy.nii.gz"
                    output_path = os.path.join(output_directory, output_filename)
                    print("Output path:", output_path)  # Check output path
                    # Convert tensor to NIfTI image
                    nii = nib.Nifti1Image(noised_image.numpy(), np.eye(4))
                    nib.save(nii, output_path)
                    print("Saved:", output_path)

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Start processing images
process_images(image_directory)
