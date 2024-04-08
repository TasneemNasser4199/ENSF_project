import os
import numpy as np
import nibabel as nib
import pandas as pd

# Function to load age data from CSV file
def load_age_data(csv_file, participant_id_column, age_column):
    age_data = pd.read_csv(csv_file, sep='\t')
    return age_data.set_index(participant_id_column)[age_column].to_dict()

# Paths
csv_file = "/home/tasneem.nasser/camcan/participants.tsv"
participant_id_column = "participant_id"
age_column = "age"
masks_directory = "/home/tasneem.nasser/camcan/masks"
noisy_images_directory = "/home/tasneem.nasser/camcan/noisy_images_with_noise"
ground_truth_directory = "/home/tasneem.nasser/camcan/ground_truth"

# Load age data from CSV
age_data = load_age_data(csv_file, participant_id_column, age_column)

# Create ground truth directory if it doesn't exist
os.makedirs(ground_truth_directory, exist_ok=True)

# Iterate through each file in the masks directory
for mask_file in os.listdir(masks_directory):
    if mask_file.endswith("_mask.nii.gz"):
        mask_file_path = os.path.join(masks_directory, mask_file)
        participant_id = mask_file.split("_")[0]
        
        # Check if there's a corresponding noisy image with noise
        noisy_image_file = f"{participant_id}_noisy.nii.gz"
        noisy_image_file_path = os.path.join(noisy_images_directory, noisy_image_file)
        if os.path.exists(noisy_image_file_path):
            print(f"Processing: {mask_file} and {noisy_image_file}")
            
            # Load the mask and noisy image
            mask_img = nib.load(mask_file_path)
            noisy_image_img = nib.load(noisy_image_file_path)
            
            # Get the affine matrices
            mask_affine = mask_img.affine
            noisy_image_affine = noisy_image_img.affine
            
            # Extract the mask and noisy image data
            mask_data = mask_img.get_fdata()
            noisy_image_data = noisy_image_img.get_fdata()
            
            # Multiply the images voxel-wise
            ground_truth = mask_data * noisy_image_data
            
            # Save the ground truth image with the same affine as the mask
            ground_truth_file = f"{participant_id}_label.nii.gz"
            ground_truth_file_path = os.path.join(ground_truth_directory, ground_truth_file)
            nib.save(nib.Nifti1Image(ground_truth, mask_affine), ground_truth_file_path)
            print("Saved ground truth:", ground_truth_file_path)
