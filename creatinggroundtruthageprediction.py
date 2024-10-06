import os
import numpy as np
import nibabel as nib
import pandas as pd

# Function to load age data from CSV file
def load_age_data(csv_file, participant_id_column, age_column):
    age_data = pd.read_csv(csv_file)
    return age_data.set_index(participant_id_column)[age_column].to_dict()

# Paths
csv_file = "/work/souza_lab/tasneem/CNSsynthstripy/output.csv"
participant_id_column = "ID"
age_column = "Age"
masks_directory = "/work/souza_lab/tasneem/CNSsynthstripy/masks"
noisy_images_directory = "/work/souza_lab/tasneem/CNSsynthstripy/noisy_images"
ground_truth_directory = "/work/souza_lab/tasneem/CNSsynthstripy/ground_truth"

# Load age data from CSV
age_data = load_age_data(csv_file, participant_id_column, age_column)

# Create ground truth directory if it doesn't exist
os.makedirs(ground_truth_directory, exist_ok=True)

# Iterate through each file in the masks directory
for mask_file in os.listdir(masks_directory):
    if mask_file.endswith("_mask.nii.gz"):
        mask_file_path = os.path.join(masks_directory, mask_file)
        participant_id_with_extension = mask_file.replace("_mask.nii.gz", ".nii.gz")
        
        # Check if there's a corresponding noisy image with the exact same name
        noisy_image_file_path = os.path.join(noisy_images_directory, participant_id_with_extension)
        if os.path.exists(noisy_image_file_path):
            print(f"Processing: {mask_file} and {participant_id_with_extension}")
            
            # Load the mask and noisy image
            mask_img = nib.load(mask_file_path)
            noisy_image_img = nib.load(noisy_image_file_path)
            
            # Get the affine matrices and headers
            mask_affine = mask_img.affine
            mask_header = mask_img.header
            
            # Extract the mask and noisy image data
            mask_data = mask_img.get_fdata()
            noisy_image_data = noisy_image_img.get_fdata()
            
            # Multiply the images voxel-wise to create the ground truth
            ground_truth = mask_data * noisy_image_data
            
            # Save the ground truth image with the mask's affine and header
            ground_truth_file = f"{participant_id_with_extension.replace('.nii.gz', '_label.nii.gz')}"
            ground_truth_file_path = os.path.join(ground_truth_directory, ground_truth_file)
            nib.save(nib.Nifti1Image(ground_truth, mask_affine, mask_header), ground_truth_file_path)
            print("Saved ground truth:", ground_truth_file_path)
