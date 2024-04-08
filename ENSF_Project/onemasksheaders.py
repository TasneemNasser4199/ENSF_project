import os
import numpy as np
import nibabel as nib

# Paths
image_directory = "/home/tasneem.nasser/camcan/stripped"
masks_directory = "/home/tasneem.nasser/camcan/masks"

# Create masks directory if it doesn't exist
os.makedirs(masks_directory, exist_ok=True)

# List to store headers
headers = []

# Iterate through each file in the image directory
for file in os.listdir(image_directory):
    if file.endswith("_T1w.nii.gz"):
        image_file_path = os.path.join(image_directory, file)
        print("Processing:", image_file_path)
        
        # Load the image
        image = nib.load(image_file_path)
        
        # Extract header information
        header = image.header
        
        # Append header information to the list
        headers.append(header)
        
        # Extract participant ID from filename
        participant_id = file.split("_")[0]
        
        # Create mask based on the condition: voxel values higher than 0
        mask = np.where(image.get_fdata() > 0, 1, 0)
        
        # Save the mask with the appropriate filename
        mask_filename = f"{participant_id}_mask.nii.gz"
        mask_file_path = os.path.join(masks_directory, mask_filename)
        nib.save(nib.Nifti1Image(mask.astype(np.int16), image.affine), mask_file_path)  # Convert mask data type to int16
        print("Saved mask:", mask_file_path)

# Save headers to a text file
headers_file_path = os.path.join(masks_directory, "headers.txt")
with open(headers_file_path, "w") as f:
    for header in headers:
        f.write(str(header) + "\n")
print("Headers saved to:", headers_file_path)
