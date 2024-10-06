import os
import numpy as np
import nibabel as nib

# Paths
image_directory = "/work/souza_lab/tasneem/CNSsynthstripy/"
masks_directory = "/work/souza_lab/tasneem/CNSsynthstripy/masks"

# Create masks directory if it doesn't exist
os.makedirs(masks_directory, exist_ok=True)

# Iterate through each file in the image directory
for file in os.listdir(image_directory):
    if file.endswith(".nii.gz"):
        image_file_path = os.path.join(image_directory, file)
        print("Processing:", image_file_path)
        
        # Load the image
        try:
            image = nib.load(image_file_path)
        except Exception as e:
            print(f"Failed to load {image_file_path}: {e}")
            continue
        
        # Create mask where voxel values > 0 are set to 1
        mask_data = (image.get_fdata() > 0).astype(np.uint8)
        
        # Create new filename by inserting '_mask' before '.nii.gz'
        base_filename = file.rstrip('.nii.gz')
        mask_filename = f"{base_filename}_mask.nii.gz"
        mask_file_path = os.path.join(masks_directory, mask_filename)
        
        # Save the mask image with np.eye(4) and original headers
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4), header=image.header)
        nib.save(mask_img, mask_file_path)
        
        print(f"Saved mask: {mask_file_path}")
