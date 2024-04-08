import os
import numpy as np
import nibabel as nib

# Function to add random noise to each voxel in the image
def add_noise_to_image(image):
    noise = np.random.randint(-2, 3, size=image.shape)  # Generate random noise
    noisy_image = image + noise
    return noisy_image

# Directory containing noisy images
noisy_image_directory = "/home/tasneem.nasser/camcan/noisy_images"

# Output directory for noisy images with added noise
output_directory = "/home/tasneem.nasser/camcan/noisy_images_with_noise"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process each file in the noisy image directory
for file in os.listdir(noisy_image_directory):
    if file.endswith("_noisy.nii.gz"):
        file_path = os.path.join(noisy_image_directory, file)
        print("Processing:", file_path)
        
        # Load the noisy image
        noisy_image = nib.load(file_path).get_fdata()
        
        # Add noise to each voxel
        noisy_image_with_noise = add_noise_to_image(noisy_image)
        
        # Save the image with added noise
        output_file_path = os.path.join(output_directory, file)
        nib.save(nib.Nifti1Image(noisy_image_with_noise, np.eye(4)), output_file_path)
        print("Saved with noise:", output_file_path)
