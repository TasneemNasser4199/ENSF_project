
####calculate means for resized images:
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from skimage.transform import resize as skimage_resize

# Function to calculate mean absolute value of non-zero voxels from a NIfTI file
def calculate_mean_absolute_value(nii_file):
    img_data = nib.load(nii_file).get_fdata()
    non_zero_voxels = img_data[np.nonzero(img_data)]
    absolute_values = np.abs(non_zero_voxels)
    mean_absolute_value = np.mean(absolute_values)
    return mean_absolute_value

# Function to resize images to match the dimensions of the reference image
def resize_to_reference(ref_img, img_data):
    return skimage_resize(img_data, ref_img.shape, order=1, preserve_range=True, affine=ref_img.affine)

# Directory paths
dir1_path = "/home/tasneem.nasser/root_dir2/test_results2/"
dir2_path = "/home/tasneem.nasser/camcan/ground_truth/"
resized_dir_path = "/home/tasneem.nasser/resized_images/"

# Check if the resized directory exists, if not, create it
if not os.path.exists(resized_dir_path):
    os.makedirs(resized_dir_path)

# Lists to store mean voxel values
dir1_mean_values = []
dir2_mean_values = []

# Iterate over files in directory 1
for filename in os.listdir(dir1_path):
    if filename.endswith(".nii.gz"):
        # Get the common part of the filename
        common_part = filename.split("test_")[1].split(".nii.gz")[0]

        # Find corresponding file in directory 2
        dir2_filename = os.path.join(dir2_path, f"{common_part}_label.nii.gz")

        # Check if the corresponding file exists in directory 2
        if os.path.exists(dir2_filename):
            # Load the NIfTI files
            img1 = nib.load(os.path.join(dir1_path, filename))
            img2 = nib.load(dir2_filename)

            # Get the data and header from the reference image (from dir1)
            img1_data = img1.get_fdata()
            img1_header = img1.header
            img1_affine = img1.affine

            # Resize the image from dir2 to match the dimensions of the image from dir1
            img2_resized_data = resize_to_reference(img1, img2.get_fdata())

            # Create a NIfTI image from the resized array with the original header
            resized_nifti = nib.Nifti1Image(img2_resized_data, img1_affine, header=img1_header)

            # Save the resized image using nibabel with the filename from dir2_path
            resized_filename = os.path.join(resized_dir_path, f"{common_part}_resized.nii.gz")
            nib.save(resized_nifti, resized_filename)

            # Calculate mean voxel values for both files
            dir1_mean_value = calculate_mean_absolute_value(os.path.join(dir1_path, filename))
            dir2_mean_value = calculate_mean_absolute_value(resized_filename)

            # Append mean values to lists
            dir1_mean_values.append(dir1_mean_value)
            dir2_mean_values.append(dir2_mean_value)

# Convert lists to numpy arrays
dir1_mean_values = np.array(dir1_mean_values).reshape(-1, 1)
dir2_mean_values = np.array(dir2_mean_values)

# Check data types and ranges
print("Data type of dir1_mean_values:", dir1_mean_values.dtype)
print("Data type of dir2_mean_values:", dir2_mean_values.dtype)
print("Minimum value of dir1_mean_values:", np.min(dir1_mean_values))
print("Maximum value of dir1_mean_values:", np.max(dir1_mean_values))
#Print the minimum and maximum values of dir2_mean_values
print(f"Minimum value of dir2_mean_values: {min(dir2_mean_values)}")
print(f"Maximum value of dir2_mean_values: {max(dir2_mean_values)}")

# Check if both arrays contain at least one sample
if len(dir1_mean_values) > 0 and len(dir2_mean_values) > 0:
    # Plot mean values
    plt.figure(figsize=(10, 10))
    plt.scatter(dir1_mean_values, dir2_mean_values, label='Mean Voxel Values')
    plt.xlabel('Mean Values Folder 1')
    plt.ylabel('Mean Values Folder 2')
    plt.title('Regression Plot of Mean Voxel Values')
    
    # Fit a line using linear regression
    lr = LinearRegression()
    lr.fit(np.array(dir1_mean_values).reshape(-1, 1), np.array(dir2_mean_values).reshape(-1, 1))
    r2 = r2_score(dir2_mean_values, lr.predict(np.array(dir1_mean_values).reshape(-1, 1)))

    # Plot the linear regression line
    plt.plot(dir1_mean_values, lr.predict(np.array(dir1_mean_values).reshape(-1, 1)), color='red', label=f'Linear Fit (R^2 = {r2:.2f})')

    plt.xlabel('Mean Absolute Values of Predicted Voxel-level Age Prediction')
    plt.ylabel('Mean Absolute Values of the Labels')
    plt.title('Regression Plot of Mean Absolute Voxel Values')

    plt.legend()
    plt.grid(True)

    # Save the figure
    save_path = os.path.join(resized_dir_path, "mean_voxel_values_plot.png")
    plt.savefig(save_path)

####calculate means directly for input_output:
# import os
# import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from skimage.transform import resize as skimage_resize

# # Function to calculate mean absolute value of non-zero voxels from a NIfTI file
# def calculate_mean_absolute_value(nii_file):
#     img_data = nib.load(nii_file).get_fdata()
#     non_zero_voxels = img_data[np.nonzero(img_data)]
#     absolute_values = np.abs(non_zero_voxels)
#     mean_absolute_value = np.mean(absolute_values)
#     return mean_absolute_value

# # Directory paths
# dir1_path = "/home/tasneem.nasser/root_dir2/test_results2/"
# dir2_path = "/home/tasneem.nasser/camcan/ground_truth/"

# # Lists to store mean voxel values
# dir1_mean_values = []
# dir2_mean_values = []

# # Iterate over files in directory 1
# for filename in os.listdir(dir1_path):
#     if filename.endswith(".nii.gz"):
#         # Get the common part of the filename
#         common_part = filename.split("test_")[1].split(".nii.gz")[0]

#         # Find corresponding file in directory 2
#         dir2_filename = os.path.join(dir2_path, f"{common_part}_label.nii.gz")

#         # Check if the corresponding file exists in directory 2
#         if os.path.exists(dir2_filename):
#             # Calculate mean absolute values for both files
#             dir1_mean_value = calculate_mean_absolute_value(os.path.join(dir1_path, filename))
#             dir2_mean_value = calculate_mean_absolute_value(dir2_filename)

#             # Append mean values to lists
#             dir1_mean_values.append(dir1_mean_value)
#             dir2_mean_values.append(dir2_mean_value)

# # Convert lists to numpy arrays
# dir1_mean_values = np.array(dir1_mean_values).reshape(-1, 1)
# dir2_mean_values = np.array(dir2_mean_values)

# # Check if both arrays contain at least one sample
# if len(dir1_mean_values) > 0 and len(dir2_mean_values) > 0:
#     # Plot mean values
#     plt.figure(figsize=(10, 10))
#     plt.scatter(dir1_mean_values, dir2_mean_values, label='Mean Voxel Values')
#     plt.xlabel('Mean Values Folder 1')
#     plt.ylabel('Mean Values Folder 2')
#     plt.title('Regression Plot of Mean Voxel Values')
    
#     # Fit a line using linear regression
#     lr = LinearRegression()
#     lr.fit(np.array(dir1_mean_values).reshape(-1, 1), np.array(dir2_mean_values).reshape(-1, 1))
#     r2 = r2_score(dir2_mean_values, lr.predict(np.array(dir1_mean_values).reshape(-1, 1)))

#     # Plot the linear regression line
#     plt.plot(dir1_mean_values, lr.predict(np.array(dir1_mean_values).reshape(-1, 1)), color='red', label=f'Linear Fit (R^2 = {r2:.2f})')

#     plt.xlabel('Mean Absolute Values of Predicted Voxel-level Age Prediction')
#     plt.ylabel('Mean Absolute Values of the Labels')
#     plt.title('Regression Plot of Mean Absolute Voxel Values')

#     plt.legend()
#     plt.grid(True)

#     # Save the figure
#     save_path = "mean_voxel_values_plot.png"
#     plt.savefig(save_path)
