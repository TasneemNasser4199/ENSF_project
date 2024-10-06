import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define directories for ground truth and test results
ground_truth_dir = "/work/souza_lab/tasneem/perfectlybalancedsynthstrip/ground_truth/"
test_results_dir = "/work/souza_lab/tasneem/root_dira/test_results/"

# Initialize lists to store mean values
ground_truth_mean_values = []
test_results_mean_values = []

# Loop through test result files
for test_file in os.listdir(test_results_dir):
    # Construct ground truth file name by replacing part of the file name
    if test_file.endswith(".nii.gz"):
        base_name = test_file.replace(".nii.gz", "")
        ground_truth_file = base_name + "_label.nii.gz"
        
        # Check if corresponding ground truth file exists
        if ground_truth_file in os.listdir(ground_truth_dir):
            # Load test result and ground truth files
            test_result_path = os.path.join(test_results_dir, test_file)
            ground_truth_path = os.path.join(ground_truth_dir, ground_truth_file)
            
            test_img = nib.load(test_result_path)
            ground_truth_img = nib.load(ground_truth_path)
            
            # Get the voxel data from the images
            test_data = test_img.get_fdata()
            ground_truth_data = ground_truth_img.get_fdata()
            
            # Calculate the mean absolute value for each
            test_mean = np.mean(np.abs(test_data))
            ground_truth_mean = np.mean(np.abs(ground_truth_data))
            
            # Append to the lists
            test_results_mean_values.append(test_mean)
            ground_truth_mean_values.append(ground_truth_mean)

# Convert lists to numpy arrays
test_results_mean_values = np.array(test_results_mean_values)
ground_truth_mean_values = np.array(ground_truth_mean_values)

# Plot mean values and perform linear regression
if len(test_results_mean_values) > 0 and len(ground_truth_mean_values) > 0:
    plt.figure(figsize=(10, 10))
    plt.scatter(test_results_mean_values, ground_truth_mean_values, label='Mean Voxel Values')
    
    # Fit a linear regression model
    lr = LinearRegression()
    lr.fit(test_results_mean_values.reshape(-1, 1), ground_truth_mean_values.reshape(-1, 1))
    
    # Calculate R-squared value
    r2 = r2_score(ground_truth_mean_values, lr.predict(test_results_mean_values.reshape(-1, 1)))
    
    # Plot the linear regression line
    plt.plot(test_results_mean_values, lr.predict(test_results_mean_values.reshape(-1, 1)), color='red', label=f'Linear Fit (R^2 = {r2:.2f})')
    
    plt.xlabel('Mean Absolute Values of Predicted Voxel-level Age Prediction')
    plt.ylabel('Mean Absolute Values of the Labels')
    plt.title('Regression Plot of Mean Absolute Voxel Values')
    
    plt.legend()
    plt.grid(True)
    
    # Save the figure to file instead of showing it
    output_file = os.path.join(test_results_dir, 'regression_plot.png')
    plt.savefig(output_file)
    print(f"Figure saved successfully at {output_file}")
else:
    print("No valid data found to perform regression.")
