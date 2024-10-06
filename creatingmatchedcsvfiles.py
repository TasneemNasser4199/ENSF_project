import os
import pandas as pd

# Paths to the directories
images_dir = "/work/souza_lab/tasneem/balancedsynthstrip/"
seg_dir = "/work/souza_lab/tasneem/balancedsynthstrip/ground_truth"
segm_dir = "/work/souza_lab/tasneem/balancedsynthseg"

# Lists to store file paths
images_paths = []
seg_paths = []
segm_paths = []

# Collect .nii.gz files directly from the synthstrip directory (not recursive)
for file in os.listdir(images_dir):
    if file.endswith(".nii.gz"):
        images_paths.append(os.path.join(images_dir, file))

# Print collected images
print(f"Collected {len(images_paths)} images from {images_dir}")

# Iterate over the seg directory and collect .nii.gz files
for root, _, files in os.walk(seg_dir):
    for file in files:
        if file.endswith("_label.nii.gz"):  # Match files with _label before the extension
            seg_paths.append(os.path.join(root, file))

# Print collected segmentation files
print(f"Collected {len(seg_paths)} segmentation files from {seg_dir}")

# Iterate over the segm directory and collect .nii.gz files
for root, _, files in os.walk(segm_dir):
    for file in files:
        if file.endswith(".nii.gz"):
            segm_paths.append(os.path.join(root, file))

# Print collected segmentation mask files
print(f"Collected {len(segm_paths)} segmentation mask files from {segm_dir}")

# Create dictionaries to map filenames to their full paths
images_dict = {os.path.basename(path): path for path in images_paths}
seg_dict = {os.path.basename(path).replace('_label', ''): path for path in seg_paths}  # Remove _label for matching
segm_dict = {os.path.basename(path): path for path in segm_paths}

# Prepare lists for CSV output
matched_images = []
matched_seg = []
matched_segm = []

# Match images with seg and segm by filename
for filename, img_path in images_dict.items():
    if filename in seg_dict and filename in segm_dict:
        print(f"Matching: {filename}")  # Debug: Print matching filenames
        matched_images.append(img_path)
        matched_seg.append(seg_dict[filename])
        matched_segm.append(segm_dict[filename])

# Check if any matches were found
if not matched_images:
    print("No matching files were found.")
else:
    # Create a DataFrame
    df = pd.DataFrame({
        'imgs': matched_images,
        'seg': matched_seg,
        'segm': matched_segm
    })

    # Output the DataFrame to a CSV file
    output_csv_path = "/work/souza_lab/tasneem/balancedsynthstrip/matched_files.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Matched files written to {output_csv_path}")
