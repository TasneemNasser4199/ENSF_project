import os
import re
import pandas as pd

# Define the path to the folder
folder_path = '/work/souza_lab/tasneem/CNSsynthstripy/'

# Create a list to store the extracted data
data = []

# Update the regex pattern to include filenames starting with CAM
pattern = re.compile(r'^(?P<id>[A-Za-z0-9_\-]+)_(?P<age>\d+\.\d+|\d+)_?(?P<sex>[MF])')

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".nii.gz"):
        match = pattern.match(filename)
        if match:
            # Extract ID, Age, and Sex from the filename
            id_ = match.group('id')
            age = match.group('age')
            sex = match.group('sex')
            
            # Append the extracted data to the list
            data.append([id_, age, sex])
        else:
            print(f"File '{filename}' does not match the expected pattern.")

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data, columns=['ID', 'Age', 'Sex'])

# Define the output CSV file
output_csv = '/work/souza_lab/tasneem/CNSsynthstripy/output.csv'

# Write the DataFrame to a CSV file
df.to_csv(output_csv, index=False)

print(f"Data successfully written to {output_csv}")
