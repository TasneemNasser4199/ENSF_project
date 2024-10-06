Voxel-Level Brain Age Prediction Project
This repository contains the code and data used in my ENSF 619.03 course project, which focuses on voxel-level brain age prediction using deep learning models.

Project Overview
This project aims to predict the brain age at the voxel level using MRI data. The following steps outline the workflow:

1. Preprocessing MRI Data
Tools Used:
SynthStrip: To perform skull stripping of the MRI volumes.
2. Creating Binary Segmentation Masks
Script: onemasksheaders.py
Converts all non-zero values in the segmentation masks to ones, producing binary brain masks.
3. Generating Ground Truth Age Maps
Ground truth age maps are created by assigning the age of participants to each voxel within the image volume.

Steps:
Creating Participant CSV File

Script: participantcsv.py
Generates a CSV file containing image file IDs, participant ages, and sex information. This script processes SynthStripped MRI data files, which have age and sex information embedded in their filenames.
Assigning Age to Each Voxel

Script: assigningageforeachvoxel.py
Assigns the participant’s age to every voxel in the corresponding image volume.
Introducing Noise

Script: introducingnoise.py
Adds noise to the voxel values, ensuring that the model learns voxel-specific features.
Reforming the Brain Shape

Script: creatinggroundtruthageprediction.py
Multiplies the noisy volume by the binary brain mask to reshape the noisy data into the correct brain structure, creating the final age maps.
4. Matching Files
Script: creatingmatchedcsvfiles.py
Combines the SynthStripped MRI data, binary segmentation masks, and the corresponding ground truth age maps into a unified dataset.
5. Model Training
The project includes two main models for voxel-level brain age prediction:
SwinUNETR: swinunetr.py
UNET: unet.py
Both models are trained and evaluated on the preprocessed dataset generated in the previous steps.

Environment Setup
To run this project, you must set up an environment that includes the following libraries:

MONAI: A deep learning framework for medical imaging.
Install with: pip install monai

TorchVision and PyTorch: PyTorch is the core deep learning library, and TorchVision provides utilities for handling images.
Install with: pip install torch torchvision
Ensure that PyTorch is installed with GPU support for faster model training:
Follow the PyTorch installation guide for the correct installation command depending on your system.

Nibabel: For working with neuroimaging data, particularly in the NIfTI format (.nii, .nii.gz).
Install with: pip install nibabel

NumPy: A core library for numerical computations.
Install with: pip install numpy

Ensure your environment has GPU support to efficiently run the deep learning models. If you have a CUDA-enabled GPU, make sure that PyTorch recognizes it by checking torch.cuda.is_available().

How to Run the Project
Clone this repository.
Set up the required environment with the mentioned dependencies.
Follow the workflow steps in the order described above, starting from preprocessing MRI data, creating masks, generating ground truth age maps, and running the models.
Run the corresponding scripts at each stage, which are located in the scripts folder.
