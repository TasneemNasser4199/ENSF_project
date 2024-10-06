import monai
import torch
import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
from monai.transforms import (
     ToTensord,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandRotate90d,
)
from monai.config import print_config
from monai.networks.nets import SwinUNETR
from monai.data import (
    CacheDataset,
    ThreadDataLoader
)

print_config()
# Set the PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Set CUDA launch blocking to help with debugging
torch.backends.cudnn.benchmark = True
#CUDA_LAUNCH_BLOCKING = 1
os.environ['TORCH_USE_CUDA_DSA'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Weights & Biases
if wandb.run is None:
    wandb.init(project="swinunetr", settings=wandb.Settings(start_method="fork"))
    wandb.run.name = 'x'

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["img", "age","mask"], ensure_channel_first=True),
    Orientationd(keys=["img", "age","mask"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandRotate90d(keys=["img", "age","mask"], prob=0.5),
    RandSpatialCropd(keys=["img", "age","mask"], roi_size=(128, 160, 128)),
    Resized(keys=["img", "age", "mask"], spatial_size=(128, 160, 128), mode=("nearest", "nearest", "nearest")),
    ToTensord(keys=["img", "age", "mask"]),
])

val_transforms = Compose([
    LoadImaged(keys=["img", "age", "mask"], ensure_channel_first=True),
    Orientationd(keys=["img", "age", "mask"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandRotate90d(keys=["img", "age", "mask"], prob=0.5),
    RandSpatialCropd(keys=["img", "age", "mask"], roi_size=(128, 160, 128)),
    Resized(keys=["img", "age", "mask"], spatial_size=(128, 160, 128), mode=("nearest", "nearest", "nearest")),
    ToTensord(keys=["img", "age", "mask"]),
])


test_transforms = Compose([
    LoadImaged(keys=["img", "age", "mask"], ensure_channel_first=True),  # Load the images
    Orientationd(keys=["img", "age", "mask"], axcodes="RAS"),            # Reorient to RAS orientation
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),            # Scale the intensity range
    ToTensord(keys=["img", "age", "mask"]),
])

def voxel_mae(pred_age, age, mask):
    voxel_mae = []
    try:
        for i in range(len(age)):
            try:
                if torch.sum(age[i]) > 3:
                    ground_truth = age[i].clone()  # Clone the age maps
                    prediction = pred_age[i].clone()

                    # Apply the mask to both prediction and ground truth to focus on the foreground
                    masked_prediction = prediction * mask[i]  # Apply mask to prediction
                    masked_ground_truth = ground_truth * mask[i]  # Apply mask to ground truth
                    # Compute loss only on the masked foreground
   
                    if torch.sum(mask[i]) > 0:  # Ensure that the mask has non-zero elements
                        print(f"Mask for iteration {i} has non-zero elements.")
                        loss_img = torch.sum(torch.abs(masked_prediction - masked_ground_truth)) / torch.sum(mask[i])
                        voxel_mae.append(loss_img)
                    else:
                        print(f"Skipping iteration {i} due to empty mask.")

                    
            except Exception as e:
                print(f"Error occurred in iteration {i}: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

    # If `voxel_mae` is empty, handle the case
    if len(voxel_mae) == 0:
        print("voxel_mae is empty. Skipping the computation.")
        return torch.tensor(0.0)  # Return a zero tensor as default to prevent failure

    # Stack the tensors and compute the mean
    voxel_mae = torch.stack(voxel_mae, 0)
    loss = torch.mean(voxel_mae)
    return loss


# Custom function to safely load images and skip corrupted ones

def safe_load(batch):
    try:
        # Attempt to load the image, age, and mask data
        img = batch["img"].to(device)
        age = batch["age"].to(device)
        mask = batch["mask"].to(device)
        return img, age, mask
    except (EOFError, OSError, ValueError) as e:
        # Log and skip the corrupted file
        print(f"Skipping corrupted file {batch['img_meta_dict']['filename_or_obj']}: {e}")
        return None, None, None  # Adjust this to return three values

# def test(test_loader, model, root_dira, test_csv):
#     device = next(model.parameters()).device
#     results_dir = os.path.join(root_dira, "test_results")
#     os.makedirs(results_dir, exist_ok=True)

#     # Load the best model weights
#     best_model_path = os.path.join(root_dira, "best_model.pt")
#     if os.path.exists(best_model_path):
#         checkpoint = torch.load(best_model_path, map_location=device)
#         model.load_state_dict(checkpoint['state_dict'])
#         print("Best model loaded for testing.")
#     else:
#         print("Best model weights not found. Using the current model.")

#     model.eval()

#     # Initialize Weights & Biases run (ensure only one initialization)
#     if wandb.run is None:
#         wandb.init(project="swinunetr", settings=wandb.Settings(start_method="fork"))
#         wandb.run.name = 'test_run'

#     # Load the test CSV to get the image paths
#     test_df = pd.read_csv(test_csv)

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
           
#             # Use safe_load to handle corrupted files
#             img, age, mask = safe_load(batch)

#             # If any are None (corrupted file), skip this iteration
#             if img is None or age is None or mask is None:
#                 continue

           

#             # Ensure batch index is within bounds of test_df
#             if batch_idx >= len(test_df):
#                 print(f"Batch index {batch_idx} out of range in test CSV.")
#                 continue
            
#             # Extract the corresponding image path from the CSV file
#             img_path = test_df.iloc[batch_idx]['imgs']

#             # Print statements for debugging
#             print(f"\nBatch Index: {batch_idx}")
#             print(f"Original image path from CSV: {img_path}")

#             # Ensure image path exists and is valid
#             if img_path is not None and os.path.exists(img_path):
#                 try:
#                     # Extract the filename and extension
#                     base_name = os.path.splitext(os.path.basename(img_path))[0]
#                     ext = os.path.splitext(img_path)[1]

#                     # Print extracted base name and extension
#                     print(f"Extracted base name: {base_name}")
#                     print(f"Extracted extension: {ext}")

#                     # Create the save filename with base name and extension
#                     save_filename = f"{base_name}{ext}"

#                     # Print final save filename
#                     print(f"Final Save Filename: {save_filename}")
#                 except Exception as e:
#                     print(f"Error extracting and creating save filename: {e}")
#                     save_filename = f"subject_{batch_idx}_test.nii.gz"  # Fallback filename
#             else:
#                 print("Image path is None or invalid, using fallback filename.")
#                 save_filename = f"subject_{batch_idx}_test.nii.gz"  # Fallback filename

#             save_path = os.path.join(results_dir, save_filename)
#             print(f"Save Path: {save_path}")

#             try:
#                 # Ensure the model and inputs are on the same device
#                 img = img.to(device)
#                 age = age.to(device)
#                 mask = mask.to(device)
#                 # Apply the mask to the input image and the ground truth age map
#                 masked_img = img * mask  # Masked input image
#                 masked_age = age * mask  # Masked ground truth (age)

#                 # Perform the forward pass through the model using the masked input image
#                 pred_age = model(masked_img)

#                 # Calculate evaluation metrics (check for empty prediction)
#                 if pred_age is None or pred_age.nelement() == 0:
#                     print(f"Empty prediction for batch index {batch_idx}. Skipping...")
#                     continue

#                 loss = voxel_mae(pred_age, age)

#                 # Log evaluation metrics to Weights & Biases
#                 wandb.log({"Test Loss": loss.item()})

#                 # Save the voxel age predicted maps with the new filename
#                 pred_age_np = pred_age.squeeze().cpu().numpy()
#                 nii_img = nib.Nifti1Image(pred_age_np, np.eye(4))

#                 nib.save(nii_img, save_path)
#                 print(f"Test result saved successfully: {save_path}")

#             except Exception as e:
#                 print(f"Error during model prediction or saving the result: {e}")


from monai.inferers import SlidingWindowInferer

def test(test_loader, model, root_dirs, test_csv):
    device = next(model.parameters()).device
    results_dir = os.path.join(root_dirs, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    # Load the best model
    best_model_path = os.path.join(root_dirs, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Best model loaded for testing.")
    else:
        print("Best model not found. Using current model.")

    model.eval()

    # Initialize Weights & Biases run (ensure only one initialization)
    if wandb.run is None:
        wandb.init(project="swinunetr", settings=wandb.Settings(start_method="fork"))
        wandb.run.name = 'test_run'

    # Load the test CSV to get the image paths
    test_df = pd.read_csv(test_csv)

    # Set up sliding window inferer to handle large input images
    window_size = (128, 160, 128)  # Adjust the size as needed to reduce memory consumption

    overlap = .50
    inferer = SlidingWindowInferer(
        roi_size=window_size,
        sw_batch_size=3,
        overlap=overlap,
        mode='gaussian',  # Gaussian blending mode
        sigma_scale=0.1,  # Adjust smoothing effect
        device=device
    )

    total_loss = 0.0  # Variable to accumulate the loss
    total_batches = 0  # Counter for the number of batches

    with torch.no_grad():  # Ensure inference is done without gradient tracking
        for batch_idx, batch in enumerate(test_loader):
            img, age, mask = safe_load(batch)
            if img is None or age is None or mask is None:
                continue

            img_path = test_df.iloc[batch_idx]['imgs']
            save_filename = os.path.basename(img_path)
            save_path = os.path.join(results_dir, save_filename)  # Use results_dir to save

            try:
                img, age, mask = img.to(device), age.to(device), mask.to(device)
                masked_img = img * mask
                
                # Perform sliding window inference
                pred_age = inferer(masked_img, model)
                pred_age = pred_age * mask
                loss = voxel_mae(pred_age, age, mask)
                 # Accumulate the loss and count the batch
                total_loss += loss.item()
                total_batches += 1

                # Log evaluation metrics to Weights & Biases
                wandb.log({"Test Loss": loss.item()})

                # Save predictions as NIfTI
                pred_age_np = pred_age.squeeze().cpu().numpy()
                nii_img = nib.Nifti1Image(pred_age_np, np.eye(4))
                nib.save(nii_img, save_path)
                print(f"Test result saved successfully: {save_path}")

            except Exception as e:
                print(f"Error during prediction or saving result: {e}")
         # Calculate and log the average loss
    # Calculate and log the average loss
    if total_batches > 0:
        average_loss = total_loss / total_batches
        print(f"Average Test Loss: {average_loss}")
        wandb.log({"Average Test Loss": average_loss})
    else:
        print("No valid batches for testing.")


def load_data(test_set_path="/work/souza_lab/tasneem/perfectlybalancedsynthstrip/test.csv"):
    # Load the CSV file containing the full dataset
    full_data = pd.read_csv("/work/souza_lab/tasneem/perfectlybalancedsynthstrip/matched_files.csv")

    # Check if the test set is already saved
    if os.path.exists(test_set_path):
        print(f"Loading existing test set from {test_set_path}")
        test_data = pd.read_csv(test_set_path)
        # Use 'imgs' column to filter out the test data from the full dataset
        train_val_data = full_data[~full_data['imgs'].isin(test_data['imgs'])]
    else:
        # Shuffle the rows of the DataFrame
        full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Define the portion of data to be used for the test set
        test_length = int(0.15 * len(full_data))  # 15% of the data as the test set

        # Separate the test set and the remaining data
        test_data = full_data.iloc[:test_length]
        train_val_data = full_data.iloc[test_length:]

        # Save the test set to a file
        test_data.to_csv(test_set_path, index=False)
        print(f"Test set saved to {test_set_path}")

    # Shuffle the training and validation data
    train_val_data = train_val_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Confirm that no test data exists in train_val_data
    common_files = set(train_val_data['imgs']).intersection(set(test_data['imgs']))
    if common_files:
        print("Warning: Some test files are still present in train_val_data:")
        print(common_files)
        # Optionally, you can remove these files or take further action
        train_val_data = train_val_data[~train_val_data['imgs'].isin(common_files)]
        print("Test files removed from train_val_data.")
    else:
        print("No common files found between train_val_data and test_data.")

    # Extract the columns as lists
    imgs_list = list(train_val_data['imgs'])
    age = list(train_val_data['age'])
    mask = list(train_val_data['mask'])

    # Calculate the lengths for training and validation splits
    train_length = int(0.75 * len(train_val_data))
    val_length = len(train_val_data) - train_length

    # Split the data into training and validation sets
    imgs_list_train = imgs_list[:train_length]
    imgs_list_val = imgs_list[train_length:]
    age_train = age[:train_length]
    age_val = age[train_length:]
    mask_train = mask[:train_length]
    mask_val = mask[train_length:]

    # Extract the test set lists
    imgs_list_test = list(test_data['imgs'])
    age_test = list(test_data['age'])
    mask_test = list(test_data['mask'])
    # Custom function to safely create CacheDataset by skipping corrupted files
    

    def safe_create_cache_dataset(data, transform, cache_rate, num_workers):
        safe_data = []
        for item in data:
            try:
                # Try loading the image and age maps to check if they're corrupted
                nib.load(item['img'])
                nib.load(item['age'])
                nib.load(item['mask'])
                safe_data.append(item)  # Add the item if both are valid
            except Exception as e:
                print(f"Skipping corrupted file: {item['img']} or {item['mask']} or {item['age']} due to error: {e}")
        
        # If no valid data remains, raise an error
        if len(safe_data) == 0:
            raise ValueError("No valid data found after removing corrupted files.")
        
        # Proceed with creating the CacheDataset with safe data
        return CacheDataset(data=safe_data, transform=transform, cache_rate=cache_rate, num_workers=num_workers)

    # Create lists of dictionaries for each split
    filenames_train = [{"img": x, "age": y, "mask": z} for (x, y, z) in zip(imgs_list_train, age_train, mask_train)]
    filenames_val = [{"img": x, "age": y, "mask": z} for (x, y, z) in zip(imgs_list_val, age_val, mask_val)]
    filenames_test = [{"img": x, "age": y, "mask": z} for (x, y, z) in zip(imgs_list_test, age_test, mask_test)]

    # Use the safe loader when creating CacheDataset objects
    ds_train = safe_create_cache_dataset(filenames_train, train_transforms, cache_rate=1.0, num_workers=4)
    ds_val = safe_create_cache_dataset(filenames_val, val_transforms, cache_rate=1.0, num_workers=4)
    ds_test = safe_create_cache_dataset(filenames_test, test_transforms, cache_rate=1.0, num_workers=4)

    # Create DataLoader objects
    train_loader = ThreadDataLoader(ds_train, num_workers=3, batch_size=2, shuffle=True)
    val_loader = ThreadDataLoader(ds_val, num_workers=3, batch_size=2, shuffle=True)
    test_loader = ThreadDataLoader(ds_test, num_workers=3, batch_size=1, shuffle=False)

    return ds_train, train_loader, ds_val, val_loader, ds_test, test_loader


   
def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dirs, start_epoch=1):
    model.train()

    #best_val_loss = float('inf')
        # Define the path to the checkpoint file
    checkpoint_path = os.path.join(root_dirs, "best_model.pt")

    # Initialize best_val_loss
    if os.path.exists(checkpoint_path):
        # Try to load the saved checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Default to infinity if key is missing
            print(f"Loaded best_val_loss from checkpoint: {best_val_loss}")
        except Exception as e:
            # If there is an error loading the checkpoint, initialize to infinity
            print(f"Error loading checkpoint: {e}")
            best_val_loss = float('inf')
    else:
        # If checkpoint doesn't exist, initialize to infinity
        print("No checkpoint found. Initializing best_val_loss to infinity.")
        best_val_loss = float('inf')

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        print("Epoch ", epoch)
        print("Train:", end ="")
        if epoch < 50:
            voxel_coef = 1
        elif 50 <= epoch < 130:
            voxel_coef = 1
        else:
            voxel_coef = 1.3
        
        step = 0
        for batch_idx, batch in enumerate(train_loader):
            img, age, mask = batch["img"].to(device), batch["age"].to(device), batch["mask"].to(device)
            file_name = batch.get("file_name", "Unknown file")  # Assuming file names are included in the batch
            
            optimizer.zero_grad()
            
            # Apply the mask to the input image and the ground truth age map
            masked_img = img * mask  # Masked input image
            masked_age = age * mask  # Masked ground truth (age)
            
            # Forward pass
            pred_age = model(masked_img)
            
            # Compute loss
            voxel_mae_value = voxel_mae(pred_age, masked_age, mask)

            # Check if voxel_mae is None (i.e., empty tensor list)
            if voxel_mae_value is None:
                #print(f"The voxel_mae list is empty for the file: {file_name}")
                #print(f"pred_age tensor: {pred_age}")
                #print(f"age tensor: {age}")
                continue  # Skip this batch to avoid errors

            loss = voxel_coef * voxel_mae_value

            # Check if loss requires gradient
            # print(f"loss.requires_grad: {loss.requires_grad}")        

            #loss = voxel_coef * voxel_mae(pred_age, age)
            if loss.requires_grad:
                loss.backward()
            # else:
            #     print("Loss tensor does not require gradients.")

            #loss.backward()
            train_loss += loss.item()
            optimizer.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            print("=", end = "")
            step += 1

        train_loss = train_loss / (step + 1)

        print()
        print("Val:", end = "")
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                img, age, mask = batch["img"].to(device), batch["age"].to(device), batch["mask"].to(device)
                
                # Apply the mask to the input image and the ground truth age map
                masked_img = img * mask  # Masked input image
                masked_age = age * mask  # Masked ground truth (age)

                # Forward pass for validation
                pred_age = model(masked_img)

                # Compute validation loss
                loss = voxel_coef * voxel_mae(pred_age, masked_age, mask)
                val_loss += loss.item()
                print("=", end="")
        val_loss = val_loss / len(val_loader)

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, " | ", optimizer.param_groups[0]['lr'])

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # if epoch == 1:
        #     best_val_loss = val_loss

        if val_loss < best_val_loss:
            print("Saving best model")
            best_val_loss = val_loss
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            save_path = os.path.join(root_dirs, "best_model.pt")
            torch.save(state, save_path)

        # Save last model weights
        print("Saving last model")
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss,
        }
        save_path = os.path.join(root_dirs, "last_model.pt")
        torch.save(state, save_path)

        # Step the scheduler and log the updated learning rate
        print(f"Before step: {optimizer.param_groups[0]['lr']}")
        scheduler.step()  # Update the learning rate
        torch.cuda.empty_cache()
        print(f"After step: {optimizer.param_groups[0]['lr']}")
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"lr": current_lr})
        print(f"Learning Rate after step: {current_lr}")

    print("Training complete.")

# def load_last_model(model, optimizer, scheduler, root_dira):
#     # Load the last model weights
#     last_model_path = os.path.join(root_dira, "last_model.pt")
#     if os.path.exists(last_model_path):
#         checkpoint = torch.load(last_model_path)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])
#         start_epoch = checkpoint['epoch'] + 1
#         last_val_loss = checkpoint['val_loss']
#         best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Get the best val loss from checkpoint
#         print(f"Last model loaded. Resuming training from epoch {start_epoch}")
#         print(f"Learning Rate after resetting: {optimizer.param_groups[0]['lr']}")
#         return model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss
#                 # Manually reset the learning rate if needed
#         #for param_group in optimizer.param_groups:
#         #     param_group['lr'] = 0.0005  # Set your desired learning rate here
#         #     print(f"Learning Rate after resetting: {optimizer.param_groups[0]['lr']}")
        
#         #return model, optimizer, scheduler, start_epoch, val_loss
#     else:
#         print("Last model weights not found. Starting training from scratch.")
#         return model, optimizer, scheduler, 1, float('inf'), float('inf')


def load_last_model(model, optimizer, scheduler, root_dirs, reset_lr=None):
    # Load the last model weights
    last_model_path = os.path.join(root_dirs, "last_model.pt")
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        last_val_loss = checkpoint['val_loss']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Get the best val loss from checkpoint if available
        print(f"Last model loaded. Resuming training from epoch {start_epoch}")
        
        # Optionally reset the learning rate if a new learning rate is provided
        if reset_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = reset_lr
            print(f"Learning Rate after resetting: {optimizer.param_groups[0]['lr']}")
        
        return model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss
    else:
        print("Last model weights not found. Starting training from scratch.")
        return model, optimizer, scheduler, 1, float('inf'), float('inf')



def main():
    # Specify the name of the directory to save models and logs
    directory_name = "root_dirs"

    # Create the directory
    os.makedirs(directory_name, exist_ok=True)

    # Initialize your model, optimizer, scheduler, etc.
    model = SwinUNETR(
        img_size=(128, 160, 128),
        #img_size=(256, 256, 256),
        in_channels=1,
        out_channels=1,
        use_checkpoint=True,
    )

    # Move model to appropriate device
    #model.to(device)
    model = torch.nn.DataParallel(model).to(device)
    # Initialize optimizer and scheduler
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)
    
    

    # Define max_epochs
    max_epochs = 450

    # Load the last model weights
    #model, optimizer, scheduler, start_epoch, last_val_loss = load_last_model(model, optimizer, scheduler, directory_name)
    # Load the last model weights, including the last validation loss and the best validation loss
    model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(model, optimizer, scheduler, directory_name, reset_lr=1e-3)

    # Load the data
    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(test_set_path="/work/souza_lab/tasneem/perfectlybalancedsynthstrip/test.csv")

    # Start training
    train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=start_epoch)
    
    # Test the trained model
    test(test_loader, model, directory_name, "/work/souza_lab/tasneem/perfectlybalancedsynthstrip/test.csv")

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    main()

