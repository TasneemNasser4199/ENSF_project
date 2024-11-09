import os
import csv
import random
import re
from collections import defaultdict
import torch
from monai.data import CacheDataset, ThreadDataLoader
import monai
import torch
import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
import re
from monai.transforms import (
     ToTensord,
    RandSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandRotate90d,
)
from monai.losses import DiceLoss

from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.config import print_config
from monai.networks.nets import SwinUNETR
from monai.data import (
    CacheDataset,
    ThreadDataLoader
)
from monai.transforms import Compose, EnsureType, AsDiscrete
from monai.data.utils import decollate_batch

print_config()
# Set the PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set CUDA launch blocking to help with debugging
torch.backends.cudnn.benchmark = False
#CUDA_LAUNCH_BLOCKING = 1
os.environ['TORCH_USE_CUDA_DSA'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Weights & Biases
if wandb.run is None:
    wandb.init(project="swinunetrseg", settings=wandb.Settings(start_method="fork"))
    wandb.run.name = 'x'

# Define transformations
train_transforms = Compose([
    
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "seg"]),
])


val_transforms = Compose([
    
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "seg"]),
])

test_transforms = Compose([
    
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),  
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    ToTensord(keys=["img", "seg"]),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
import csv
import random
import re
from collections import defaultdict
import torch
import nibabel as nib
import wandb
import pandas as pd
import numpy as np

from monai.transforms import (
    Compose,
    ToTensord,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandSpatialCropd,
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.config import print_config
from monai.networks.nets import SwinUNETR
from monai.data import CacheDataset, ThreadDataLoader

print_config()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if wandb.run is None:
    wandb.init(project="swinunetrseg", settings=wandb.Settings(start_method="fork"))
    wandb.run.name = 'x'

train_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "seg"]),
])

val_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    ToTensord(keys=["img", "seg"]),
])

test_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    ToTensord(keys=["img", "seg"]),
])

# def multiclass_dice_loss(pred, target):
#     dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
#     dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    
#     unique_classes = torch.unique(target)
#     total_dice = torch.tensor(0.0, device=pred.device)
#     class_count = 0

#     for cls in unique_classes:
#         if cls.item() == 0:  # Skip background
#             continue
#         pred_mask = (pred.argmax(dim=1, keepdim=True) == cls).float()
#         target_mask = (target == cls).float()
#         dice = dice_metric(y_pred=pred_mask, y=target_mask)
        
#         # Extract the scalar value from dice to avoid shape mismatch
#         total_dice += dice.item()  
#         class_count += 1

#     avg_dice = total_dice / class_count if class_count > 0 else torch.tensor(1.0, device=pred.device)
#     dice_loss_val = dice_loss(pred, target)

#     return dice_loss_val, avg_dice
###########################################################################################################
def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice score for binary masks.
    """
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def multiclass_dice_loss(pred, target):
    unique_classes = torch.unique(target)
    total_dice = torch.tensor(0.0, device=pred.device, requires_grad=True)
    class_count = 0

    for cls in unique_classes:
        if cls.item() == 0:  # Skip background
            continue

        pred_mask = (pred.argmax(dim=1, keepdim=True) == cls).float()
        target_mask = (target == cls).float()

        dice = dice_coefficient(pred_mask, target_mask)
        print(f"Class {cls.item()}, Dice Score: {dice.item()}")

        total_dice = total_dice + dice
        class_count += 1

    avg_dice = total_dice / class_count if class_count > 0 else torch.tensor(1.0, device=pred.device)
    dice_loss_val = 1 - avg_dice

    # Ensure avg_dice and dice_loss_val have gradients enabled
    print(f"Average Dice: {avg_dice}, Dice Loss: {dice_loss_val}")
    print(f"avg_dice requires grad: {avg_dice.requires_grad}, dice_loss_val requires grad: {dice_loss_val.requires_grad}")

    return dice_loss_val, avg_dice
###########################################################################################################

# def multiclass_dice_loss(pred, target):
#     """
#     Calculate Dice loss for multi-class segmentation where classes are based on unique intensities.
    
#     Args:
#         pred (torch.Tensor): Predicted segmentation with shape (B, C, D, H, W) and continuous values.
#         target (torch.Tensor): Ground truth segmentation with shape (B, 1, D, H, W) and discrete intensity values for each class.
        
#     Returns:
#         tuple: Dice loss and average Dice score across unique intensity classes.
#     """
#     unique_classes = torch.unique(target)
#     total_dice = torch.tensor(0.0, device=pred.device, requires_grad=True)  # Ensuring requires_grad=True
#     class_count = 0

#     for cls in unique_classes:
#         if cls.item() == 0:  # Skip background
#             continue

#         # Create binary masks for each class
#         pred_mask = (pred.argmax(dim=1, keepdim=True) == cls).float()
#         target_mask = (target == cls).float()

#         # Calculate Dice for the current class
#         dice = dice_coefficient(pred_mask, target_mask)
#         total_dice = total_dice + dice  # Accumulate Dice score without detaching
#         class_count += 1

#     # Calculate average Dice
#     avg_dice = total_dice / class_count if class_count > 0 else torch.tensor(1.0, device=pred.device, requires_grad=True)
#     dice_loss_val = 1 - avg_dice  # Calculate Dice loss as (1 - avg_dice)

#     return dice_loss_val, avg_dice


# def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dirsseg, start_epoch=1):
#     model.train()
  
    
#     checkpoint_path = os.path.join(root_dirsseg, "best_model.pt")
#     best_val_loss = float('inf')

#     if os.path.exists(checkpoint_path):
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location=device)
#             best_val_loss = checkpoint.get('best_val_loss', float('inf'))
#             print(f"Loaded best_val_loss from checkpoint: {best_val_loss}")
#         except Exception as e:
#             print(f"Error loading checkpoint: {e}")
#     else:
#         print("No checkpoint found. Initializing best_val_loss to infinity.")

#     for epoch in range(start_epoch, max_epochs + 1):
#         model.train()
#         train_loss = 0.0
#         step = 0

#         for batch_idx, batch in enumerate(train_loader):
#             img, seg = batch["img"].to(device), batch["seg"].to(device)
#             optimizer.zero_grad()
#             pred_seg = model(img)
#             pred_seg = torch.softmax(pred_seg, dim=1)
#             loss, dice_score = multiclass_dice_loss(pred_seg, seg)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             wandb.log({"train_loss": loss.item(), "train_dice_score": dice_score.item()})
#             step += 1

#         train_loss /= (step + 1)
#         print(f"\nEpoch {epoch} training loss: {train_loss:.4f}")

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_loader):
#                 img, seg = batch["img"].to(device), batch["seg"].to(device)
#                 pred_seg = model(img)
#                 pred_seg = torch.softmax(pred_seg, dim=1)
#                 loss, dice_score = multiclass_dice_loss(pred_seg, seg)
#                 val_loss += loss.item()

#         val_loss /= len(val_loader)
#         print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
#         wandb.log({"val_loss": val_loss})

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             state = {
#                 'epoch': epoch,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'scheduler': scheduler.state_dict(),
#                 'best_val_loss': best_val_loss,
#             }
#             torch.save(state, checkpoint_path)

#         scheduler.step()
#         print(f"Epoch {epoch} complete. Best validation loss: {best_val_loss:.4f}")
def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dirsseg, start_epoch=1):
    """
    Train a model for a given number of epochs, saving checkpoints for the best model and the last epoch.
    """
    model.train()
    checkpoint_path = os.path.join(root_dirsseg, "best_model.pt")
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Loaded best_val_loss from checkpoint: {best_val_loss}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found. Initializing best_val_loss to infinity.")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_dice_score = 0.0  # Initialize cumulative Dice score for training
        step = 0

        for batch_idx, batch in enumerate(train_loader):
            img, seg = batch["img"].to(device), batch["seg"].to(device)
            print(f"\nBatch {batch_idx + 1} - Input Image Stats: Min={img.min()}, Max={img.max()}, Mean={img.mean()}")
            print(f"Batch {batch_idx + 1} - Target Segmentation Stats: Min={seg.min()}, Max={seg.max()}, Mean={seg.mean()}")
            
            # Ensure that the input and target tensors have the correct shapes
            print(f"Input Image Shape: {img.shape}")
            print(f"Target Segmentation Shape: {seg.shape}")
            
            # Proceed with the model prediction and loss calculation
            optimizer.zero_grad()
            pred_seg = model(img)
            # Check predictions and target range
            print(f"Prediction Stats: Min={pred_seg.min()}, Max={pred_seg.max()}, Mean={pred_seg.mean()}")
            print(f"Target Stats: Min={seg.min()}, Max={seg.max()}, Mean={seg.mean()}")

            # Print shapes to ensure they align
            print(f"Prediction Shape: {pred_seg.shape}")
            print(f"Target Shape: {seg.shape}")

            #pred_seg = torch.softmax(pred_seg, dim=1)

            # # Check predictions and target range
            # print(f"Prediction Stats SOFTMAX: Min={pred_seg.min()}, Max={pred_seg.max()}, Mean={pred_seg.mean()}")
            # print(f"Target Stats: Min={seg.min()}, Max={seg.max()}, Mean={seg.mean()}")

            # # Print shapes to ensure they align
            # print(f"Prediction Shape: {pred_seg.shape}")
            # print(f"Target Shape: {seg.shape}")

            loss, dice_score = multiclass_dice_loss(pred_seg, seg)
            # Perform backpropagation and check if gradients are computed
            loss.backward()

            # Check if gradients are non-zero for model parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: Min={param.grad.min()}, Max={param.grad.max()}, Mean={param.grad.mean()}")
                else:
                    print(f"No gradient computed for {name}")

            optimizer.step()

            train_loss += loss.item()
            train_dice_score += dice_score.item()  # Accumulate Dice score
            wandb.log({"train_loss": loss.item(), "train_dice_score": dice_score.item()})
            step += 1

        # Calculate average training loss and Dice score
        train_loss /= step
        avg_train_dice_score = train_dice_score / step
        print(f"\nEpoch {epoch} training loss: {train_loss:.4f}, training Dice score: {avg_train_dice_score:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_score = 0.0  # Initialize cumulative Dice score for validation
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                img, seg = batch["img"].to(device), batch["seg"].to(device)

                pred_seg = model(img)
                #pred_seg = torch.softmax(pred_seg, dim=1)

                loss, dice_score = multiclass_dice_loss(pred_seg, seg)
                val_loss += loss.item()
                val_dice_score += dice_score.item()  # Accumulate Dice score

        # Calculate average validation loss and Dice score
        val_loss /= len(val_loader)
        avg_val_dice_score = val_dice_score / len(val_loader)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}, validation Dice score: {avg_val_dice_score:.4f}")
        
        wandb.log({"val_loss": val_loss, "val_dice_score": avg_val_dice_score})

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(state, checkpoint_path)

        scheduler.step()
        print(f"Epoch {epoch} complete. Best validation loss: {best_val_loss:.4f}")


def test(test_loader, model, root_dirsseg):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    results_dir = os.path.join(root_dirsseg, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    total_loss = 0.0
    dice_scores = []
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img, seg = batch["img"].to(device), batch["seg"].to(device)
            pred_seg = model(img)
            #pred_seg = torch.softmax(pred_seg, dim=1)
            
            loss, dice_score = multiclass_dice_loss(pred_seg, seg)
            total_loss += loss.item()
            dice_scores.append(dice_score.item())

            pred_seg_np = pred_seg.argmax(dim=1).cpu().numpy()
            save_path = os.path.join(results_dir, f"prediction_{batch_idx}.nii.gz")
            nii_img = nib.Nifti1Image(pred_seg_np[0], np.eye(4))
            nib.save(nii_img, save_path)

            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_dice_score = sum(dice_scores) / len(dice_scores) if dice_scores else 0

    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average Dice Score: {avg_dice_score:.4f}")
    wandb.log({"Average Test Loss": avg_loss, "Average Dice Score": avg_dice_score})


def safe_load(batch):
    """
    Safely load data from a batch, handling any exceptions that may occur.

    Args:
        batch (dict): A batch of data from the DataLoader, including "img", "mask", and "age" or "nonnoisyage" keys.

    Returns:
        tuple: Tensors for image, age (or nonnoisy age), and mask. Returns (None, None, None) if loading fails.
    """
    try:
        img = batch.get("img", None).to(device)
        mask = batch.get("mask", None).to(device)
        seg = batch.get("seg", None).to(device)
        return img, seg, mask
    except (EOFError, OSError, ValueError) as e:
        print(f"Skipping corrupted file {batch['img_meta_dict']['filename_or_obj']}: {e}")
        return None, None, None

def extract_details_from_filename(filename):
    """
    Extract age, sex, and dataset prefix from the filename.

    Args:
        filename (str): The filename containing embedded information about age, sex, and prefix.

    Returns:
        tuple: Prefix (str), age (float), and sex (str) extracted from the filename.
    """
    pattern = re.compile(r'(\d+(\.\d+)?_F|\d+(\.\d+)?_M)')
    match = pattern.search(filename)

    if not match:
        raise ValueError(f"Unable to extract age and sex from filename: {filename}")

    # Extract age and sex from the matched pattern
    info = match.group(0)
    age_str, sex = info.split('_')
    age = float(age_str)

    # Get the base name to determine the prefix
    base_name = filename.split('/')[-1].replace('.nii.gz', '')
    parts = base_name.split('_')

    # Handle special prefixes (AB1, AB2, CO, OAS1, OAS3, NORM, CC, IXI, CAM)
    if base_name.startswith('AB1') or base_name.startswith('AB2'):
        prefix = base_name[:3]  # Extract AB1 or AB2 as the prefix
    elif base_name.startswith('CO'):
        prefix = base_name[:2]  # Extract CO as the prefix
    elif base_name.startswith('OAS1') or base_name.startswith('OAS3'):
        prefix = base_name[:4]  # Extract OAS1 or OAS3 as the prefix
    elif base_name.startswith('NORM'):
        prefix = base_name[:4]  # Extract NORM as the prefix
    elif base_name.startswith('CC'):
        prefix = base_name[:2]  # Extract CC as the prefix
    elif base_name.startswith('IXI'):
        prefix = base_name[:3]  # Extract IXI as the prefix
    elif base_name.startswith('CAM'):
        prefix = base_name.split('-')[0]  # Extract CAM before the dash
    else:
        # For other cases, use the first part of the filename as the prefix
        prefix = parts[0]

    # Debugging output to check extracted details
    print(f"Filename: {filename}, Extracted -> Prefix: {prefix}, Age: {age}, Sex: {sex}")

    return prefix, age, sex

def load_data(
    full_data_path="/work/souza_lab/tasneem/root_dirsseg/matched_files2.csv",
    output_dir="root_dirsseg",
    test_samples_per_group=20,
    val_samples_per_group=20,
    train_samples_total=500,
    train_transforms=None,
    val_transforms=None,
    test_transforms=None,
    cache_rate=1.0,
    num_workers=4,
    batch_size=1
):
    """
    Load and split the dataset into training, validation, and test sets, and configure DataLoaders for each.

    Args:
        full_data_path (str): Path to the CSV file containing the full dataset.
        output_dir (str): Directory to save the split CSV files.
        test_samples_per_group (int): Number of samples per sex and age group for the test set.
        val_samples_per_group (int): Number of samples per sex and age group for the validation set.
        train_samples_total (int): Total number of samples for the training set.
        train_transforms (callable): Transformations to apply to the training data.
        val_transforms (callable): Transformations to apply to the validation data.
        test_transforms (callable): Transformations to apply to the test data.
        cache_rate (float): Fraction of data to cache in memory.
        num_workers (int): Number of worker threads for data loading.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: CacheDatasets and DataLoaders for training, validation, and testing splits.
    """
    # Load the dataset from the CSV file
    dataset = []
    with open(full_data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prefix, age, sex = extract_details_from_filename(row['imgs'])
            dataset.append({
                'imgs': row['imgs'],
                'seg': row['seg'],
                'mask': row['mask'],
                'age': age,
                'sex': sex
            })
        

    # Define age ranges for stratification
    age_bins = [(18, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), 
                (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80)]
    age_labels = ['18-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', 
                  '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']

    # Group data by sex and age groups
    grouped_data = defaultdict(list)
    for entry in dataset:
        for (lower, upper), label in zip(age_bins, age_labels):
            if lower <= entry['age'] < upper:
                grouped_data[(entry['sex'], label)].append(entry)
                break

    test_data, val_data, train_data = [], [], []
    used_filenames = set()

    # Select samples for test and validation sets
    for (sex, age_group), group_data in grouped_data.items():
        available_data = [entry for entry in group_data if entry['imgs'] not in used_filenames]
        
        if len(available_data) >= test_samples_per_group + val_samples_per_group:
            random.seed(42)
            test_samples = random.sample(available_data, test_samples_per_group)
            used_filenames.update([sample['imgs'] for sample in test_samples])
            
            remaining_data = [d for d in available_data if d not in test_samples]
            val_samples = random.sample(remaining_data, val_samples_per_group)
            used_filenames.update([sample['imgs'] for sample in val_samples])
            
            test_data.extend(test_samples)
            val_data.extend(val_samples)
            
            remaining_train_data = [d for d in remaining_data if d not in val_samples]
            train_samples_needed = (train_samples_total // 2) // len(age_labels)
            train_samples = random.sample(remaining_train_data, min(len(remaining_train_data), train_samples_needed))
            used_filenames.update([sample['imgs'] for sample in train_samples])
            
            train_data.extend(train_samples)
        else:
            print(f"Not enough samples for Sex: {sex}, AgeGroup: {age_group}")

    # Save the datasets to CSV files
    os.makedirs(output_dir, exist_ok=True)

    def save_to_csv(data, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['imgs', 'seg', 'mask', 'age', 'sex'])
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    save_to_csv(test_data, os.path.join(output_dir, "test_set.csv"))
    save_to_csv(val_data, os.path.join(output_dir, "val_set.csv"))
    save_to_csv(train_data, os.path.join(output_dir, "train_set.csv"))

    print("Datasets have been saved successfully.")

    # Function to create datasets and dataloaders
    def create_dataset_and_loader(data, transforms):
        filenames = [{"img": entry['imgs'], "seg": entry['seg'], "mask": entry['mask']} for entry in data]
        dataset = CacheDataset(data=filenames, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
        loader = ThreadDataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        return dataset, loader

    # Create CacheDataset and DataLoader for each split
    ds_train, train_loader = create_dataset_and_loader(train_data, train_transforms)
    ds_val, val_loader = create_dataset_and_loader(val_data, val_transforms)
    ds_test, test_loader = create_dataset_and_loader(test_data, test_transforms)
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    return ds_train, train_loader, ds_val, val_loader, ds_test, test_loader
def load_last_model(model, optimizer_class, optimizer_params, scheduler_class, scheduler_params, root_dirsseg, map_location=None):
    """
    Load the last saved model checkpoint to resume training with weights but reset optimizer and start from epoch 1.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer_class (torch.optim.Optimizer): Class of the optimizer to be reinitialized.
        optimizer_params (dict): Parameters for the optimizer.
        scheduler_class (torch.optim.lr_scheduler._LRScheduler): Class of the scheduler to be reinitialized.
        scheduler_params (dict): Parameters for the scheduler.
        root_dirsseg (str): Directory where model checkpoints are stored.
        map_location (str or torch.device, optional): Device location for loading model weights (e.g., 'cpu', 'cuda'). Defaults to None.

    Returns:
        tuple: Updated model, optimizer, scheduler, start_epoch, last_val_loss, and best_val_loss.
    """
    # Load the last model weights
    last_model_path = os.path.join(root_dirsseg, "last_model.pt")
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=map_location)
        #model.load_state_dict(checkpoint['state_dict'], strict=False)
        # *** Added Part: Filter out the mismatched final layer weights ***
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if "out.conv.conv" not in k}
        model.load_state_dict(state_dict, strict=False)
        print("Last model weights loaded (excluding final layer).")
        print("Last model weights loaded.")

        # Reinitialize the optimizer and scheduler
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params)

        last_val_loss = checkpoint['val_loss']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Start training from epoch 1
        print("Starting training from epoch 1.")
        return model, optimizer, scheduler, 1, last_val_loss, best_val_loss

    else:
        print("Last model weights not found. Starting training from scratch.")
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params)
        return model, optimizer, scheduler, 1, float('inf'), float('inf')


def main():
    """
    Main function to initialize the model, load data, and run training and testing phases.

    Returns:
        None
    """
    # Specify the name of the directory to save models and logs
    directory_name = "root_dirsseg"
    os.makedirs(directory_name, exist_ok=True)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your model
    model = SwinUNETR(
        img_size=(128, 160, 128),
        in_channels=1,
        out_channels=61,
        use_checkpoint=True,
    )
    model = model.to(device)  # Ensure model is on the same device as input


    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Initialize optimizer and scheduler parameters
    optimizer_params = {'lr': 1e-2, 'weight_decay': 1e-4}
    optimizer_class = torch.optim.Adam
    scheduler_class = torch.optim.lr_scheduler.StepLR
    scheduler_params = {'step_size': 5, 'gamma': 0.8}
    max_epochs = 450

    # Load the last model weights (if available)
    model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(
        model, optimizer_class, optimizer_params, scheduler_class, scheduler_params, directory_name
    )

    # Load the data
    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(
        full_data_path="/work/souza_lab/tasneem/root_dirsseg/matched_files2.csv",
        output_dir=directory_name,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
    )

    # Start training
    train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=start_epoch)

    # Test the trained model
    test(test_loader, model, directory_name, "/work/souza_lab/tasneem/synthstripforsegmentation/test.csv")

if __name__ == "__main__":
    main()