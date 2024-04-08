import monai
import torch
import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
from monai.config import print_config
from monai.transforms import (
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandRotate90d,
    
)
from monai.networks.nets import SwinUNETR
from monai.data import (
    CacheDataset,
    ThreadDataLoader
)


print_config()

# Set CUDA launch blocking to help with debugging
torch.backends.cudnn.benchmark = True
CUDA_LAUNCH_BLOCKING = 1
os.environ['TORCH_USE_CUDA_DSA'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Weights & Biases
wandb.init(project="xx", settings=wandb.Settings(start_method="fork"))
wandb.run.name = 'xxxxxx'

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    SpatialPadd(["img", "seg"], 16),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandRotate90d(keys=["img", "seg"], prob=0.5),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    Resized(keys=["img", "seg"], spatial_size=(128, 160, 128), mode=("nearest", "nearest")),
])

val_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    SpatialPadd(["img", "seg"], 16),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    RandRotate90d(keys=["img", "seg"], prob=0.5),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 160, 128)),
    Resized(keys=["img", "seg"], spatial_size=(128, 160, 128), mode=("nearest", "nearest")),
])
test_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True),
    Orientationd(keys=["img", "seg"], axcodes="RAS"),
    SpatialPadd(["img", "seg"], 16),
    ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=1.0),
    Resized(keys=["img", "seg"], spatial_size=(128, 160, 128), mode=("nearest", "nearest")),
])

def voxel_mae(pred_seg, seg):
    voxel_mae = []
    try:
        for i in range(len(seg)):
            try:
                if torch.sum(seg[i]) != 0:
                    print("Index:", i)
                    ground_truth = seg[i].clone()  # Clone the segmentation
                    prediction = pred_seg[i].clone()

                    loss_img = torch.sum(torch.abs(prediction - ground_truth)) / torch.sum(seg[i] != 0)
                    voxel_mae.append(loss_img)
            except Exception as e:
                print(f"Error occurred in iteration {i}: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

    voxel_mae = torch.stack(voxel_mae, 0)
    loss = torch.mean(voxel_mae)
    return loss


def load_data():
    shuff_data = pd.read_csv("/home/tasneem.nasser/camcan/CAMCAN.csv")
    imgs_list = list(shuff_data['imgs'])
    seg = list(shuff_data['seg'])

    length = len(imgs_list)
    train_length = int(0.75 * length)
    val_length = int(0.10 * length)
    test_length = length - train_length - val_length

    imgs_list_train = imgs_list[:train_length]
    imgs_list_val = imgs_list[train_length:train_length + val_length]
    imgs_list_test = imgs_list[train_length + val_length:]
    seg_train = seg[:train_length]
    seg_val = seg[train_length:train_length + val_length]
    seg_test = seg[train_length + val_length:]

    filenames_train = [{"img": x, "seg": y} for (x, y) in zip(imgs_list_train, seg_train)]
    filenames_val = [{"img": x, "seg": y} for (x, y) in zip(imgs_list_val, seg_val)]
    filenames_test = [{"img": x, "seg": y} for (x, y) in zip(imgs_list_test, seg_test)]
    ds_test = CacheDataset(data=filenames_test, transform=test_transforms, cache_rate=1.0, num_workers=4)
    filenames_train = [{"img": x, "seg": y} for (x, y) in zip(imgs_list_train, seg_train)]
    ds_train = CacheDataset(
    data=filenames_train,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=4,
    )
    train_loader = ThreadDataLoader(ds_train, num_workers=1, batch_size=1, shuffle=True)
    filenames_val = [{"img": x, "seg": y} for (x, y) in zip(imgs_list_val, seg_val)]
    ds_val = CacheDataset(data=filenames_val, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = ThreadDataLoader(ds_val, num_workers=1, batch_size=1, shuffle=True)
    test_loader = ThreadDataLoader(ds_test, num_workers=1, batch_size=1, shuffle=True)

    return ds_train, train_loader, ds_val, val_loader, ds_test, test_loader
    

def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir2, start_epoch=1):
    model.train()

    best_val_loss = float('inf')
    for epoch in range(1, max_epochs + 1):
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
            img, seg = batch["img"].to(device), batch["seg"].to(device)
            optimizer.zero_grad()
            pred_seg = model(img)
            loss = voxel_coef * voxel_mae(pred_seg, seg)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            print("=", end = "")
            step += 1


        train_loss = train_loss / (step + 1)

        print()
        print("Val:", end = "")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                img, seg = batch["img"].to(device), batch["seg"].to(device)
                pred_seg = model(img)
                # Calculate the mean absolute error
                loss = voxel_coef * voxel_mae(pred_seg, seg)
                val_loss += loss.item()
                print("=", end = "")

        val_loss = val_loss / len(val_loader)

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, " | ", optimizer.param_groups[0]['lr'])

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if epoch == 1:
            best_val_loss = val_loss

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
            save_path = os.path.join(root_dir2, "best_model.pt")
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
        save_path = os.path.join(root_dir2, "last_model.pt")
        torch.save(state, save_path)

        scheduler.step()

    print("Training complete.")

def load_last_model(model, optimizer, scheduler, root_dir2):
    # Load the last model weights
    last_model_path = os.path.join(root_dir2, "last_model.pt")
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        print("Last model loaded. Resuming training from epoch", start_epoch)
        return model, optimizer, scheduler, start_epoch, val_loss
    else:
        print("Last model weights not found. Starting training from scratch.")
        return model, optimizer, scheduler, 1, None

    
def test(test_loader, model, root_dir2, csv_file, participant_id_column):
    device = next(model.parameters()).device
    results_dir = os.path.join(root_dir2, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Read CSV file to get participant IDs
    df = pd.read_csv(csv_file, delimiter='\t')
    participant_ids = df[participant_id_column]

    # Load the best model weights
    best_model_path = os.path.join(root_dir2, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Best model loaded for testing.")
    else:
        print("Best model weights not found. Using the current model.")

    model.eval()

    # Initialize Weights & Biases run
    wandb.init(project="swinunetr", settings=wandb.Settings(start_method="fork"))
    wandb.run.name = 'test_run'

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img, seg = batch["img"].to(device), batch["seg"].to(device)
            filename = batch.get("img_meta_dict", {}).get("filename_or_obj", [None])[0]  # Safely get filename
            pred_seg = model(img)

            # Get participant ID for the current batch
            participant_id = participant_ids[batch_idx]

            # Calculate evaluation metrics (e.g., loss)
            loss = voxel_mae(pred_seg, seg)

            # Log evaluation metrics to Weights & Biases
            wandb.log({"Test Loss": loss.item()})

            # Save test results with participant ID as filename
            if filename is not None:
                # Extract the participant ID from the filename if available
                participant_id = os.path.splitext(os.path.basename(filename))[0]

            save_filename = f"test_{participant_id}.nii.gz"
            save_path = os.path.join(results_dir, save_filename)

            # Convert pred_seg to numpy array, squeeze, and create NIfTI image
            pred_seg_np = pred_seg.squeeze().cpu().numpy()
            nii_img = nib.Nifti1Image(pred_seg_np, np.eye(4))

            # Save NIfTI image
            nib.save(nii_img, save_path)

            print("Saving test result:", save_path)


def main():
    # Specify the path to the CSV file and the participant ID column name
    csv_file = "/home/tasneem.nasser/camcan/participants.tsv"
    participant_id_column = "participant_id"
    # Specify the name of the directory
    directory_name = "root_dir2"

    # Create the directory
    os.makedirs(directory_name, exist_ok=True)

    # Initialize your model, optimizer, scheduler, etc.
    model = SwinUNETR(
        img_size=(128, 160, 128),
        in_channels=1,
        out_channels=1,
        use_checkpoint=False,
    )

    # Move model to appropriate device
    model.to(device)

    # Initialize optimizer and scheduler
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.6)
    # Define max_epochs
    max_epochs = 100

    # Load the last model weights
    model, optimizer, scheduler, start_epoch, last_val_loss = load_last_model(model, optimizer, scheduler, directory_name)

    
    # Load the data
    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data()


    # Start training
    train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=start_epoch)
    # Test the trained model
    test(test_loader, model, directory_name, csv_file, participant_id_column)

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    main()

