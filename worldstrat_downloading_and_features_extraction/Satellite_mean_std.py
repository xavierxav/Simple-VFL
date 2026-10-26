"""
This script calculates the mean and standard deviation of the Sentinel L2A dataset. 
It loads all data in GPU memory and then calculates the mean and std, channel after channel.
It takes a lot of GPU memory but it is still doable on a 80 Gb GPU.
"""

import argparse
import os

import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class SatelliteDataset(Dataset):
    def __init__(self, metadata, root, transform=None):
        self.metadata = metadata
        self.root = root
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        poi = self.metadata.at[idx, 'POI']  # Get the POI corresponding to the index
        n = self.metadata.at[idx, 'n']
        
        img_path = os.path.join(self.root, 'satellite_dataset', 'lr_dataset', poi, 'L2A', f"{poi}-{n}-L2A_data.tiff")
        image = torch.tensor(tiff.imread(img_path), dtype=torch.float32)
        image = image.permute(2, 0, 1)
        return image

def load_all_data(data_loader):
    all_data = []
    total_samples = len(data_loader.dataset)
    
    progress_bar = tqdm(total=total_samples, desc="Loading all data to CPU")
    for batch in data_loader:
        all_data.append(batch)
        progress_bar.update(batch.size(0))
    progress_bar.close()
    
    return torch.cat(all_data, dim=0)

def calculate_mean_std_sequential(all_data, device, nb_channels):
    means = []
    stds = []

    for channel in range(nb_channels):
        print(f"\nProcessing channel {channel}")
        channel_data = all_data[:, channel].to(device)
        
        print("Calculating mean...")
        mean = torch.mean(channel_data)
        means.append(mean)
        
        print("Calculating std...")
        std = torch.std(channel_data)
        stds.append(std)
        
        print(f"Channel {channel} - Mean: {mean.item():.6f}, Std: {std.item():.6f}")
        
        # Clear GPU memory
        del channel_data
        torch.cuda.empty_cache()

    return torch.tensor(means), torch.tensor(stds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the per-channel mean and standard deviation of the Sentinel L2A dataset."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="Root folder containing the 'satellite_dataset' directory "
             "(default: $DATA_ROOT).",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error(
            "--data-root is required (or set the DATA_ROOT environment variable)."
        )

    # Folder containing satellite_dataset/
    data_root = os.path.normpath(os.path.expanduser(args.data_root))

    csv_file = os.path.join(data_root, 'satellite_dataset', 'metadata.csv')
    df = pd.read_csv(os.path.normpath(csv_file), sep=';')

    # Rename 1st column to 'POI'
    df = df.rename(columns={df.columns[0]: 'POI'})
    df = df.loc[~df['POI'].isin(['ASMSpotter-1-1-1', 'Landcover-1062091', 'UNHCR-HUNs002665'])]
    #reset index
    df = df.reset_index(drop=True)

    dataset = SatelliteDataset(df, root=data_root)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading all data to CPU memory...")
    all_data = load_all_data(data_loader)
    print(f"Data loaded. Shape: {all_data.shape}")

    nb_channels = all_data.shape[1]
    mean, std = calculate_mean_std_sequential(all_data, device, nb_channels)

    print("\nMean per channel:")
    print(mean.cpu().numpy())
    print("\nStd per channel:")
    print(std.cpu().numpy())

"""
Mean per channel:
[0.12543254, 0.13614272, 0.16135749, 0.17593418, 0.21157593, 0.26788434, 0.2906735, 0.2978914, 0.30594572, 0.33509994, 0.26702592, 0.20812556]

Std per channel:
[0.19563292, 0.19202784, 0.18203962, 0.18508251, 0.18341799, 0.16249457, 0.15715967, 0.15798034, 0.15200324, 0.22788922, 0.1411692, 0.13475654]
"""