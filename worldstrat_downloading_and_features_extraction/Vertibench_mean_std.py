"""
This script calculates the mean and standard deviation of the Vertibench dataset. 
It loads all data in GPU memory and then calculates the mean and std, channel after channel.
It takes a lot of GPU memory but it is still doable on a 80 Gb GPU.
"""


import argparse
import multiprocessing as mp
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __name__ == "__main__":
    mp.set_start_method('spawn')

class VertibenchData(Dataset):
    def __init__(self, data_path, parties, split):
        self.data_path = data_path
        self.parties = parties
        self.split = split
        self.ys = np.load(os.path.join(self.data_path, f'label_{self.split}.npy'))
        self.ys = torch.from_numpy(self.ys).long()
        
    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, index):
        Xs = []
        for i in range(self.parties):
            Xs.append(np.load(os.path.join(self.data_path, f'satellite_party{i}_{self.split}_{index}.npy')))
        return [torch.from_numpy(X).float() for X in Xs], self.ys[index]

def load_all_data(data_loader):
    all_data = []
    total_samples = len(data_loader.dataset)
    
    progress_bar = tqdm(total=total_samples, desc="Loading all data to CPU")
    for batch in data_loader:
        batch = torch.cat(batch[0], dim=0)
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
        description="Compute the per-channel mean and standard deviation of the Vertibench dataset."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="Folder containing the Vertibench files (label_train.npy, "
             "satellite_party*_*.npy), i.e. '.../satellite_dataset/Vertibench' "
             "(default: $DATA_ROOT, which must point to that folder).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of samples per batch (default: 256).",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error(
            "--data-root is required (or set the DATA_ROOT environment variable)."
        )

    parties = 16
    data_path = os.path.normpath(os.path.expanduser(args.data_root))
    nb_channels = 13
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = VertibenchData(data_path, parties, split='train')
    test_data = VertibenchData(data_path, parties, split='test')

    combined_data = torch.utils.data.ConcatDataset([train_data, test_data])
    data_loader = DataLoader(combined_data, batch_size=batch_size, num_workers=4, pin_memory=True)

    print(f"Total samples: {len(combined_data)}")
    print(f"Number of batches: {len(data_loader)}")

    print("Loading all data to CPU memory...")
    all_data = load_all_data(data_loader)
    print(f"Data loaded. Shape: {all_data.shape}")

    mean, std = calculate_mean_std_sequential(all_data, device, nb_channels)

    print("\nMean per channel:")
    print(mean.cpu().numpy())
    print("\nStd per channel:")
    print(std.cpu().numpy())
