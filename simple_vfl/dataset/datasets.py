import torch
from torch.utils.data import Dataset
import tifffile as tiff
import os
from functools import lru_cache
from torchvision import transforms
import numpy as np
import random


@lru_cache(maxsize=8192)
def _load_feature_file(path):
    """Load a flattened feature vector from a .npy file and cache it in memory."""
    features = np.load(path).flatten()
    return torch.tensor(features, dtype=torch.float32)


class CreditDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = [torch.tensor(self.data[idx,:11], dtype=torch.float32), torch.tensor(self.data[idx,11:], dtype=torch.float32)]
        label =torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

class SatelliteDataset(Dataset):
    """
    SatelliteDataset is the base class for all satellite datasets
    """
    def __init__(self, dataset_dict, metadata, k = 1):
        self.root = dataset_dict.data_root
        # Optional feature folders. None means raw images are used instead.
        # SPOT and Sentinel features are extracted into different folders, hence the two keys.
        self.feature_instead = dataset_dict.get('features_instead', None)
        self.spot_features_instead = dataset_dict.get('spot_features_instead', None) or self.feature_instead
        self.flips = dataset_dict.get('flips', None)
        self.rgb = dataset_dict.get('rgb', None)
        self.metadata = metadata
        self.k = k
        self.dataset_name = dataset_dict.dataset_name

        transform_list = []
        if self.flips:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
        
        self.transform = transforms.Compose(transform_list) if transform_list else None

    def __len__(self):
        return len(self.metadata)

    def _load_feature(self, path):
        return _load_feature_file(os.path.normpath(path))

    def _load_image(self, path):
        image = torch.tensor(tiff.imread(path), dtype=torch.float32)
        image = image.permute(2, 0, 1)
        if self.rgb:
            image = image[1:4, :, :]
        return image

    def _apply_transform(self, image):
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_spot_image(self, poi):
        if self.spot_features_instead is not None:
            path = os.path.join(self.root, 'satellite_dataset', self.spot_features_instead, poi, f"{poi}_ps_class_token.npy")
            image = self._load_feature(path)
        else:
            path = os.path.join(self.root, 'satellite_dataset', 'hr_dataset', poi, f"{poi}_ps.tiff")
            image = self._load_image(path)
        return self._apply_transform(image)
    
    def get_sentinel_image(self, poi, n):
        if self.feature_instead is not None:
            path = os.path.join(self.root, 'satellite_dataset', self.feature_instead, poi, 'L2A', f"{poi}-{n}-L2A_data_class_token.npy")
            image = self._load_feature(path)
        else:
            path = os.path.join(self.root, 'satellite_dataset', 'lr_dataset', poi, 'L2A', f"{poi}-{n}-L2A_data.tiff")
            image = self._load_image(path)
        return self._apply_transform(image)

class SentinelDataset(SatelliteDataset):
    """
    child of SatelliteDataset
    For each POI, it returns the k first images of the Sentinel dataset as a list of tensors
    """
    def __init__(self, dataset_dict, metadata, k):
        super().__init__(dataset_dict, metadata, k)

    def __getitem__(self, idx):
        poi = self.metadata.iloc[idx]['POI']
        label = self.metadata.iloc[idx]['label']
        images = []
        for n in range(1, self.k + 1):
            images.append(self.get_sentinel_image(poi, n))
        return images, label

class SPOTDataset(SatelliteDataset):
    """
    child of SatelliteDataset
    For each POI, it returns the SPOT image as a list of only one tensor
    We return it as a list to be consistent with the other datasets
    """
    def __init__(self, dataset_dict, metadata, k = 1):
        super().__init__(dataset_dict, metadata, k)

    def __getitem__(self, idx):
        poi = self.metadata.iloc[idx]['POI']  # Get the POI corresponding to the index
        label = self.metadata.iloc[idx]['label']
        image = self.get_spot_image(poi)
        return [image], label
    
class SPOTAndSentinelDataset(SatelliteDataset):
    """
    child of SatelliteDataset
    For each POI, it returns the SPOT image and all the Sentinel images as a list of 17 tensors.
    SPOT and Sentinel features are read from `spot_features_instead` and `features_instead` respectively.
    """
    def __init__(self, dataset_dict, metadata, k = 17):
        super().__init__(dataset_dict, metadata, k)

    def __getitem__(self, idx):
        poi = self.metadata.iloc[idx]['POI']
        label = self.metadata.iloc[idx]['label']
        images = []
        images.append(self.get_spot_image(poi))
        for n in range(1, 17):
            images.append(self.get_sentinel_image(poi, n))
        return images, label

class HorizontalSentinelDataset(SatelliteDataset):
    """
    child of SatelliteDataset
    For each POI, it returns a random Sentinel image as a list of only one tensor
    It simulates a horizontal federated learning setting
    """
    def __init__(self, dataset_dict, metadata, k):
        super().__init__(dataset_dict, metadata, k)
    
    def __getitem__(self, idx):
        poi = self.metadata.iloc[idx]['POI']
        n = random.randint(1, 16)
        label = self.metadata.iloc[idx]['label']
        image = self.get_sentinel_image(poi, n)
        return [image], label

class HorizontalSPOTAndSentinelDataset(SatelliteDataset):
    """
    child of SatelliteDataset
    For each POI, it returns a random image from the 16 Sentinel images and the SPOT image
    It simulates a horizontal federated learning setting where SPOT is not considered differently from the Sentinel images
    """
    def __init__(self, dataset_dict, metadata, k):
        super().__init__(dataset_dict, metadata, k)
    
    def __getitem__(self, idx):
        poi = self.metadata.iloc[idx]['POI']
        label = self.metadata.iloc[idx]['label']
        n = random.randint(0, 16)
        if n == 0:
            image = self.get_spot_image(poi)
        else:
            image = self.get_sentinel_image(poi, n)
        return [image], label

class VertibenchData(Dataset):
    """
    VertibenchData is the base class for all vertibench datasets
    For each POI, it returns the k first images of the vertibench dataset as a list of tensors
    """
    def __init__(self, args, split):
        features_instead = args.dataset.get('features_instead', None)
        self.data_path = os.path.normpath(os.path.join(args.dataset.data_root, 'satellite_dataset', features_instead or 'Vertibench'))
        self.k = args.k
        self.split = split
        self.ys = np.load(os.path.join(self.data_path, f'label_{self.split}.npy'))
        self.ys = torch.from_numpy(self.ys).long()
        
    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, index):
        Xs = []
        for i in range(self.k):
            Xs.append(np.load(os.path.join(self.data_path, f'satellite_party{i}_{self.split}_{index}.npy')))
        return [torch.from_numpy(X).float() for X in Xs], self.ys[index]

class VertibenchHorizontal(VertibenchData):
    """
    child of VertibenchData
    For each POI, it returns a random image from the k vertibench images
    It simulates a horizontal federated learning setting
    """
    def __init__(self, args, split):
        super().__init__(args, split)
        
    def __getitem__(self, index):
        i = random.randint(0, self.k-1)
        X = np.load(os.path.join(self.data_path, f'satellite_party{i}_{self.split}_{index}.npy'))
        return [torch.from_numpy(X).float()], self.ys[index]
    
class VertibenchIsolated(VertibenchData):
    """
    child of VertibenchData
    for each POI, it always returns the image from its argument party_id
    It is an isolated setting, where a party trains only on its own data
    """
    def __init__(self, args, split):
        super().__init__(args, split)
        self.party_id = args.dataset.party_id
    
    def __getitem__(self, index):
        Xs = []
        Xs.append(np.load(os.path.join(self.data_path, f'satellite_party{self.party_id}_{self.split}_{index}.npy')))
        return [torch.from_numpy(X).float() for X in Xs], self.ys[index]
