import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from supervised_contrastive.utils import Center, RandomCoordsFlip


def get_MEF_loaders(path_to_dataset, batch_size=10, transforms=None):
    # we asign label 0 to the wild-type and 1 to the lamin deficient cell
    mef_data = np.load(path_to_dataset, allow_pickle=True).item()
    data = []
    for key, pts in mef_data.items():
        label = 0 if 'wildtype' in key else 1
        data += [(label, pts.T)]
    random.shuffle(data)
    split_index = int(len(data) * 0.75)
    train_data = MEFDataset(data[:split_index])
    val_data = MEFDataset(data[split_index:])

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    return train_loader, val_loader


def get_MEF_loader(path_to_dataset, batch_size=100):
    mef_data = np.load(path_to_dataset, allow_pickle=True).item()
    data = []
    for key, pts in mef_data.items():
        label = 0 if 'wildtype' in key else 1
        data += [(label, pts.T)]
    return DataLoader(MEFDataset(data),
                      batch_size=batch_size)


class MEFDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()
        self.dataset = dataset
        self.base_transform = Center()
        self.transforms = transforms

    def apply_transforms(self, pts):
        for transform in self.transforms:
            pts = transform(pts)
        return pts

    def __getitem__(self, idx):
        label, pts = self.dataset[idx]
        pts = self.base_transform(pts)
        if self.transforms:
            pts = self.apply_transforms(pts)
        return (label, pts.astype(np.float32))

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    loaders = get_MEF_loaders('data/MEF_LMNA/mef_data.npy')
    for data in loaders[0]:
        import ipdb; ipdb.set_trace()
