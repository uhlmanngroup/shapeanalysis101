import os
import csv
import random
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
from data_utils import load_data, Center, RandomCoordsFlip


def get_celegans_loaders(masks_dir, labels_dir, batch_size=10):
    data = load_data(masks_dir, labels_dir)

    idx_1 = int(len(data) * 0.8)
    idx_2 = int(len(data) * 0.9)
    train_data = CelegansDataset(data[:idx_1])
    val_data = CelegansDataset(data[idx_1: idx_2])
    test_data = CelegansDataset(data[idx_2:])

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=1,
                            shuffle=False)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False)
    return train_loader, val_loader, test_loader


class CelegansDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.center = Center()
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = [RandomCoordsFlip()]
        return transforms

    def __getitem__(self, idx):
        data = self.dataset[idx]
        label = data['label']
        points = data['points']
        indices = np.random.choice(points.shape[0],
                                   size=200,
                                   replace=False)
        points = points[indices].astype(np.float32)

        transform = random.sample(self.transforms, 1)[0]
        points = transform(self.center(points))
        points = points.T

        return (points, label)

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    import napari
    masks_dir = 'data/BBBC010/masks'
    images_dir = 'data/BBBC010/images'
    labels_dir = 'data/BBBC010/labels'

    loaders = get_celegans_loaders(masks_dir, labels_dir)
    train_loader, val_loader, test_loader = loaders

    for data in train_loader:
        points = data[0]
        labels = data[1]
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_points(points[0].transpose(0, 1), size=0.5, face_color='red')
