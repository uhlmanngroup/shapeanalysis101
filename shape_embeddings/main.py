import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
import napari
from torch.utils.data import Dataset, DataLoader
from skimage.measure import label
from skimage.exposure import rescale_intensity
from model import PointNet
from criterion import DiscriminiativeLoss
from data_loading import get_celegans_loaders


def validation_loop(network, criterion, val_loader):
    with torch.no_grad():
        accuracy = 0.0
        running_loss = 0.0
        for raw_cells, labels in val_loader:
            outputs = network(raw_cells)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred_label = 0 if outputs < .5 else 1
            accuracy += int(pred_label == labels)
    print('validation accuracy/loss: [%.3f, %.3f]' %
          (accuracy / len(val_loader), running_loss / len(val_loader)))


def train_loop(network, criterion, optimizer, train_loader, val_loader):
    for epoch in range(100):
        running_loss = 0.0
        i = 0
        for data in train_loader:
            points = data[0]
            labels = data[1]

            # zero parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = network(points)

            # loss + backward
            loss = criterion(outputs, labels)
            loss.backward()
            print(loss)

            # optimizer update
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            i+=1


def main():

    # Data handling
    loaders = get_celegans_loaders(masks_dir, labels_dir, args.batch_size)
    train_loader, val_loader, test_loader = loaders

    # Model
    network = PointNet(k=16, feature_transform=True)

    # criterion and optimizer
    criterion = DiscriminiativeLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.005)

    # train
    train_loop(network, criterion, optimizer, train_loader, val_loader)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    args = parser.parse_args()
    masks_dir = 'data/BBBC010/masks'
    images_dir = 'data/BBBC010/images'
    labels_dir = 'data/BBBC010/labels'
    main()
