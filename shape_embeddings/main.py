import os
import argparse
import random
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from skimage.measure import label
from skimage.exposure import rescale_intensity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from model import PointNet
from criterion import DiscriminiativeLoss
from data_loading import get_celegans_loaders
import napari


def validation_loop(network, criterion, kmeans, val_loader):
    network.eval()
    X = []
    labels = []
    with torch.no_grad():
        accuracy = 0.0
        running_loss = 0.0
        for data in val_loader:
        #for data in tqdm.tqdm(val_loader):
            points = data[0]
            label = data[1]

            outputs = network(points)
            loss = criterion(outputs, label)

            with napari.gui_qt():
                viewer = napari.Viewer()
                viewer.add_points(points[0].numpy().T, size=0.5, face_color='red')
            print(label)

            X += [outputs.cpu().numpy()]
            labels += [label]
            running_loss += loss.item()
    labels = np.concatenate(labels)
    X = np.concatenate(X)
    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(X, labels):
        logistic_regr = LogisticRegression(C=1, multi_class='auto', solver='lbfgs')
        logistic_regr.fit(X[train_idx], labels[train_idx])
        score = logistic_regr.score(X[test_idx], labels[test_idx])
        scores.append(score)

        preds = logistic_regr.predict(X[test_idx])
        conf_matrix = metrics.confusion_matrix(labels[test_idx], preds)
        # print("The accuracy is {0}".format(score))
        # print(conf_matrix)

    scores = np.array(scores)

    # print('validation accuracy/loss: [%.3f, %.3f]' %
    #       (accuracy, running_loss / len(val_loader)))
    # print('valdiation accuracy: [%.3f]' % (np.mean(scores)))
    pred_labels = kmeans.predict(X)
    # accuracy = (pred_labels == labels).sum() / len(labels)
    coords1 = np.where(labels == 0)
    coords2 = np.where(labels == 1)
    acc1 = np.mean((pred_labels[coords1] == labels[coords1]))
    acc2 = np.mean((pred_labels[coords2] == labels[coords2]))

    print((acc1 + acc2) / 2, acc1, acc2)
    print('rand', metrics.adjusted_rand_score(labels, pred_labels))
    network.train()


def train_loop(network, criterion, optimizer, train_loader, val_loader):
    for epoch in range(1000):
        running_loss = 0.0
        i = 0
        d = 0
        embed = []
        for data in train_loader:
            points = data[0]
            labels = data[1]
            d += labels.sum()

            # zero parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = network(points)

            # loss + backward
            loss = criterion(outputs, labels)
            loss.backward()
            embed += [outputs.detach().numpy()]

            # optimizer update
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:
                # print('[%d, %d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            i+=1
        embed = np.concatenate(embed)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embed)
        validation_loop(network, criterion, kmeans, val_loader)


def main():
    # Data handling
    loaders = get_celegans_loaders(celegans_masks_dir,
                                   celegans_labels_dir,
                                   cells_masks_dir,
                                   nuclei_masks_dir,
                                   args.batch_size)
    train_loader, val_loader, test_loader = loaders

    # Model
    network = PointNet(channels=16, embed_dim=8, feature_transform=True)

    # criterion and optimizer
    criterion = DiscriminiativeLoss(0.5, 2.0)
    optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=0.0001)

    # train
    train_loop(network, criterion, optimizer, train_loader, val_loader)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    args = parser.parse_args()
    celegans_masks_dir = 'data/BBBC010/masks'
    celegans_labels_dir = 'data/BBBC010/labels'
    cells_masks_dir = 'data/BBBC020/BBC020_v1_outlines_cells'
    nuclei_masks_dir = 'data/BBBC020/BBC020_v1_outlines_nuclei'
    main()
