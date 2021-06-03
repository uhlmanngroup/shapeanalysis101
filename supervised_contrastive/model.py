import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(
            np.array([1,0,0,1]).astype(np.float32)
            ).view(1, 4).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 2, 2)
        return x


class SpatialTransformerkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(
            np.eye(self.k).flatten().astype(np.float32)
        ).view(1, self.k*self.k).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super().__init__()
        self.transformer = SpatialTransformer3d()
        if feature_transform:
            self.feat_transformer = SpatialTransformerkd()

        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat
        self.feat_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.transformer(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feat_transform:
            trans_feat = self.feat_transformer(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
    def __init__(self,
                 head='mlp',
                 embed_dim=256,
                 feat_dim=64,
                 feature_transform=False):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=True,
                                 feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, embed_dim)

        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

        if head == 'linear':
            self.head = nn.Linear(embed_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, feat_dim)
            )

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        out = F.normalize(self.head(x), dim=1)
        return out, x
