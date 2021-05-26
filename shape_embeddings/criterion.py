import torch
import torch.nn as nn


class DiscriminiativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def compute_means(inputs, masks):
        masked_pred = inputs.unsqueeze(0) * masks.unsqueeze(1)
        means = masked_pred.sum(2) \
                / (masks.sum(1).unsqueeze(-1) + 1e-8)
        return means

    @staticmethod
    def compute_variance_loss(inputs, masks, means, delta_v):
        masked_pred = inputs.unsqueeze(0) * masks.unsqueeze(1)
        num_instances, embd_dim, num_shapes = masked_pred.shape
        loss = 0.0
        dst_to_mean = (means[:, :, None] - masked_pred[:, :, :]) * masks.unsqueeze(1)
        margin = torch.clamp(
            torch.norm(dst_to_mean, dim=1) - delta_v, min=0.0
        ) ** 2
        margin = margin.sum(1) / (masks.sum(1) + 1e-8)
        loss = margin.sum() / num_instances
        return loss

    @staticmethod
    def compute_distance_loss(means, delta_d):
        num_instances, embed_dim = means.shape
        loss = 0.0
        diff_tensor = means.unsqueeze(0) - means.unsqueeze(1)
        margin = torch.clamp(
                2 * delta_d - torch.norm(diff_tensor, dim=2), min=0.0
        ) ** 2
        loss = 2 * torch.triu(margin, diagonal=1).sum() / (num_instances * (num_instances - 1))
        return loss

    @staticmethod
    def compute_regression_loss(means):
        num_instances, embed_dim = means.shape
        loss = means.norm(dim=1).sum() / (num_instances + 1e-8)
        return loss


    def forward(self, inputs, targets):
        # inputs [batch_size, embed_dim]
        # targets [batch_size]
        inputs = inputs.transpose(0, 1)
        labels = torch.unique(targets)
        targets = targets.view(-1, 1)
        batchsize = targets.size()[0]
        num_instances = batchsize
        masks = targets == labels[None, :]
        masks = masks.transpose(0, 1)
        means = self.compute_means(inputs, masks)
        loss_var = self.compute_variance_loss(inputs, masks, means, self.delta_v)
        loss_dist = self.compute_distance_loss(means, self.delta_d)
        loss_reg = self.compute_regression_loss(means)
        # print(loss_var, loss_dist, loss_reg)
        return self.alpha * loss_var + self.beta * loss_dist + self.gamma * loss_reg


# Test
if __name__ == '__main__':
    import time
    import numpy as np

    batchsize = 12
    x = torch.randn(batchsize, 16)
    mask = torch.randint(0, 2, (batchsize, ))

    criterion = DiscriminiativeLoss(0.5, 1.5)
    t = time.time()
    loss = criterion(x, mask)
    print('time:', time.time() - t)
    print(loss)
