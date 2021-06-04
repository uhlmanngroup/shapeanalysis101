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


# Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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
