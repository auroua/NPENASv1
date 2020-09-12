# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrtpi = 2.5066282746310002

    def forward(self, mu, sigma, target_y):
        """ mu : (bs, n_target)
            sigma : (bs, n_target)
            target_y : (bs, n_target)
        """
        l1 = 0.5*torch.pow((target_y - mu), 2)/torch.pow(sigma, 2)
        l3 = -1 * torch.log(self.sqrtpi*sigma)
        loss = -1 * torch.mean(l3 - l1)
        # loss2 = self.forward_test(mu, sigma, target_y)
        # print('the absolute error of two loss is %.10f' % (loss-loss2))
        return loss

    def forward_test(self, mu, sigma, target_y):
        loss = 0.0
        bs = mu.shape[0]
        for i in range(bs):
            # dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu[i],
            # covariance_matrix=torch.diag(sigma[i]))
            dist = torch.distributions.normal.Normal(loc=mu[i], scale=sigma[i])
            log_prob = dist.log_prob(target_y[i])
            loss = loss - 1.0 * torch.mean(log_prob)
        loss = loss / bs
        return loss