# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from gnn_lib import GINConv, global_mean_pool as gmp


class NasBenchGINPredictorAgentCELU(nn.Module):
    def __init__(self, input_dim=6):
        super(NasBenchGINPredictorAgentCELU, self).__init__()
        layers = []
        dim = 32
        dim2 = 16
        nn1 = Sequential(Linear(input_dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim,  bias=True), ReLU(), Linear(dim, dim,  bias=True))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        #
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.linear_before = torch.nn.Linear(dim, dim2, bias=True)

        self.linear_mean = Linear(dim2, 1)
        layers.append(self.linear_mean)
        layers.append(self.linear_before)
        self.out_layer = torch.nn.Sigmoid()

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data, edge_index, batch, alpha=None):
        return self.forward_batch(data, edge_index, batch, alpha=alpha)

    def forward_batch(self, data, edge_index, batch, alpha=None):
        x1 = F.celu(self.conv1(data, edge_index))
        x1 = self.bn1(x1)

        x2 = F.celu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)

        x3 = F.celu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)

        x_embedding = gmp(x3, batch)
        x_embedding_mean = F.celu(self.linear_before(x_embedding))
        x_embedding_drop = F.dropout(x_embedding_mean, p=0.1, training=self.training)
        mean = self.linear_mean(x_embedding_drop)
        mean = self.out_layer(mean)
        return mean