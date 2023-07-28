import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, BatchNorm  # noqa


class GAAE(torch.nn.Module):
    def __init__(self, input_dim):
        super(GAAE, self).__init__()

        self.GConv1 = ChebConv(input_dim, 64, K=2)
        self.bn1 = BatchNorm(64)
        self.GConv2 = ChebConv(64, 16, K=2)
        self.GConv3 = ChebConv(16, 64, K=2)
        self.bn3 = BatchNorm(64)
        self.GConv4 = ChebConv(64, input_dim, K=2)

    def forward(self, dataset_num, data, mode):
        x, edge_index = data.x, data.edge_index

        if mode in 'train':
            noise = dataset_num*torch.randn(x.shape).cuda() * x.mean()
            x = x + noise
        x1 = self.GConv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.GConv2(x1, edge_index)

        x3 = self.GConv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x4 = self.GConv4(x3, edge_index)

        return x4


