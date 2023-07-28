import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def LinearBnRelu(input_dim, out_dim):
    linear_bn_relu = nn.Sequential(
        nn.Linear(input_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))
    return linear_bn_relu


class MAE(nn.Module):
    ''' Vanilla AE '''
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            LinearBnRelu(input_dim, 64),
            nn.Linear(64, 16))

        self.mem = MemoryModule(mem_dim=1000, fea_dim=16)

        self.decoder = nn.Sequential(
            LinearBnRelu(16, 64),
            nn.Linear(64, input_dim))

    def forward(self, x, mode):
        z = self.encoder(x)
        z = self.mem(z)
        output = self.decoder(z)

        return output, z


class MemoryModule(nn.Module):
    ''' Memory Module '''

    def __init__(self, mem_dim, fea_dim):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        # attention
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        self.reset_parameters()

    def reset_parameters(self):
        ''' init memory elements : Very Important !! '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # calculate attention weight
        att_weight = F.linear(x, self.weight)  # x:(N,∗,in_features) weight:(out_features,in_features)
        att_weight = F.softmax(att_weight, dim=1)
        # recovery x:(N,∗,in_features)
        mem_T = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_T)

        return output

