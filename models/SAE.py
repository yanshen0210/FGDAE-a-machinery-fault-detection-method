import torch.nn as nn
import torch


def LinearBnRelu(input_dim, out_dim):
    linear_bn_relu = nn.Sequential(
        nn.Linear(input_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))
    return linear_bn_relu


class SAE(nn.Module):
    ''' Vanilla AE '''
    def __init__(self, input_dim):
        super(SAE, self).__init__()

        self.encoder = nn.Sequential(
            LinearBnRelu(input_dim, 64),
            nn.Linear(64, 16))

        self.decoder = nn.Sequential(
            LinearBnRelu(16, 64),
            nn.Linear(64, input_dim))

    def forward(self, x, mode):
        z = self.encoder(x)
        output = self.decoder(z)

        return output, z
