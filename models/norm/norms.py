import torch
import torch.nn as nn


def create_norm_layer(out_channels, use_frn):
    if use_frn:
        return FRN(out_channels)
    else:
        return nn.BatchNorm2d(out_channels)


class FRN(nn.Module):
    # follows https://www.sciencedirect.com/science/article/pii/S0924271621000873?casa_token=mvUvU9gfe40AAAAA:Q2bmM0SfriQfHXiZ7HC821yrQrE7yqiacFk3Sw1NJs3s-EYQjWPHBEWq1w4SqOfCGAVH2zuXXm9c#s0035
    # they came to conclusion that batch norm on low batch size (which we have) is unstable training (which I also noticed)
    def __init__(self, num_features, eps=1e-6):
        super(FRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.tau = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
        return torch.max(y, self.tau.view(1, -1, 1, 1))
