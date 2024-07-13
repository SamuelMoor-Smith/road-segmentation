import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=None):
        super(ResBlock, self).__init__()

        if strides is None:
            strides = [1, 1]  # Only the bridge has a different stride (See table page 3 of paper)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides[0], padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=1)

        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides[0]),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.residual_connection(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += residual

        return x
