import torch
import torch.nn as nn
from torchvision import models
# follows https://arxiv.org/pdf/1711.10684


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.residual_connection(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += residual

        return x


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


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        resnet = models.resnet34(pretrained=True)

        self.block1 = Block(3, 64)
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        return [x1, x2, x3]


class ResNetDecoder(nn.Module):
    def __init__(self):
        super(ResNetDecoder, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.block1 = ResBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.block2 = ResBlock(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block3 = ResBlock(128, 64)

    def forward(self, x, encoder_features):
        x = self.upconv1(x)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.block1(x)

        x = self.upconv2(x)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.block2(x)

        x = self.upconv3(x)
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.block3(x)

        return x


class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()

        self.encoder = ResNetEncoder()
        self.bridge = ResBlock(256, 512, strides=[2, 1])
        self.decoder = ResNetDecoder()
        self.projection = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        x = self.bridge(encoder_features[2])
        x = self.decoder(x, encoder_features)
        x = nn.Sequential(self.projection(x),
                          nn.Sigmoid())

        return x







