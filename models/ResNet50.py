import torch
import torch.nn as nn
from torchvision import models
from models.blocks.ConvBlock import Block
from models.blocks.ResBlock import ResBlock


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        resnet = models.resnet50(pretrained=True)

        self.block1 = Block(3, 64)
        self.block2 = resnet.layer1
        self.block3 = resnet.layer2
        self.block4 = resnet.layer3
        self.block5 = resnet.layer4

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        return [x1, x2, x3, x4, x5]


class ResNetDecoder(nn.Module):
    def __init__(self):
        super(ResNetDecoder, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=2)
        self.block1 = ResBlock(4096, 2048)
        self.upconv2 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.block2 = ResBlock(2048, 1024)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.block3 = ResBlock(1024, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.block4 = ResBlock(512, 256)
        self.upconv5 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1) # Special 1x1 stride, lets see if it works
        self.block5 = ResBlock(128, 64)

    def forward(self, x, encoder_features):
        x = self.upconv1(x)
        x = torch.cat([x, encoder_features[4]], dim=1)
        x = self.block1(x)

        x = self.upconv2(x)
        x = torch.cat([x, encoder_features[3]], dim=1)
        x = self.block2(x)

        x = self.upconv3(x)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.block3(x)

        x = self.upconv4(x)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.block4(x)

        x = self.upconv5(x)
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.block5(x)

        return x


class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()

        self.encoder = ResNetEncoder()
        self.bridge = ResBlock(2048, 4096, strides=[2, 1])
        self.decoder = ResNetDecoder()
        self.projection = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        encoder_features = self.encoder(x)
        x = self.bridge(encoder_features[4])
        x = self.decoder(x, encoder_features)
        x = self.projection(x)

        return x