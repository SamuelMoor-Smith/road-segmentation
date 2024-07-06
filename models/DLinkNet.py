import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
# follows the architecture described here: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8575492


class DilationBlock(nn.Module):
    def __init__(self, channels):
        super(DilationBlock, self).__init__()

        self.dil1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.dil2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dil3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.dil4 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8)
        #self.dil5 = nn.Conv2d(channels, channels, kernel_size=3, padding=16, dilation=16)

    def forward(self, x):
        dil1 = F.relu(self.dil1(x))
        dil2 = F.relu(self.dil2(dil1))
        dil3 = F.relu(self.dil3(dil2))
        dil4 = F.relu(self.dil4(dil3))
        #dil5 = F.relu(self.dil5(dil4))

        out = x + dil1 + dil2 + dil3 + dil4 #+ dil5

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # batch norm and relu between all conv layers

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class LinkNet(nn.Module):
    def __init__(self):
        super(LinkNet, self).__init__()

        resnet = models.resnet34(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        filter_sizes = [64, 128, 256, 512]

        self.dilation_block = DilationBlock(filter_sizes[3])

        self.decoder4 = DecoderBlock(filter_sizes[3], filter_sizes[2])
        self.decoder3 = DecoderBlock(filter_sizes[2], filter_sizes[1])
        self.decoder2 = DecoderBlock(filter_sizes[1], filter_sizes[0])
        self.decoder1 = DecoderBlock(filter_sizes[0], filter_sizes[0])

        self.fin_full_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.fin_bn1 = nn.BatchNorm2d(32)
        self.fin_relu1 = nn.ReLU()

        self.fin_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fin_bn2 = nn.BatchNorm2d(32)
        self.fin_relu2 = nn.ReLU()

        self.fin_full_conv_3 = nn.ConvTranspose2d(32, 1, kernel_size=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        dil_out = self.dilation_block(e4)

        d4 = self.decoder4(dil_out)
        d3 = self.decoder3(d4 + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)

        x = self.fin_full_conv1(d1)
        x = self.fin_bn1(x)
        x = self.fin_relu1(x)

        x = self.fin_conv2(x)
        x = self.fin_bn2(x)
        x = self.fin_relu2(x)

        x = self.fin_full_conv_3(x)

        x = torch.sigmoid(x)

        return x