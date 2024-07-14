import torch
import torch.nn as nn
from torchvision import models
from models.blocks.DecoderBlock import DecoderBlock
from models.blocks.DilationBlock import DilationBlock
from models.tools.norms import create_norm_layer
# follows the architecture described here: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8575492


class LinkNet(nn.Module):
    def __init__(self, use_frn=False):
        super(LinkNet, self).__init__()

        self.use_frn = use_frn
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

        self.decoder4 = DecoderBlock(filter_sizes[3], filter_sizes[2], self.use_frn)
        self.decoder3 = DecoderBlock(filter_sizes[2], filter_sizes[1], self.use_frn)
        self.decoder2 = DecoderBlock(filter_sizes[1], filter_sizes[0], self.use_frn)
        self.decoder1 = DecoderBlock(filter_sizes[0], filter_sizes[0], self.use_frn)

        self.fin_full_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.fin_bn1 = create_norm_layer(32, self.use_frn)
        self.fin_relu1 = nn.ReLU()

        self.fin_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fin_bn2 = create_norm_layer(32, self.use_frn)
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