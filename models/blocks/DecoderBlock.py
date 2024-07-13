import torch.nn as nn
from models.norm.norms import create_norm_layer


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_frn=False):
        super(DecoderBlock, self).__init__()

        # batch norm and relu between all conv layers

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.norm1 = create_norm_layer(in_channels // 4, use_frn)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = create_norm_layer(in_channels // 4, use_frn)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        self.norm3 = create_norm_layer(out_channels, use_frn)
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
