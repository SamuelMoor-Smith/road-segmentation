import torch.nn as nn
import torch.nn.functional as F


class DilationBlock(nn.Module):
    def __init__(self, channels):
        super(DilationBlock, self).__init__()

        self.dil1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.dil2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dil3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.dil4 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8)
        #self.dil5 = nn.Conv2d(channels, channels, kernel_size=3, padding=16, dilation=16)
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #        if m.bias is not None:
        #            m.bias.data.zero_()

    def forward(self, x):
        dil1 = F.relu(self.dil1(x))
        dil2 = F.relu(self.dil2(dil1))
        dil3 = F.relu(self.dil3(dil2))
        dil4 = F.relu(self.dil4(dil3))
        #dil5 = F.relu(self.dil5(dil4))

        out = x + dil1 + dil2 + dil3 + dil4 #+ dil5

        return out