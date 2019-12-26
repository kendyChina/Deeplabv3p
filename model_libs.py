import torch.nn as nn

class SeperateConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1):
        super(SeperateConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      groups=in_channels, bias=False,
                      padding=(kernel_size // 2) * dilation,
                      dilation=dilation),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      groups=1, bias=False, dilation=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )