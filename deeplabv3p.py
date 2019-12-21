import torch
from torch import nn
from torch.nn import functional as F

def xception(x):
    data = None
    decode_shortcut = None
    return data, decode_shortcut

def mobilenetv2(x):
    conv1 = nn.Conv2d(3, 128, kernel_size=1, stride=32)
    conv2 = nn.Conv2d(3, 64, kernel_size=1, stride=8)
    data = conv1(x)
    decode_shortcut = conv2(x)
    return data, decode_shortcut

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

class ASPP1x1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPP1x1, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, seperate_conv=False):
        if seperate_conv:
            modules = SeperateConv(in_channels, out_channels, 3, dilation=dilation)
        else:
            modules = [
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPolling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPolling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for module in self:
            x = module(x)
        return F.interpolate(x, size, mode="bilinear", align_corners=False)

class Concat1x1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Concat1x1, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.9),
        )
    def forward(self, featuers):
        # featuers is a list of aspp's output
        x = torch.cat(featuers, dim=1)
        for module in self:
            x = module(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16, seperate_conv=False):
        super(Encoder, self).__init__()
        if output_stride == 16:
            aspp_ratios = [6, 12, 18]
        elif output_stride == 8:
            aspp_ratios = [12, 24, 36]
        else:
            raise ValueError("output_stride only support 16 or 8.")

        modules = []
        modules.append(ASPP1x1(in_channels, out_channels))
        for i in range(len(aspp_ratios)):
            modules.append(ASPPConv(in_channels, out_channels, aspp_ratios[i], seperate_conv))
        modules.append(ASPPPolling(in_channels, out_channels))
        self.aspp = nn.Sequential(*modules)

        self.concat1x1 = Concat1x1(out_channels * len(modules), out_channels)

    def forward(self, input):
        res = []
        for module in self.aspp:
            res.append(module(input))
        res = self.concat1x1(res)
        return res

class Concat3x3(nn.Sequential):
    def __init__(self, in_channels, out_channels, seperate_conv=False):
        if seperate_conv:
            modules = [
                SeperateConv(in_channels, out_channels, 3, dilation=1),
                SeperateConv(out_channels, out_channels, 3, dilation=1),
            ]
        else:
            modules = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        super(Concat3x3, self).__init__(*modules)
    def forward(self, tensors):
        res = torch.cat(tensors, dim=1)
        for module in self:
            res = module(res)
        return res

class Decoder(nn.Module):
    def __init__(self, sc_in, sc_out, enc_in, out_channels, seperate_conv=False):
        super(Decoder, self).__init__()
        self.conv_sc = nn.Sequential(
            nn.Conv2d(sc_in, sc_out, 1, bias=False),
            nn.BatchNorm2d(sc_out),
            nn.ReLU(inplace=True),
        )
        self.concat3x3 = Concat3x3(sc_out + enc_in, out_channels, seperate_conv)

    def forward(self, data, decode_shortcut):
        decode_shortcut = self.conv_sc(decode_shortcut)
        data = F.interpolate(data, decode_shortcut.shape[-2:], mode="bilinear", align_corners=False)
        res = self.concat3x3([decode_shortcut, data])
        return res

class Deeplabv3p(nn.Module):
    def __init__(self, num_class=22, output_stride=16,
                 backbone="mobilenet", seperate_conv=False):
        super(Deeplabv3p, self).__init__()
        if backbone == "xception":
            self.backbone = xception
        elif "mobilenet" in backbone:
            self.backbone = mobilenetv2
        else:
            raise ValueError("deeplabv3p just support backbone in xception or mobilenet,"
                             "got {}".format(backbone))
        self.encoder = Encoder(128, 256, output_stride=output_stride,
                               seperate_conv=seperate_conv)
        self.decoder = Decoder(64, 48, 256, 256, seperate_conv)
        self.last = nn.Conv2d(256, num_class, 1, padding=0, bias=True)

    def forward(self, x):
        data, decode_shortcut = self.backbone(x)
        data = self.encoder(data)
        res = self.decoder(data, decode_shortcut)
        res = self.last(res)
        res = F.interpolate(res, x.shape[-2:], mode="bilinear", align_corners=False)
        return res


if __name__ == '__main__':
    pass
