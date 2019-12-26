import torch.nn as nn
import torch.nn.functional as F

def check_data(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data

class Xception(nn.Module):
    def __init__(self, backbone, output_stride, decode_point, end_points):
        self.backbone = backbone
        self.gen_bottleneck_param()
        self.stride = 2
        self.block_point = 0
        self.output_stride = output_stride
        self.decode_point = decode_point
        self.end_points = end_points

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

    def gen_bottleneck_param(self):
        if self.backbone == 65:
            bottleneck_param = {
                "entry_flow": None
            }
        elif self.backbone == 41:
            bottleneck_param = {
                "entry_flow": (3, [2, 2, 2], [128, 256, 728]),
                "middle_flow": (8, 1, 728),
                "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536, 2048]])
            }
        elif self.backbone == 71:
            pass
        else:
            raise Exception("just balabala")
        self.bottleneck_param = bottleneck_param

    def entry_flow(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        block_num = self.bottleneck_param["entry_flow"][0]
        strides = self.bottleneck_paramp["entry_flow"][1]
        chnls = self.bottleneck_param["entry_flow"][2]
        strides = check_data(strides, block_num)
        chnls = check_data(chnls, block_num)




    def middle_flow(self, x):
        pass

    def exit_flow(self, x):
        pass

    def forward(self, x):
        x = self.entry_flow(x)


if __name__ == '__main__':
    Xception()