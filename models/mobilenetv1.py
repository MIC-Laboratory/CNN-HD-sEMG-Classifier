import torch
import torch.nn as nn
from torchsummary import summary
import math
class MobilenetV1(nn.Module):
    def __init__(self, ch_in, n_classes,Global_ratio):
        super(MobilenetV1, self).__init__()
        self.Global_ratio = Global_ratio
        def conv_bn(inp, oup, stride):
            oup = math.trunc(oup*self.Global_ratio)+1 if math.trunc(oup*self.Global_ratio) == 0 else math.trunc(oup*self.Global_ratio)
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            inp = math.trunc(inp*self.Global_ratio)+1 if math.trunc(inp*self.Global_ratio) == 0 else math.trunc(inp*self.Global_ratio)
            oup = math.trunc(oup*self.Global_ratio)+1 if math.trunc(oup*self.Global_ratio) == 0 else math.trunc(oup*self.Global_ratio)
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(math.trunc(1024*self.Global_ratio), n_classes)
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, math.trunc(1024*self.Global_ratio))
        x = self.fc(x)
        return x


