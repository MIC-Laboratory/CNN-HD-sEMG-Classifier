import torch.nn as nn
from torchsummary import summary

class MobilenetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobilenetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
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
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
