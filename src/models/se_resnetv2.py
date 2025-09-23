# https://github.com/hysts/pytorch_resnet_preact/blob/master/resnet_preact.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register
from src.models.base import ModelBase

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        if module.bias is not None:
            module.bias.data.zero_()
        

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dropout = 0.0
        ):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False)
            )

    def forward(self, x):
        x = F.relu(self.bn1(x), inplace=True)  # shortcut after preactivation
        out = self.conv1(x)

        out = F.relu(self.bn2(out), inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.se(out)

        out += self.shortcut(x)
        return out


class SE_ResNet_mini_v2(nn.Module):
    def __init__(self,
                 layers:List[int] = [2, 2, 2],
                 num_classes : int=10,
                 k = 4,
                 dropout = 0
        ):
        super(SE_ResNet_mini_v2, self).__init__()

        n_classes = num_classes
        base_channels = 16        

        n_channels = [
            base_channels * k,
            base_channels * 2 * k,
            base_channels * 4 * k,
        ]

        self.conv = nn.Conv2d(
            3,
            base_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)

        self.stage1 = self._make_stage(
            base_channels,
            n_channels[0],
            n_blocks=layers[0],
            stride=1,
            dropout = dropout)
        self.stage2 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks=layers[1],
            stride=2,
            dropout = dropout)
        self.stage3 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks=layers[2],
            stride=2,
            dropout = dropout)
        self.bn = nn.BatchNorm2d(n_channels[2])
        self.fc = nn.Linear(n_channels[2], n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride, dropout = 0.0):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    BasicBlock(
                        in_channels,
                        out_channels,
                        stride=stride,
                        dropout = dropout)
                )
            else:
                stage.add_module(
                    block_name,
                    BasicBlock(
                        out_channels,
                        out_channels,
                        stride=1,
                        dropout = dropout
                    )
                )
        return stage

    def forward(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(
            self.bn(x),
            inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@register("model", "se_resnet_mini_v2_28")
class SE_ResNet_mini_v2_28(SE_ResNet_mini_v2, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        super().__init__(layers=[3, 4, 6], k=4, **kwargs)

@register("model", "se_resnet_mini_v2_14x2")
class SE_ResNet_mini_v2_14x2(SE_ResNet_mini_v2, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        super().__init__(layers=[2, 2, 2], k=8, **kwargs)