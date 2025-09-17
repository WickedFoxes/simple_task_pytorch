# https://github.com/hysts/pytorch_resnet_preact/blob/master/resnet_preact.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
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
        out = self.conv2(out)

        out += self.shortcut(x)
        return out
    


class ResNet_mini_v2(nn.Module):
    def __init__(self,
                 layers:List[int] = [2, 2, 2],
                 num_classes : int=10,
                 k = 4):
        super(ResNet_mini_v2, self).__init__()

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
            stride=1)
        self.stage2 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks=layers[1],
            stride=2)
        self.stage3 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks=layers[2],
            stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])
        self.fc = nn.Linear(n_channels[2], n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    BasicBlock(
                        in_channels,
                        out_channels,
                        stride=stride)
                )
            else:
                stage.add_module(
                    block_name,
                    BasicBlock(
                        out_channels,
                        out_channels,
                        stride=1,
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