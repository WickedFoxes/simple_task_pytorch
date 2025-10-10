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
        module.bias.data.zero_()

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

        out += self.shortcut(x)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float = 0.0
    ):
        super(Bottleneck, self).__init__()

        mid = out_channels  # 축소/복원 전의 중간 채널 수
        out_expanded = out_channels * self.expansion  # 최종 채널 수

        # Pre-activation blocks
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            mid,
            kernel_size=1,
            stride=1,         # 채널 축소
            padding=0,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(
            mid,
            mid,
            kernel_size=3,
            stride=stride,    # 다운샘플은 여기에서
            padding=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(
            mid,
            out_expanded,
            kernel_size=1,
            stride=1,         # 채널 확장
            padding=0,
            bias=False
        )

        self.dropout = nn.Dropout(p=dropout)

        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_expanded or stride != 1:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_expanded,
                    kernel_size=1,
                    stride=stride,  # 본문과 동일 stride로 다운샘플
                    padding=0,
                    bias=False
                )
            )

    def forward(self, x):
        # pre-activation
        x_preact = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(x_preact)

        out = F.relu(self.bn2(out), inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)

        out = F.relu(self.bn3(out), inplace=True)
        out = self.conv3(out)

        out += self.shortcut(x)
        return out


class ResNet_mini_v2(nn.Module):
    def __init__(self,
                 layers: List[int] = [2, 2, 2],
                 num_classes: int = 10,
                 k: int = 4,
                 dropout: float = 0.0,
                 block = BasicBlock
                 ):
        super(ResNet_mini_v2, self).__init__()

        self.block_cls = block
        self.expansion = getattr(self.block_cls, "expansion", 1)

        n_classes = num_classes
        base_channels = 16

        # planes: 각 stage에서의 '기준 채널 수(planes)'. 실제 블록 출력 채널은 planes * expansion
        planes = [
            base_channels * k,
            base_channels * 2 * k,
            base_channels * 4 * k,
        ]

        # Stem
        self.conv = nn.Conv2d(
            3,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Stages
        self.stage1 = self._make_stage(
            in_channels=base_channels,
            planes=planes[0],
            n_blocks=layers[0],
            stride=1,
            dropout=dropout
        )
        self.stage2 = self._make_stage(
            in_channels=planes[0] * self.expansion,
            planes=planes[1],
            n_blocks=layers[1],
            stride=2,
            dropout=dropout
        )
        self.stage3 = self._make_stage(
            in_channels=planes[1] * self.expansion,
            planes=planes[2],
            n_blocks=layers[2],
            stride=2,
            dropout=dropout
        )

        final_channels = planes[2] * self.expansion
        self.bn = nn.BatchNorm2d(final_channels)
        self.fc = nn.Linear(final_channels, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels: int, planes: int, n_blocks: int, stride: int, dropout: float = 0.0):
        """
        planes: 본문 conv의 기준 채널(예: Bottleneck의 1x1/3x3에서 쓰일 mid 채널).
        해당 블록의 출력 채널 = planes * expansion
        """
        stage = nn.Sequential()

        # 첫 블록: stride 적용 및 in/out 매칭
        stage.add_module(
            'block1',
            self.block_cls(
                in_channels=in_channels,
                out_channels=planes,   # block 내부에서 expansion 적용
                stride=stride,
                dropout=dropout
            )
        )

        # 이후 블록들: 입력 채널은 직전 블록 출력(planes * expansion), stride=1
        for i in range(1, n_blocks):
            stage.add_module(
                f'block{i+1}',
                self.block_cls(
                    in_channels=planes * self.expansion,
                    out_channels=planes,
                    stride=1,
                    dropout=dropout
                )
            )

        return stage

    def forward(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)  # pre-activation 스타일 마감
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@register("model", "resnet_mini_v2_28")
class ResNet_mini_v2_28(ResNet_mini_v2, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        super().__init__(layers=[3, 4, 6], k=4, block = BasicBlock, **kwargs)

@register("model", "resnet_mini_v2_14x2")
class ResNet_mini_v2_14x2(ResNet_mini_v2, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        super().__init__(layers=[2, 2, 2], k=8, block = BasicBlock, **kwargs)


@register("model", "resnet_mini_v2_41")
class ResNet_mini_v2_41(ResNet_mini_v2, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        super().__init__(layers=[3, 4, 6], k=4, block = Bottleneck, **kwargs)

@register("model", "resnet_mini_v2_28x2")
class ResNet_mini_v2_28x2(ResNet_mini_v2, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        super().__init__(layers=[3, 4, 6], k=8, block = BasicBlock, **kwargs)