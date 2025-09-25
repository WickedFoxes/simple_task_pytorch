# REF)
# https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py 
# https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register
from src.models.base import ModelBase
import math

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
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


class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SEBlock(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SEBlock(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def _make_divisible(
        v, 
        divisor:int=8, 
        min_value=None
    ):
    """
    width multiplier로 채널 수를 조절할 때,
    8의 배수가 되도록 보장하는 함수
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=100, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1)] # stride 1로 수정


        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:  # ✅ bias 존재 여부 확인
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:  # ✅ 추가 확인
                    m.bias.data.zero_()


@register("model", "mobilenet_v3_small")
class mobilenet_v3_small(MobileNetV3, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        cfgs = [
            # k, t, c, SE, HS, s 
            [3,    1,  16, 1, 0, 2], 
            [3,  4.5,  24, 0, 0, 2],
            [3, 3.67,  24, 0, 0, 1], # exp size=88(16*3.67), out_ch=16, SE=false, NL=Relu, stride=1
            [5,    4,  40, 1, 1, 2],
            [5,    6,  40, 1, 1, 1],
            [5,    6,  40, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    6,  96, 1, 1, 2],
            [5,    6,  96, 1, 1, 1],
            [5,    6,  96, 1, 1, 1],
        ]
        super().__init__(cfgs=cfgs, mode='small', **kwargs)


@register("model", "mobilenet_v3_large")
class mobilenet_v3_large(MobileNetV3, ModelBase):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the fixed layers
        cfgs = [
            # k, t, c, SE, HS, s 
            [3,   1,  16, 0, 0, 1],
            [3,   4,  24, 0, 0, 1],
            [3,   3,  24, 0, 0, 1],
            [5,   3,  40, 1, 0, 2],
            [5,   3,  40, 1, 0, 1], # kernel=3, expand_size=120, out_ch=40, SE=True, NL=Relu, stride=1
            [5,   3,  40, 1, 0, 1],
            [3,   6,  80, 0, 1, 1],
            [3, 2.5,  80, 0, 1, 1],
            [3, 2.3,  80, 0, 1, 1],
            [3, 2.3,  80, 0, 1, 1],
            [3,   6, 112, 1, 1, 1],
            [3,   6, 112, 1, 1, 1],
            [5,   6, 160, 1, 1, 2],
            [5,   6, 160, 1, 1, 1],
            [5,   6, 160, 1, 1, 1]
        ]
        super().__init__(cfgs=cfgs, mode='large', **kwargs)

import torchvision
@register("model", "mobilenet_v3_large_pretrained")
class mobilenet_v3_large_pretrained(ModelBase):
    def __init__(self, num_classes=100, **kwargs):
        super().__init__(**kwargs)

        # 사전학습된 MobileNetV3-Large 불러오기
        self.model = torchvision.models.mobilenet_v3_large(
            weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # classifier (마지막 FC 레이어) 수정 가능
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

@register("model", "efficientnet_b0_mini")
class efficientnet_b0_mini(MobileNetV3, ModelBase):
    def __init__(self, **kwargs):
        cfgs = [
            # k, t, c, SE, HS, s 
            [3,   1,  16, 1, 0, 1],
            [3,   2,  24, 1, 0, 1],
            [3,   2,  24, 1, 0, 1],
            [5,   2,  40, 1, 0, 2],
            [5,   2,  40, 1, 0, 1], # kernel=3, expand_size=120, out_ch=40, SE=True, NL=Relu, stride=1
            [3,   2,  80, 1, 0, 1],
            [3,   2,  80, 1, 0, 1],
            [3,   2,  80, 1, 0, 1],
            [5,   2, 112, 1, 0, 1],
            [5,   2, 112, 1, 0, 1],
            [5,   2, 112, 1, 0, 1],
            [5,   2, 192, 1, 0, 2],
            [5,   2, 192, 1, 0, 1],
            [5,   2, 320, 1, 0, 1]
        ]
        super().__init__(cfgs=cfgs, mode='large', **kwargs)