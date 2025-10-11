# EfficientNet Implementation
# Based on the MobileNetV3 structure provided
# Reference: https://arxiv.org/abs/1905.11946

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register
from src.models.base import ModelBase
import math


class Swish(nn.Module):
    """Swish activation function used in EfficientNet"""
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, se_channels, 1, bias=False),
            Swish(),
            nn.Conv2d(se_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block (MBConv)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_path=0.0):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.drop_path = drop_path
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )
        else:
            self.expand_conv = nn.Identity()
            
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )
        
        # Squeeze and Excitation
        self.se_block = SEBlock(hidden_dim, se_ratio) if se_ratio > 0 else nn.Identity()
        
        # Output phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Drop path for regularization
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Squeeze and Excitation
        x = self.se_block(x)
        
        # Output projection
        x = self.project_conv(x)
        
        # Residual connection with drop path
        if self.use_residual:
            x = self.drop_path_layer(x)
            x = x + residual
            
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def _make_divisible(v, divisor=8, min_value=None):
    """Make channels divisible by divisor"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class EfficientNet(nn.Module):
    """EfficientNet architecture"""
    def __init__(self, block_args_list, width_mult=1.0, depth_mult=1.0, num_classes=1000, drop_rate=0.2, drop_path_rate=0.0):
        super(EfficientNet, self).__init__()
        
        # Stem
        stem_channels = _make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=1, padding=1, bias=False), # stried 2 -> 1
            nn.BatchNorm2d(stem_channels),
            Swish()
        )
        
        # Build blocks
        self.blocks = nn.ModuleList([])
        in_channels = stem_channels
        total_blocks = sum([args['num_repeat'] for args in block_args_list])
        block_idx = 0
        
        for stage_idx, block_args in enumerate(block_args_list):
            num_repeat = max(1, int(math.ceil(block_args['num_repeat'] * depth_mult)))
            
            for block_repeat in range(num_repeat):
                # Drop path rate increases linearly
                drop_path = drop_path_rate * block_idx / total_blocks
                
                # First block in stage might have different stride
                stride = block_args['stride'] if block_repeat == 0 else 1
                
                # Scale channels
                input_channels = in_channels if block_repeat == 0 else output_channels
                output_channels = _make_divisible(block_args['output_channels'] * width_mult)
                
                block = MBConvBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=block_args['kernel_size'],
                    stride=stride,
                    expand_ratio=block_args['expand_ratio'],
                    se_ratio=block_args['se_ratio'],
                    drop_path=drop_path
                )
                self.blocks.append(block)
                block_idx += 1
                
            in_channels = output_channels
        
        # Head
        head_channels = _make_divisible(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(drop_rate),
        )
        
        self.classifier = nn.Linear(head_channels, num_classes)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.head(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# EfficientNet scaling coefficients
EFFICIENTNET_PARAMS = {
    'b0': (1.0, 1.0, 224),  # width_mult, depth_mult, image_size
    'b1': (1.0, 1.1, 240),
    'b2': (1.1, 1.2, 260),
    'b3': (1.2, 1.4, 300),
    'b4': (1.4, 1.8, 380),
    'b5': (1.6, 2.2, 456),
    'b6': (1.8, 2.6, 528),
    'b7': (2.0, 3.1, 600),
}

# Base block arguments for EfficientNet-B0
# BASE_BLOCK_ARGS = [
#     {'kernel_size': 3, 'num_repeat': 1, 'output_channels': 16, 'expand_ratio': 1, 'stride': 1, 'se_ratio': 0.25}, 
#     {'kernel_size': 3, 'num_repeat': 2, 'output_channels': 24, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
#     {'kernel_size': 5, 'num_repeat': 2, 'output_channels': 40, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
#     {'kernel_size': 3, 'num_repeat': 3, 'output_channels': 80, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
#     {'kernel_size': 5, 'num_repeat': 3, 'output_channels': 112, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
#     {'kernel_size': 5, 'num_repeat': 4, 'output_channels': 192, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
#     {'kernel_size': 3, 'num_repeat': 1, 'output_channels': 320, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
# ]
# cifar (32,32)에 맞게 수정
BASE_BLOCK_ARGS = [
    {'kernel_size': 3, 'num_repeat': 1, 'output_channels': 16, 'expand_ratio': 1, 'stride': 1, 'se_ratio': 0.25}, 
    {'kernel_size': 3, 'num_repeat': 2, 'output_channels': 24, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'num_repeat': 2, 'output_channels': 40, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'num_repeat': 3, 'output_channels': 80, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'num_repeat': 3, 'output_channels': 112, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'num_repeat': 4, 'output_channels': 192, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'num_repeat': 1, 'output_channels': 320, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
]

@register("model", "efficientnet_b0")
class EfficientNetB0(EfficientNet, ModelBase):
    def __init__(self, **kwargs):
        width_mult, depth_mult, _ = EFFICIENTNET_PARAMS['b0']
        super().__init__(
            block_args_list=BASE_BLOCK_ARGS,
            width_mult=width_mult,
            depth_mult=depth_mult,
            **kwargs
        )