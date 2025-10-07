# references
# - https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from src.registry import register
from src.models.base import ModelBase

@register("model", "ViT")
class VisionTransformer(ModelBase):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 100,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size

        self.conv_proj = nn.Conv2d( # 구현상의 편의(convenience)로 Conv2D
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        seq_length = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None: # classifier로 작동
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else: # 
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # init
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)


    # 이미지 텐서(x)를 패치 단위로 분할하고, 각 패치를 임베딩 벡터로 변환
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1) # -1은 기존 차원 그대로 유지
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    def replace_head(self, new_num_classes, representation_size=None):
        """사전학습된 본체는 유지, 분류 head만 교체."""
        hidden_dim = self.hidden_dim
        if representation_size is None:
            self.heads = nn.Sequential(nn.Linear(hidden_dim, new_num_classes))
            nn.init.zeros_(self.heads[0].weight)
            nn.init.zeros_(self.heads[0].bias)
        else:
            # 사전학습 때 representation_size를 썼다면 동일하게 구성
            self.heads = nn.Sequential(
                nn.Linear(hidden_dim, representation_size),
                nn.Tanh(),
                nn.Linear(representation_size, new_num_classes),
            )
            for m in self.heads.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=(1 / m.in_features) ** 0.5)
                    nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MultiheadAttention(num_heads, hidden_dim, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)



def scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0,
) -> torch.Tensor:
    # query,key,value: [B,H,L,d], [B,H,S,d], [B,H,S,d]
    d_k = query.size(-1)
    scale = 1.0 / math.sqrt(d_k)

    # [B,H,L,S]
    attn_weight = (query @ key.transpose(-2, -1)) * scale

    if attn_mask is not None:
        # attn_mask shape: broadcastable to [B,H,L,S]
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_weight = attn_weight + attn_mask  # additive mask (-inf for masked, 0 for valid)

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, p=dropout_p, training=query.requires_grad)  # or training flag from module
    return attn_weight @ value  # [B,H,L,d]


class MultiheadAttention(nn.Module):
    def __init__(
            self, 
            num_heads, 
            embed_dim, 
            dropout_p=0.0,
        ):
        super(MultiheadAttention, self).__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        # self.batch_first = batch_first
        self.dropout_p = dropout_p
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        # pytorch의 경우 out_proj의 경우 양자화를 대비하여 NonDynamicallyQuantizableLinear를 사용하나 생략
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        

    def _split_heads(self, x):
        # x: (B, L, D) -> (B, H, L, d_k)
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, L, H, d_k) -> (B, H, L, d_k)
        return x  # (B, H, L, d_k)

    def forward(
            self, 
            query, 
            key, 
            value, 
            attn_mask=None,
    ):
        """
        query: (B, Lq, C)
        key:   (B, Lk, C)
        value: (B, Lk, C)
        mask:  broadcastable to (B, num_heads, L, d_k) 또는 (L, d_k) 등
        """
        # 1) 선형 투영
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2) (B, H, L, D)로 변형
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 3) SDPA 호출 (학습 시에만 dropout 적용)
        attn_out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, H, L, d_k)

        # 4) 헤드 결합 및 출력 투영
        B, H, L, d_k = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H * d_k)  # (B, L, D)
        out = self.out_proj(attn_out)  # (B, L, D)
        return out
    