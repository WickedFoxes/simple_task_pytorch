# references
# - https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py
# - https://cpm0722.github.io/pytorch-implementation/transformer
# - https://github.com/cpm0722/transformer_pytorch/tree/main
# - https://nlp.seas.harvard.edu/2018/04/03/attention.html
# - https://wikidocs.net/31379

import copy
import warnings
import math
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.container import ModuleList

from src.registry import register
from src.models.base import ModelBase

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out

def scaled_dot_product_attention(
        query, 
        key, 
        value, 
        attn_mask=None, 
        dropout_p=0.0,
) -> torch.Tensor:
    d_k = query.size(-1)
    L = query.size(-2) # [batch, heads, L, d_k]
    S = key.size(-2) # [batch, heads, S, d_k]

    scale_factor = 1 / math.sqrt(d_k)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class MultiheadAttention(nn.Module):
    def __init__(
            self, 
            num_heads, 
            embed_dim, 
            dropout_p=0.1,
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
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # (B, H, L, d_k)

        # 4) 헤드 결합 및 출력 투영
        B, H, L, d_k = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H * d_k)  # (B, L, D)
        out = self.out_proj(attn_out)  # (B, L, D)
        return out

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, embed_dim, feedforward_dim, dropout=0.1, activation=F.relu):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(embed_dim, feedforward_dim)
        self.w2 = nn.Linear(feedforward_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.w2(self.dropout(self.activation(self.w1(x))))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout_p, activation=F.relu):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(num_heads, embed_dim, dropout_p)
        self.feedforward = PositionwiseFeedForward(embed_dim, feedforward_dim, dropout_p, activation)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.activation = activation

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = residual + x
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, dropout_p, activation=F.relu):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout_p, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout_p, activation=F.relu):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(num_heads, embed_dim, dropout_p)
        self.cross_attn = MultiheadAttention(num_heads, embed_dim, dropout_p) 
        self.feedforward = PositionwiseFeedForward(embed_dim, feedforward_dim, dropout_p, activation)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)
        self.activation = F.relu

    def forward(self, x, memory, self_attn_mask=None, cross_attn_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.dropout(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, attn_mask=cross_attn_mask)
        x = self.dropout(x)
        x = residual + x
        residual = x
        x = self.norm3(x)
        x = self.feedforward(x)
        x = self.dropout3(x)
        x = residual + x
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, dropout_p, activation=F.relu):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, feedforward_dim, dropout_p, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, self_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(
            self,
            embed_dim, 
            num_heads, 
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            feedforward_dim: int = 2048,
            dropout_p: float = 0.1,
            activation = F.relu,
    ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(embed_dim, num_heads, feedforward_dim, num_encoder_layers, dropout_p, activation)
        self.decoder = TransformerDecoder(embed_dim, num_heads, feedforward_dim, num_decoder_layers, dropout_p, activation)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.encoder(src, src_mask)
        tgt = self.decoder(tgt, src, tgt_mask, src_mask)
        return tgt

@register("model", "transformer_tl")
class TransformerTL(ModelBase):
    """
    - 입력: src(input ids), tgt_in (teacher forcing용 <bos> + y[:-1])
    - 출력: tgt_logits (단어 분포)
    - 손실: CrossEntropy(ignore_index=pad)
    """
    def __init__(
            self, 
            src_vocab_size: int, 
            tgt_vocab_size: int,
            src_pad_id: int, 
            tgt_pad_id: int,
            embed_dim: int, 
            num_heads: int,
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            feedforward_dim=2048, 
            dropout_p=0.1
    ):
        super().__init__()
        self.src_tok = nn.Embedding(src_vocab_size, embed_dim, padding_idx=src_pad_id)
        self.tgt_tok = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=tgt_pad_id)
        self.pos = PositionalEncoding(embed_dim)
        self.transformer = Transformer(
            embed_dim=embed_dim, num_heads=num_heads,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            feedforward_dim=feedforward_dim, dropout_p=dropout_p, activation=F.relu
        )
        self.generator = nn.Linear(embed_dim, tgt_vocab_size)
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id


    def forward(self, src_ids, tgt_in_ids):
        """
        src_ids: (batch, src_len)  tgt_in_ids: (batch, tgt_len)
        """
        src = self.pos(self.src_tok(src_ids))
        tgt_in = self.pos(self.tgt_tok(tgt_in_ids))

        # 마스크 생성
        src_key_padding_mask = (src_ids == self.src_pad_id)  # (batch, src_len) True=pad
        tgt_key_padding_mask = (tgt_in_ids == self.tgt_pad_id)  # (batch, tgt_len)

        src_mask = self.make_src_mask(src_key_padding_mask)   # (batch, 1, 1, src_len) 형태 가정
        tgt_mask = self.make_tgt_mask(tgt_key_padding_mask)   # (batch, 1, tgt_len, tgt_len) (look-ahead + pad)

        hidden = self.transformer(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)  # (batch, tgt_len, d)
        logits = self.generator(hidden)  # (batch, tgt_len, vocab)
        return logits
    
    def make_src_mask(self, src_key_padding: torch.Tensor):
        """
        src_key_padding: (batch, src_len) where True indicates PAD
        반환: (batch, 1, 1, src_len), attention 에서 add-mask(-inf)로 사용 가정
        """
        return src_key_padding.unsqueeze(1).unsqueeze(2)  # True=masked

    def make_tgt_mask(self, tgt_key_padding: torch.Tensor):
        """
        look-ahead(하삼각) + pad mask 결합
        반환: (batch, 1, tgt_len, tgt_len)
        """
        bsz, tgt_len = tgt_key_padding.size()
        # subsequent mask: (1, 1, tgt_len, tgt_len)
        subsequent = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool), diagonal=1)  # True=mask 상삼각
        subsequent = subsequent.unsqueeze(0).unsqueeze(0)  # (1,1,t,t)
        # pad mask: (batch, 1, 1, tgt_len)
        pad_mask = tgt_key_padding.unsqueeze(1).unsqueeze(2)  # True=mask
        # 브로드캐스트 결합
        # 최종: True 위치는 attention에서 가려짐
        return subsequent | pad_mask