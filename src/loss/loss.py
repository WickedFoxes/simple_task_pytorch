import torch
import torch.nn as nn
from typing import Optional, Sequence, Union
from src.registry import register
import os, json

from src.loss.FocalLoss import FocalLoss
from src.loss.SoftTargetCrossEntropyLoss import SoftTargetCrossEntropyLoss


def _maybe_tensor_weights(w, device="cpu"):
    if w is None:
        return None
    if isinstance(w, str):  # 파일 경로 지원(.json, .txt 등)
        with open(w, "r") as f:
            data = json.load(f) if w.endswith(".json") else [float(x) for x in f.read().strip().split()]
        return torch.tensor(data, dtype=torch.float, device=device)
    if isinstance(w, (list, tuple)):
        return torch.tensor(w, dtype=torch.float, device=device)
    if torch.is_tensor(w):
        return w.to(device)
    return None

@register("loss", "cross_entropy")
def build_ce(reduction: str="mean",
             ignore_index: int=-100,
             class_weight: Optional[Union[str, Sequence[float]]]=None,
             **kwargs):
    weight = _maybe_tensor_weights(class_weight)
    return nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

@register("loss", "label_smoothing")
def build_ce_ls(reduction: str="mean",
                ignore_index: int=-100,
                label_smoothing: float=0.1,
                class_weight: Optional[Union[str, Sequence[float]]]=None,
                **kwargs):
    weight = _maybe_tensor_weights(class_weight)
    # PyTorch>=1.10: CrossEntropyLoss(label_smoothing=...)
    return nn.CrossEntropyLoss(weight=weight, reduction=reduction,
                               ignore_index=ignore_index, label_smoothing=label_smoothing)

@register("loss", "focal")
def build_focal(gamma: float=2.0, alpha: Optional[Union[float, Sequence[float], str]]=None,
                reduction: str="mean", ignore_index: int=-100, **kwargs):
    alpha_t = _maybe_tensor_weights(alpha)
    return FocalLoss(gamma=gamma, alpha=alpha_t, reduction=reduction, ignore_index=ignore_index)

@register("loss", "bce_logits")
def build_bce_logits(reduction: str="mean",
                     pos_weight: Optional[Union[Sequence[float], str]]=None,
                     **kwargs):
    pw = _maybe_tensor_weights(pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)

@register("loss", "soft_target")
def build_soft_target(reduction: str="mean", **kwargs):
    return SoftTargetCrossEntropyLoss(reduction=reduction)