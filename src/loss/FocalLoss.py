from typing import Optional, Union
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    단일-레이블 다중분류용 focal loss (logits 입력, CE 기반)
    """
    def __init__(self, gamma: float=2.0, alpha: Optional[Union[float, torch.Tensor]]=None,
                 reduction: str="mean", ignore_index: int=-100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.alpha = alpha  # 스칼라 or [C] 텐서

    def forward(self, logits, target):
        # CE의 log_prob 기반 구현
        ce = nn.functional.cross_entropy(logits, target, reduction="none", ignore_index=self.ignore_index)
        # pt = exp(-CE)
        pt = torch.exp(-ce)
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_t = torch.full_like(target, fill_value=self.alpha, dtype=torch.float)
            else:
                # class-wise alpha
                alpha_vec = self.alpha.to(logits.device)
                alpha_t = alpha_vec.gather(0, target.view(-1)).view_as(target)
            ce = alpha_t * ce
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss