import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss for soft labels (e.g., Mixup, CutMix).
    Supports (B, C) shaped soft-label tensors.
    """
    def __init__(self, reduction: str = 'mean'):
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1)  # shape: (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss