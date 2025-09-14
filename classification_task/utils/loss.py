import torch
import torch.nn as nn
import torch.nn.functional as F

def build_loss(loss_type : str):
    if loss_type == "CE":
        loss = nn.CrossEntropyLoss()
        return loss
    elif loss_type == "Focal":
        loss = FocalLoss()
        return loss
    elif loss_type == "LabelSmoothing":
        loss = LabelSmoothingCrossEntropy()
        return loss
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (for logits).
    - inputs: (N, C) logits
    - targets: (N,) class indices in [0, C-1]
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,                   # float or 1D tensor of shape [C] for per-class alpha
        reduction: str = "mean",      # "none" | "mean" | "sum"
        ignore_index: int = -100
    ):
        super().__init__()
        assert reduction in ["none", "mean", "sum"]
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Mask for valid targets (ignore_index 제외)
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return inputs.new_tensor(0.)

        # log_softmax & NLL for per-sample CE
        log_prob = F.log_softmax(inputs, dim=1)              # (N, C)
        nll = F.nll_loss(log_prob, targets, reduction="none", ignore_index=self.ignore_index)  # (N,)

        # pt = exp(-CE) = predicted prob of the target class
        pt = torch.exp(-nll).clamp_min(1e-12)

        # 기본 focal term
        focal_factor = (1 - pt) ** self.gamma

        # alpha weighting (scalar or per-class)
        if self.alpha is not None:
            if self.alpha.dim() == 0:
                alpha_t = self.alpha
            else:
                # gather per-class alpha
                alpha_t = self.alpha.to(inputs.device).gather(0, targets.clamp_min(0))
            loss = alpha_t * focal_factor * nll
        else:
            loss = focal_factor * nll

        # 유효 샘플만 집계
        loss = loss[valid_mask]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Multi-class Label Smoothing Cross-Entropy (for logits).
    - inputs: (N, C) logits
    - targets: (N,) class indices in [0, C-1]
    """
    def __init__(
        self,
        smoothing: float = 0.1,       # epsilon
        reduction: str = "mean",      # "none" | "mean" | "sum"
        ignore_index: int = -100
    ):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        assert reduction in ["none", "mean", "sum"]
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n, c = inputs.size()
        log_prob = F.log_softmax(inputs, dim=1)   # (N, C)

        # valid mask & safe targets
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return inputs.new_tensor(0.)

        targets_valid = targets[valid_mask]

        # Negative log-likelihood for the true class: -log p_y
        nll = F.nll_loss(log_prob, targets, reduction="none", ignore_index=self.ignore_index)  # (N,)

        # Uniform distribution term: -mean_c log p_c
        mean_log_prob = -log_prob.mean(dim=1)  # (N,)

        eps = self.smoothing
        loss = (1 - eps) * nll + eps * mean_log_prob

        loss = loss[valid_mask]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
