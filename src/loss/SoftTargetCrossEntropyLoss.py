import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropyLoss(nn.Module):
    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,) int class indices  or  (B, C) soft/one-hot
        """
        if targets.dim() == 1:  # class indices
            num_classes = logits.size(-1)
            targets = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        elif targets.dim() == 2:
            if targets.size(-1) != logits.size(-1):
                raise ValueError(
                    f"[SoftTargetCELoss] targets C={targets.size(-1)} != logits C={logits.size(-1)}. "
                    "모델 헤드(out_features)와 num_classes(원-핫/소프트 라벨 생성)가 일치해야 합니다."
                )
            targets = targets.to(dtype=logits.dtype)
        else:
            raise ValueError(f"[SoftTargetCELoss] 지원하지 않는 targets shape: {targets.shape}")

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss