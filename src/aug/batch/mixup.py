import torch
import torch.nn.functional as F
import random
from src.registry import register
from src.aug.batch.base import BatchAugBase


@register("batch_aug", "mixup")
class Mixup(BatchAugBase):
    def __init__(self, alpha=0.2, p=1.0, num_classes=None):
        super().__init__(p)
        self.alpha = alpha
        self.num_classes = num_classes  # 선택: y가 인덱스일 때 C를 명시적으로 줄 수 있음

    def _to_one_hot(self, y):
        # y: [B] (long) or [B, C] (float/bool)
        if y.dim() == 1:
            if self.num_classes is None:
                C = int(y.max().item()) + 1
            else:
                C = self.num_classes
            y_oh = F.one_hot(y.to(torch.long), num_classes=C).to(torch.float)
        elif y.dim() == 2:
            y_oh = y.to(torch.float)
        else:
            raise ValueError(f"y must be [B] or [B, C], got shape {tuple(y.shape)}")
        return y_oh

    def __call__(self, x, y):
        y = self._to_one_hot(y)  # [B, C]
        if (random.random() > self.p) or (self.alpha <= 0):
            # 적용 안 할 때도 [B, C] 유지
            return x, y, False

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        index = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index]
        mixed_y = lam * y + (1.0 - lam) * y[index]  # [B, C]

        return mixed_x, mixed_y, True