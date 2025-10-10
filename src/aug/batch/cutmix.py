import torch
import torch.nn.functional as F
import random
from src.registry import register
from src.aug.batch.base import BatchAugBase

def _rand_bbox(H, W, lam):
    # CutMix bbox 생성
    cut_rat = (1. - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W);  y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

@register("batch_aug", "cutmix")
class CutMix(BatchAugBase):
    def __init__(self, alpha=1.0, p=1.0, num_classes=None):
        super().__init__(p)
        self.alpha = alpha
        self.num_classes = num_classes

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
        # 레이블을 [B, C]로 통일
        y = self._to_one_hot(y)  # [B, C]

        if (random.random() > self.p) or (self.alpha <= 0):
            # 미적용 시에도 [B, C] 유지
            return x, y, False

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        B, C, H, W = x.size()
        index = torch.randperm(B, device=x.device)

        # 박스 샘플링
        x1, y1, x2, y2 = _rand_bbox(H, W, lam)

        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        # 잘린 영역 비율을 반영하여 lam 재계산
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))

        # 레이블도 soft-mix
        mixed_y = lam * y + (1.0 - lam) * y[index]  # [B, C]

        return mixed_x, mixed_y, True