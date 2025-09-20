import torch
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
    def __init__(self, cutmix_alpha=1.0, p=1.0):
        super().__init__(p)
        self.alpha = cutmix_alpha

    def __call__(self, x, y):
        if random.random() > self.p or self.alpha <= 0:
            return x, (y, y, 1.0), False
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        B, C, H, W = x.size()
        index = torch.randperm(B, device=x.device)
        x1, y1, x2, y2 = _rand_bbox(H, W, lam)
        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        # 실제 lam은 잘려진 영역 반영해 재계산
        lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam), True