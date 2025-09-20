import torch
import random
from src.registry import register
from src.aug.batch.base import BatchAugBase

@register("batch_aug", "mixup")
class Mixup(BatchAugBase):
    def __init__(self, mixup_alpha=0.2, p=1.0):
        super().__init__(p)
        self.alpha = mixup_alpha

    def __call__(self, x, y):
        if random.random() > self.p or self.alpha <= 0:
            return x, (y, y, 1.0), False
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam), True