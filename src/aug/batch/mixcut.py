import torch
import random
from src.registry import register
from src.aug.batch.base import BatchAugBase
from src.aug.batch.mixup import Mixup
from src.aug.batch.cutmix import CutMix


@register("batch_aug", "mixcut")
class MixCut(BatchAugBase):
    """ mixup/cutmix를 확률로 스위칭 """
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, p=1.0, switch_prob=0.5):
        super().__init__(p)
        self.mix = Mixup(mixup_alpha=mixup_alpha, p=1.0)
        self.cut = CutMix(cutmix_alpha=cutmix_alpha, p=1.0)
        self.switch_prob = switch_prob

    def __call__(self, x, y):
        if random.random() > self.p:
            return x, (y, y, 1.0), False
        if random.random() < self.switch_prob:
            return self.mix(x, y)
        else:
            return self.cut(x, y)