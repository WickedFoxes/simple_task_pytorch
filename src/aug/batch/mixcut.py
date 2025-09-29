import torch
import torch.nn.functional as F
import random
from src.registry import register
from src.aug.batch.base import BatchAugBase
from src.aug.batch.mixup import Mixup
from src.aug.batch.cutmix import CutMix


@register("batch_aug", "mixcut")
class MixCut(BatchAugBase):
    """ mixup/cutmix를 확률로 스위칭 (항상 [B, C] 레이블 반환) """
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, p=1.0, switch_prob=0.5, num_classes=None):
        super().__init__(p)
        # 내부에서 [B, C] 레이블을 만들 수 있도록 num_classes 전달
        self.mix = Mixup(alpha=mixup_alpha, p=1.0, num_classes=num_classes)
        self.cut = CutMix(alpha=cutmix_alpha, p=1.0, num_classes=num_classes)
        self.switch_prob = switch_prob
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
        # 미적용 분기에서도 [B, C] 형식 유지
        if random.random() > self.p:
            y = self._to_one_hot(y, self.num_classes)  # [B, C]
            return x, y, False

        # 스위칭
        if random.random() < self.switch_prob:
            # Mixup 호출: (mixed_x, mixed_y, flag)
            return self.mix(x, y)
        else:
            # CutMix 호출: (mixed_x, mixed_y, flag)
            return self.cut(x, y)