import torch.nn as nn
from abc import ABC, abstractmethod

class ModelBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        ...

    @classmethod
    def from_config(cls, cfg):
        # 통일된 생성 인터페이스
        return cls(**{k: v for k, v in cfg.items() if k != "name"})