from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class DatasetBase(Dataset, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, transform=None, train=True):
        ...
