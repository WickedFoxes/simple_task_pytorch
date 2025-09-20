from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class DatasetBase(Dataset, ABC):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    @abstractmethod
    def __len__(self):
        """데이터셋의 크기 반환"""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """데이터셋에서 하나의 샘플 반환"""
        pass

    # from_config를 모든 베이스에 공통 제공(선택)
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, **kwargs):
        """cfg(dict)로부터 인스턴스 생성"""
        pass