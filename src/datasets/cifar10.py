from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from src.registry import register
from src.datasets.base import DatasetBase
from typing import Optional, Callable, Dict, Any


class CIFAR10Wrap(DatasetBase):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(transform)
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any],
                    transform: Optional[Callable] = None,
                    train: bool = True):
        return cls(
            root=cfg.get("root", "./data"),
            train=train,
            download=cfg.get("download", True),
            transform=transform,
        )

@register("dataset", "cifar10")
def build_cifar10_dataloaders(cfg, train_tf, eval_tf):
    # 통일된 DataLoader 생성 함수 (원한다면 레지스트리에 등록)
    train_set = CIFAR10Wrap.from_config(cfg, transform=train_tf, train=True)
    val_set   = CIFAR10Wrap.from_config(cfg, transform=eval_tf, train=False)

    # train 데이터 길이 제한
    train_max_len = cfg.get("max_train_data_len", None)
    valid_max_len = cfg.get("max_valid_data_len", None)
    if train_max_len is not None and train_max_len < len(train_set):
        train_set = Subset(train_set, range(train_max_len))
    if valid_max_len is not None and valid_max_len < len(val_set):
        val_set = Subset(val_set, range(valid_max_len))

    return (
      DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                 num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]),
      DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                 num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    )

    