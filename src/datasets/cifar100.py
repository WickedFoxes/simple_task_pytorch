from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from src.registry import register
from .base import DatasetBase
from typing import Optional, Callable, Dict, Any

class CIFAR100Wrap(DatasetBase):
    def __init__(self, root, train, transform, download=True):
        self.ds = CIFAR100(root=root, train=train, transform=transform, download=download)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i]
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


@register("dataset", "cifar100")
def build_cifar100_dataloaders(cfg, train_tf, eval_tf):
    # 통일된 DataLoader 생성 함수 (원한다면 레지스트리에 등록)
    train_set = CIFAR100Wrap.from_config(cfg, transform=train_tf, train=True)
    val_set   = CIFAR100Wrap.from_config(cfg, transform=eval_tf, train=False)

    return (
      DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                 num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]),
      DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                 num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    )