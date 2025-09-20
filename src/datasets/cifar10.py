from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from src.registry import register
from .base import DatasetBase

class CIFAR10Wrap(DatasetBase):
    def __init__(self, root, train, transform, download=True):
        self.ds = CIFAR10(root=root, train=train, transform=transform, download=download)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i]

@register("dataset", "cifar10")
def build_cifar10_dataloaders(cfg, train_tf, eval_tf):
    # 통일된 DataLoader 생성 함수 (원한다면 레지스트리에 등록)
    train_set = CIFAR10Wrap.from_config(cfg["data"], transform=train_tf, train=True)
    val_set   = CIFAR10Wrap.from_config(cfg["data"], transform=eval_tf, train=False)

    return (
      DataLoader(train_set, batch_size=cfg["data"]["batch_size"], shuffle=True,
                 num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"]),
      DataLoader(val_set, batch_size=cfg["data"]["batch_size"], shuffle=False,
                 num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"])
    )

    