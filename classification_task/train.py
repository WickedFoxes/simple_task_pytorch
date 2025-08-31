# -*- coding: utf-8 -*-
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from models import CustomNet, ResNet

# ------------------------------------------------------------
# 0) 재현성 고정
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ------------------------------------------------------------
# 1) 데이터 로더
# ------------------------------------------------------------
def build_dataloaders(
    data_root="./data",
    batch_size=128,
    num_workers=4,
    use_aug=True,
):
    """
    CIFAR-10용 공통 전처리 & 로더. (train/test 동일 환경 보장)
    """
    # CIFAR-10 통계
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ------------------------------------------------------------
# 3) 학습/평가 루프
# ------------------------------------------------------------
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_from_logits(logits, targets) * bs
        n += bs

    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, targets) * bs
        n += bs
    return total_loss / n, total_acc / n

def fit(
    model,
    train_loader,
    test_loader,
    epochs=50,
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    device=None,
    use_amp=True,
    scheduler_type="cosine",
):
    """
    동일한 학습 조건으로 모델을 학습/평가합니다.
    - scheduler_type: "cosine", "multistep", None
    반환: history(dict), best_state_dict
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.startswith("cuda")) else None

    best_acc = 0.0
    best_state = None
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "lr": []
    }

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        test_loss, test_acc   = evaluate(model, test_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        print(f"[{epoch:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} acc={test_acc:.4f} | lr={current_lr:.5f}")

    return history, best_state

# ------------------------------------------------------------
# 4) 통합 실험 헬퍼
# ------------------------------------------------------------
def run_experiment(
    model_builder,
    model_name: str,
    epochs=50,
    batch_size=128,
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    use_aug=True,
    scheduler_type="cosine",
    seed=42,
    data_root="./data",
    num_workers=4,
):
    """
    동일한 설정으로 모델을 학습/평가하고 결과를 반환.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        use_aug=use_aug,
    )

    model = model_builder()
    history, best_state = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        device=device,
        use_amp=True,
        scheduler_type=scheduler_type,
    )

    # best checkpoint 저장(선택)
    ckpt_path = f"./{model_name}_cifar10_best.pth"
    torch.save({"model": best_state, "meta": {"model_name": model_name}}, ckpt_path)
    print(f"Saved best checkpoint to: {ckpt_path}")
    return history, best_state

# ------------------------------------------------------------
# 5) 사용 예시
# ------------------------------------------------------------
# (1) CustomNet 실험
# from your_module import CustomNet  # 이미 정의되어 있다고 가정
def build_customnet():
    return CustomNet(num_classes=10)

if __name__ == "__main__":
    # 동일한 하이퍼파라미터로 두 모델 비교
    hist_custom, state_custom = run_experiment(
        model_builder=build_customnet,
        model_name="CustomNet",
        epochs=50,
        batch_size=128,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        scheduler_type="cosine",
        seed=42,
    )
