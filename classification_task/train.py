import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import CustomNet, ResNet_mini

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
def create_cifar10_dataloaders(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        use_aug=True,
        img_size=224):
    # ImageNet 통계(사전학습 모델과 일치시켜야 성능 손실을 줄일 수 있음)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def create_mnist_dataloaders(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        use_aug=True,
        img_size=224):
    # ImageNet 통계 (사전학습 모델 사용 시 일치 필요)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(28, padding=4),   # MNIST 원본 해상도는 28x28
            transforms.RandomRotation(10),          # MNIST는 회전 증강이 효과적
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Grayscale(num_output_channels=3),  # 1채널 → 3채널
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ------------------------------------------------------------
# 3) 학습/평가 루프
# ------------------------------------------------------------
def accuracy_from_logits(logits, targets):
    with torch.no_grad():
        preds = logits.argmax(dim=1) # 가장 큰 값 logits 값의 index를 출력
        return (preds == targets).float().mean().item() # 정확도를 0.0 ~ 1.0 사이의 수치로 표현

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
    optimizer="SGD",
    patience=10,          # 조기 중단: 성능 향상을 기다리는 최대 epoch 수
    min_delta=1e-4        # 성능 향상으로 인정할 최소 개선 폭
):
    """
    동일한 학습 조건으로 모델을 학습/평가합니다.
    - scheduler_type: "cosine", "multistep", None
    - patience: 조기 중단 기준 epoch 수
    - min_delta: 성능 향상으로 인정할 최소 개선 폭
    반환: history(dict), best_state_dict
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 옵티마이저 선택
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 스케줄 선택
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs*0.5), int(epochs*0.75)],
            gamma=0.1
        )
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.startswith("cuda")) else None

    best_acc = 0.0
    best_epoch = 0
    best_state = None
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "lr": [], "time": []
    }

    # 조기 중단 변수
    patience_counter = 0

    for epoch in range(1, epochs+1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        test_loss, test_acc   = evaluate(model, test_loader, device, criterion)

        if scheduler is not None:
            scheduler.step()

        # 베스트 모델 저장 및 early stopping 체크
        if test_acc > best_acc + min_delta:
            best_acc = test_acc
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0  # 향상이 있으면 카운터 초기화
            print(f"New best accuracy: {best_acc:.4f} at epoch {epoch}")
        else:
            patience_counter += 1

        # 조기 중단 조건 만족 시 종료
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
            print(f"Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
            break

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)
        history["time"].append(elapsed_time)

        print(f"[{epoch:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} acc={test_acc:.4f} | "
              f"lr={current_lr:.5f} | time={elapsed_time:.2f}s | "
              f"patience: {patience_counter}/{patience}")
    print(f"Training completed. Final best accuracy: {best_acc:.4f}")
    return history, best_state


# ------------------------------------------------------------
# 4) 통합 실험 헬퍼
# ------------------------------------------------------------
def run_experiment(
    model_builder,
    model_name: str,
    dataloaders_builder,
    num_classes=10,
    epochs=50,
    batch_size=128,
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    use_aug=True,
    scheduler_type="cosine",
    optimizer_type="SGD",
    seed=42,
    data_dir="./data",
    num_workers=4,
    img_size=224,
    patience=20,
):
    """
    동일한 설정으로 모델을 학습/평가하고 결과를 반환.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, test_loader = dataloaders_builder(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_aug=use_aug,
        img_size=img_size,
    )

    model = model_builder(num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} Total parameters: {total_params/1000000}M")

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
        optimizer=optimizer_type,
        patience=patience,
    )

    # best checkpoint 저장(선택)
    ckpt_path = f"./{model_name}_best.pth"
    torch.save({"model": best_state, "meta": {"model_name": model_name}}, ckpt_path)
    print(f"Saved best checkpoint to: {ckpt_path}")
    return history, best_state


def build_customnet(num_classes=10):
    return CustomNet(num_classes=num_classes)

def build_resnet_mini(num_classes=10):
    return ResNet_mini(num_classes=num_classes)