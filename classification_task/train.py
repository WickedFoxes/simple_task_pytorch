import os
import time
import argparse
import torch

from models import CustomNet, ResNet_mini
from utils.scheduler import build_scheduler
from utils.util import set_seed
from utils.optimizer import build_optimizer
from utils.evaluate import evaluate, accuracy
from utils.loss import build_loss
from utils.datasets import build_train_dataloader


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
        running_acc  += accuracy(logits, targets) * bs
        n += bs

    return running_loss / n, running_acc / n

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
    optimizer_type="SGD",
    loss_type="CE",
    use_early_stopping=True,
    patience=10,          
    min_delta=1e-4
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 손실함수 선택
    criterion = build_loss(loss_type)
    
    # 옵티마이저 선택
    optimizer = build_optimizer(model, lr, weight_decay, momentum, optimizer_type)

    # 스케줄 선택
    scheduler = build_scheduler(optimizer, scheduler_type, epochs)

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
        if use_early_stopping and patience_counter >= patience:
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
        
        if use_early_stopping:
            print(f"[{epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} acc={test_acc:.4f} | "
                f"lr={current_lr:.5f} | time={elapsed_time:.2f}s | "
                f"patience: {patience_counter}/{patience}")
        else:
            print(f"[{epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} acc={test_acc:.4f} | "
                f"lr={current_lr:.5f} | time={elapsed_time:.2f}s ")
    
    print(f"Training completed. Final best accuracy: {best_acc:.4f}")
    return history, best_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification Training')
    parser.add_argument('--model', default='resnet', type=str,
                        help='model name')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='path to image data')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset name')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        help='learning rate scheduler')
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='optimizer')
    parser.add_argument('--loss', default='CE', type=str,
                        help='loss function')
    parser.add_argument('--use_early_stopping', action='store_true',
                        help='use early stopping')
    parser.add_argument('--patience', default=20, type=int,
                        help='patience for early stopping')
    parser.add_argument('--img_size', default=32, type=int,
                        help='image size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--use_amp', action='store_true',
                        help='use mixed precision training')
    parser.add_argument('--ckpt_dir', default='./', type=str,
                        help='directory to checkpoint')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes')
    args = parser.parse_args()

    model_name = args.model
    data_dir = args.data_dir
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum
    seed = args.seed
    scheduler_type = args.scheduler
    optimizer_type = args.optimizer
    loss_type = args.loss
    use_early_stopping = args.use_early_stopping
    patience = args.patience
    img_size = args.img_size
    num_workers = args.num_workers
    use_amp = args.use_amp
    ckpt_dir = args.ckpt_dir
    num_classes = args.num_classes
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, valid_loader = build_train_dataloader(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        valid_ratio=0.1
    )

    if model_name == 'resnet':
        model = ResNet_mini(num_classes=num_classes)
    elif model_name == 'custom':
        model = CustomNet(num_classes=num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} Total parameters: {total_params/1000000}M")

    history, best_state = fit(
        model=model,
        train_loader=train_loader,
        test_loader=valid_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        device=device,
        use_amp=True,
        scheduler_type=scheduler_type,
        optimizer_type=optimizer_type,
        patience=patience,
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")

    torch.save({"model": best_state, "meta": {"model_name": model_name}}, ckpt_path)
    print(f"Saved best checkpoint to: {ckpt_path}")