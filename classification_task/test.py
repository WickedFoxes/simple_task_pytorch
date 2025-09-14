import os
import time
import argparse
import torch

from models import CustomNet, ResNet_mini
from utils.datasets import get_cifar10_test_dataloader
from utils.evaluate import accuracy
from utils.loss import build_loss
from utils.util import set_seed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', type=str, help='model name')
    parser.add_argument('--chk_path', type=str, required=True, help='the checkpoint file you want to test')
    parser.add_argument('--data_dir', default='./data', type=str,
                    help='path to dataset')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--img_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes')
    parser.add_argument('--loss', default='CE', type=str,
                        help='loss function')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset name')
    args = parser.parse_args()

    seed = args.seed
    dataset = args.dataset
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    img_size = args.img_size
    num_classes = args.num_classes
    model_name = args.model
    ckpt_path = args.chk_path
    loss_type = args.loss


    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if dataset == 'cifar10':
        test_loader = get_cifar10_test_dataloader(
            data_path=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
        )

    if model_name == 'resnet':
        model = ResNet_mini(num_classes=num_classes)
    elif model_name == 'custom':
        model = CustomNet(num_classes=num_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 손실 함수 빌드
    criterion = build_loss(loss_type)

    # 모델을 디바이스로 이동
    model = model.to(device)
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_acc  += accuracy(logits, targets) * bs
            n += bs
    print(f"Test Acc: {running_acc / n}")
    print(f"Test Loss: {running_loss / n}")