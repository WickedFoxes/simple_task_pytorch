import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset


import torchvision.datasets as datasets
import torchvision.transforms as transforms

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std  = [0.2471, 0.2435, 0.261]

def build_train_dataloader(
        dataset='cifar10',
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        img_size=32,
        valid_ratio=0.1,
    ):
    if dataset == 'cifar10':
        train_loader, valid_loader = get_cifar10_train_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            valid_ratio=valid_ratio
        )
    elif dataset == 'cifar10_randaug':
        train_loader, valid_loader = get_cifar10_randaug_train_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            valid_ratio=valid_ratio
        )
    elif dataset == 'cifar10_mixup':
        train_loader, valid_loader = get_cifar10_mixup_train_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            valid_ratio=valid_ratio,
            mixup_alpha=1.0
        )
    elif dataset == 'cifar10_cutmix':
        train_loader, valid_loader = get_cifar10_cutmix_train_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            valid_ratio=valid_ratio,
            cutmix_alpha=1.0
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_loader, valid_loader

def build_test_dataloader(
        dataset='cifar10',
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        img_size=32,
    ):
    if dataset == 'cifar10':
        test_loader = get_cifar10_test_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size
        )
    return test_loader

def get_cifar10_train_dataloader(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        img_size=32,
        valid_ratio=0.1,
    ):
    common_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    full_train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf
    )

    n_total = len(full_train_set)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_subset, valid_subset = random_split(full_train_set, [n_train, n_valid])

    # 서로 다른 Dataset 인스턴스 생성
    train_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    valid_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=common_tf)

    # 동일한 인덱스로 Subset 구성
    train_set = Subset(train_base, train_subset.indices)
    valid_set = Subset(valid_base, valid_subset.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader

def get_cifar10_test_dataloader(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        img_size=32
    ):
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return test_loader

def get_cifar10_randaug_train_dataloader(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        ra_num_ops=2,                 # RandAugment: 연산 개수(N)
        ra_magnitude=9,               # RandAugment: 강도(M, 0~10 권장)
        img_size=32,
        valid_ratio=0.1,                 
    ):
    """
    Returns:
        train_loader, valid_loader
    """

    # 공통(검증/테스트) 변환: 텐서화/정규화만 포함
    common_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    full_train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf
    )

    n_total = len(full_train_set)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_subset, valid_subset = random_split(full_train_set, [n_train, n_valid])

    # 서로 다른 Dataset 인스턴스 생성
    train_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    valid_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=common_tf)

    # 동일한 인덱스로 Subset 구성
    train_set = Subset(train_base, train_subset.indices)
    valid_set = Subset(valid_base, valid_subset.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader

def get_cifar10_mixup_train_dataloader(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        img_size=32,
        valid_ratio=0.1,
        mixup_alpha=1.0,
    ):
    common_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    full_train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf
    )

    n_total = len(full_train_set)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_subset, valid_subset = random_split(full_train_set, [n_train, n_valid])

    # 서로 다른 Dataset 인스턴스 생성
    train_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    valid_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=common_tf)

    # 동일한 인덱스로 Subset 구성
    train_set = Subset(train_base, train_subset.indices)
    valid_set = Subset(valid_base, valid_subset.indices)
    
    def mixup_collate_fn(batch, alpha=1.0, num_classes=10):
        """Mixup이 적용된 Collate function"""
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.tensor(targets)

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[index, :]

        return mixed_images, mixed_targets

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: mixup_collate_fn(b, alpha=mixup_alpha)  # Mixup 적용
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, valid_loader


def get_cifar10_cutmix_train_dataloader(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        img_size=32,
        valid_ratio=0.1,
        cutmix_alpha=1.0,   # Beta 분포 파라미터(논문에서 α). 0이면 CutMix 비활성화
    ):
    """CIFAR-10 CutMix 학습/검증 DataLoader 생성"""

    # 공통/학습 변환: MixUp 예시와 동일한 파이프라인
    common_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # 전체 학습셋 로드 및 인덱스 분할
    full_train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    n_total = len(full_train_set)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_subset, valid_subset = random_split(full_train_set, [n_train, n_valid])

    # 누수 방지를 위해 동일 인덱스로 서로 다른 변환의 Dataset 인스턴스 구성
    train_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    valid_base = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=common_tf)
    train_set = Subset(train_base, train_subset.indices)
    valid_set = Subset(valid_base, valid_subset.indices)

    def _rand_bbox(W, H, lam):
        """CutMix bbox 생성 (W: width, H: height)"""
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def cutmix_collate_fn(batch, alpha=1.0, num_classes=10):
        """CutMix 적용 Collate function"""
        images, targets = zip(*batch)
        images = torch.stack(images)           # [B, C, H, W]
        targets = torch.tensor(targets)        # [B]

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0  # 비활성화

        B, C, H, W = images.size()
        index = torch.randperm(B)

        shuffled_images = images[index]
        shuffled_targets = targets[index]

        if lam < 1.0:
            x1, y1, x2, y2 = _rand_bbox(W, H, lam)
            # 영역 교체
            images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
            # 실제 잘린 면적 기반 λ 재계산(보정)
            box_area = (x2 - x1) * (y2 - y1)
            lam = 1.0 - box_area / float(W * H)

        # one-hot 라벨과 라벨 혼합
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        shuffled_onehot = F.one_hot(shuffled_targets, num_classes=num_classes).float()
        mixed_targets = lam * targets_onehot + (1.0 - lam) * shuffled_onehot

        return images, mixed_targets

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: cutmix_collate_fn(b, alpha=cutmix_alpha)
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, valid_loader