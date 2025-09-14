import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std  = [0.2471, 0.2435, 0.261]
                
def get_cifar10_train_dataloader(
        data_path="./data",
        batch_size=128,
        num_workers=4,
        use_aug=True,
        img_size=32,
        valid_ratio=0.1):
    common_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        train_tf = common_tf

    full_train_set = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_tf
    )

    n_total = len(full_train_set)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_set, valid_set = random_split(full_train_set, [n_train, n_valid])

    # valid transform 교체
    valid_set.dataset.transform = common_tf

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, valid_loader



def get_cifar10_test_dataloader(
        data_path="./data",
        batch_size=128,
        num_workers=4,
        img_size=32):
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_set = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return test_loader