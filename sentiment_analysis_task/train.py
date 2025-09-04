import os
import re
import math
import random
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer

from .data import ReviewDataset
from .models import LSTMClassifier

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(text: str) -> str:
    # HTML 태그/줄바꿈 제거, 공백 정리
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------
# 1) 데이터 로더
# ------------------------------------------------------------
def make_dataloaders(tokenizer,
                     train_texts,
                     test_texts,
                     train_labels,
                     test_labels,
                     batch_size=128,
                     num_workers=4,
                     max_len=400):
    vocab = tokenizer.get_vocab()
    pad_idx = vocab["[PAD]"]
    train_ds = ReviewDataset(train_texts, train_labels, tokenizer, max_len)
    test_ds  = ReviewDataset(test_texts,  test_labels, tokenizer, max_len)
    
    def collate_fn(batch):
        ids_list, lengths, labels = [], [], []
        for ids, length, label in batch:
            ids_list.append(ids)
            lengths.append(length)
            labels.append(label)
        
        padded = pad_sequence(ids_list, batch_first=True, padding_value=pad_idx)
        lengths = torch.tensor(lengths, dtype=torch.long)
        labels = torch.stack(labels)  # float tensor [B]
        return padded, lengths, labels

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    return train_loader, test_loader


# ----------------------------
# 2) 학습/평가 루프
# ----------------------------
def binary_accuracy_from_logits(logits, y):
    # logits: raw score, y: {0,1} float
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, max_grad_norm = 1.0):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths   = lengths.to(device)
        labels    = labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            # gradient clipping 전에 unscale 필수
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += binary_accuracy_from_logits(logits.detach(), labels) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths   = lengths.to(device)
        labels    = labels.to(device)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += binary_accuracy_from_logits(logits, labels) * bs
        n += bs
    return total_loss / n, total_acc / n



def fit(
    model,
    train_loader,
    test_loader,
    epochs=20,
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    device=None,
    use_amp=True,
    scheduler_type="cosine",
    optimizer="AdamW",
):
    """
    동일한 학습 조건으로 모델을 학습/평가합니다.
    - scheduler_type: "cosine", "multistep", None
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



def run_experiment(
    model_builder,
    model_name: str,
    epochs=20,
    batch_size=128,
    lr=0.001,
    weight_decay=5e-4,
    momentum=0.9,
    scheduler_type="cosine",
    seed=42,
    num_workers=4,
    max_len=400,
):
    """
    동일한 설정으로 모델을 학습/평가하고 결과를 반환.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    pad_idx = vocab["[PAD]"]
    
    ds = load_dataset("imdb")
    train_texts = [clean_text(t) for t in ds["train"]['text']]
    test_texts  = [clean_text(t) for t in ds["test"]['text']]

    train_loader, test_loader = make_dataloaders(
        tokenizer=tokenizer,
        train_texts=train_texts,
        test_texts=test_texts,
        train_labels=ds["train"]['label'],
        test_labels=ds["test"]['label'],
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
    )

    model = model_builder(
        vocab_size=len(vocab),
        embed_dim=300,
        hidden_dim=256,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
        pad_idx=pad_idx,
    )
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
    ckpt_path = f"./{model_name}_imdb_best.pth"
    torch.save({"model": best_state, "meta": {"model_name": model_name}}, ckpt_path)
    print(f"Saved best checkpoint to: {ckpt_path}")
    return history, best_state


def build_lstm_classfication(vocab_size, 
                             embed_dim, 
                             hidden_dim, 
                             num_layers, 
                             bidirectional, 
                             dropout, 
                             pad_idx):
    return LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout, pad_idx)