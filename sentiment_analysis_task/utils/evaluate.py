import torch
import torch.nn as nn

import torch

def accuracy(logits, targets):
    with torch.no_grad():
        # logits가 (outputs, ...) 형태의 튜플일 가능성 처리
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # targets 차원 정리: [B,1] -> [B]
        if targets.ndim == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if logits.ndim == 2:
            # [B, C] 또는 [B, 1]
            if logits.size(1) == 1:
                # 이진분류: 로짓 -> 시그모이드 -> 0.5 임계값
                preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
                # targets을 정수 0/1로 맞춤
                targets_i = targets.long() if targets.dtype != torch.long else targets
            else:
                # 다중분류
                preds = logits.argmax(dim=1)
                targets_i = targets.long()
        elif logits.ndim == 1:
            # 이진분류: [B]
            preds = (torch.sigmoid(logits) >= 0.5).long()
            targets_i = targets.long() if targets.dtype != torch.long else targets
        else:
            raise ValueError(f"Unsupported logits ndim: {logits.ndim}, shape={tuple(logits.shape)}")

        # 길이 불일치 방지
        if preds.shape != targets_i.shape:
            # 예: targets가 [B,1]로 남아있는 경우 등
            targets_i = targets_i.view_as(preds)

        return (preds == targets_i).float().mean().item()


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
        total_acc  += accuracy(logits, labels) * bs
        n += bs
    return total_loss / n, total_acc / n