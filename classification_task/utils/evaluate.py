import torch
import torch.nn as nn

def accuracy(logits, targets):
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()

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
        total_acc  += accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, total_acc / n