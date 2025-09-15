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