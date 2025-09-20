import torch, os

def save(path, model, optimizer, scheduler, epoch, best_metric=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)

def load(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt