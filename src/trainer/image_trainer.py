import torch
from torch.cuda.amp import autocast, GradScaler

from src.registry import register

@register("trainer", "image_trainer")
class ImageTrainer:
    def __init__(self, model, optimizer, scheduler, logger, hooks=[], cfg=None):
        self.model = model; self.opt = optimizer; self.sched = scheduler
        self.logger = logger; self.hooks = hooks; self.cfg = cfg
        self.scaler = GradScaler(enabled=cfg["amp"])

    def train(self, train_loader, val_loader, criterion, device):
        self.model.to(device)
        for h in self.hooks: h.on_train_start()
        global_step = 0

        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                self.opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.cfg["amp"]):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                self.scaler.scale(loss).backward()
                if self.cfg["grad_clip"] is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()
                if self.sched and hasattr(self.sched, "step") and not isinstance(self.sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.sched.step()  # 스텝 단위 스케줄러일 경우

                global_step += 1

            # 에폭 단위 스케줄러일 경우
            if self.sched and isinstance(self.sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                self.sched.step()

            # 검증
            val_acc, val_loss = self.evaluate(val_loader, device, criterion)
            self.logger.log_metrics({"val/acc": val_acc, "val/loss": val_loss, "epoch": epoch}, step=global_step)

            for h in self.hooks:
                h.on_validation_end(metric=val_acc, epoch=epoch)

            # 조기 종료
            if any(getattr(h, "should_stop", False) for h in self.hooks):
                break

            for h in self.hooks: h.on_epoch_end(epoch=epoch)

        for h in self.hooks: h.on_train_end()

    @torch.no_grad()
    def evaluate(self, loader, device, criterion):
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = self.model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        return correct/total, total_loss/total
