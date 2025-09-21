
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from src.registry import register

@register("trainer", "text_classifier_trainer")
class TextClassifierTrainer:
    def __init__(self, model, optimizer, scheduler, logger, hooks=[], cfg=None):
        self.model = model; self.opt = optimizer; self.sched = scheduler
        self.logger = logger; self.hooks = hooks; self.cfg = cfg or {}
        self.scaler = GradScaler(enabled=self.cfg.get("amp", False))


    def train(self, train_loader, val_loader, criterion, device):
        self.model.to(device)
        for h in self.hooks:
            if hasattr(h, "on_train_start"):
                h.on_train_start()

        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()

            epoch_start = time.time()
            train_loss, train_acc, n = 0.0, 0.0, 0
            
            for input_ids, lengths, labels in train_loader:
                input_ids = input_ids.to(device)
                lengths   = lengths.to(device)
                labels    = labels.to(device)
                
                self.opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.cfg["amp"]):
                    logits = self.model(input_ids, lengths)
                    loss = criterion(logits, labels)
                
                self.scaler.scale(loss).backward()

                if self.cfg["grad_clip"] is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()

                if self.sched and hasattr(self.sched, "step") and not isinstance(self.sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.sched.step()  # 스텝 단위 스케줄러일 경우

                train_loss += loss.item() * labels.size(0)
                train_acc  += self.accuracy(logits, labels) * labels.size(0)
                n += labels.size(0)

            train_loss /= n; train_acc /= n
            current_lr = self.opt.param_groups[0]["lr"]


            # 에폭 단위 스케줄러일 경우
            if self.sched and isinstance(self.sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                self.sched.step()

            # 검증
            val_loss, val_acc  = self.evaluate(self.model, val_loader, device, criterion)
            elapsed = time.time() - epoch_start

            if hasattr(self.logger, "log_metrics"):
                self.logger.log_metrics(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                        "lr": float(current_lr) if current_lr is not None else float("nan"),
                        "elapsed_time": elapsed,
                    },
                )

            for h in self.hooks:
                if hasattr(h, "on_validation_end"):
                    h.on_validation_end(
                        metric=val_acc,
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.opt,
                        scheduler=self.sched,
                        logger=self.logger,
                    )

            # 조기 종료
            if any(getattr(h, "should_stop", False) for h in self.hooks):
                break

            for h in self.hooks:
                if hasattr(h, "on_epoch_end"):
                    h.on_epoch_end(epoch=epoch)

        for h in self.hooks:
            if hasattr(h, "on_train_end"):
                h.on_train_end(self.model, self.opt, self.sched, epoch)
        
        if hasattr(self.logger, "finalize"):
            self.logger.finalize(status="success")

    @torch.no_grad()
    def evaluate(self, model, loader, device, criterion):
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
            total_acc  += self.accuracy(logits, labels) * bs
            n += bs
        return total_loss / n, total_acc / n

    @torch.no_grad()
    def accuracy(self, logits, targets):
        # logits: [B, C] (CrossEntropyLoss 기준)
        # targets: [B] (인덱스) 또는 [B, C] (one-hot)
        if targets.ndim > 1:              # one-hot or soft label
            targets = targets.argmax(dim=1)
        else:
            targets = targets.long()
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()