
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from src.registry import register

@register("trainer", "classifier_trainer")
class ClassifierTrainer:
    def __init__(self, model, optimizer, scheduler, logger, hooks=[], batch_aug=None, cfg=None):
        self.model = model; self.opt = optimizer; self.sched = scheduler
        self.logger = logger; self.hooks = hooks; self.cfg = cfg or {}
        self.scaler = GradScaler(enabled=self.cfg.get("amp", True))
        self.batch_aug = batch_aug


    def train(self, train_loader, val_loader, criterion, device):
        self.model.to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log_params({"total_params": total_params})

        for h in self.hooks:
            if hasattr(h, "on_train_start"):
                h.on_train_start()

        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()

            epoch_start = time.time()
            train_loss, train_acc, n = 0.0, 0.0, 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                if self.batch_aug is not None:
                    inputs, labels, _ = self.batch_aug(inputs, labels)
                
                self.opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.cfg["amp"]):
                    logits = self.model(inputs)
                    loss = criterion(logits, labels)
                
                self.scaler.scale(loss).backward()

                if self.cfg["grad_clip"] is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                
                self.scaler.step(self.opt)
                self.scaler.update()
                
                # 스텝 단위 스케줄러일 경우
                if self.sched and hasattr(self.sched, "step"):
                    if isinstance(self.sched, torch.optim.lr_scheduler.OneCycleLR):
                        self.sched.step()
                    elif not isinstance(self.sched, (
                            torch.optim.lr_scheduler.ReduceLROnPlateau, 
                            torch.optim.lr_scheduler.CosineAnnealingLR, 
                            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                    )):
                        self.sched.step() 
                
                train_loss += loss.item() * batch_size
                train_acc  += self.accuracy(logits, labels) * batch_size
                n += batch_size
                
                if hasattr(self.cfg, "log_step") and hasattr(self.logger, "log_metrics"):
                    log_step = self.cfg["log_step"]
                    if n % log_step == 0:
                        elapsed = time.time() - epoch_start
                        self.logger.log_metrics(
                            {
                                "step": n//batch_size,
                                "train_loss": train_loss/n,
                                "train_acc": train_acc/n,
                                "lr": float(current_lr) if current_lr is not None else float("nan"),
                                "elapsed_time": elapsed,
                            },
                        )
                        for h in self.hooks:
                            if hasattr(h, "on_step_end"):
                                h.on_step_end(
                                    model=self.model,
                                    optimizer=self.opt,
                                    scheduler=self.sched,
                                    epoch=epoch,
                                )

            train_loss /= n; train_acc /= n
            current_lr = self.opt.param_groups[0]["lr"]

            # 검증
            val_loss, val_acc  = self.evaluate(self.model, val_loader, device, criterion)

            # 에폭 단위 스케줄러일 경우
            if self.sched:
                if isinstance(self.sched, (
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                )):
                    self.sched.step()
                elif isinstance(self.sched, (
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                )):
                    self.sched.step(val_loss)

            
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

    @torch.no_grad()
    def evaluate(self, model, loader, device, criterion):
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
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