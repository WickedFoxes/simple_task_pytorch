
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from src.registry import register

@register("trainer", "translate_trainer")
class TranslateTrainer:
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
            
            for src_ids, tgt_in_ids, tgt_out_ids in train_loader:
                src_ids = src_ids.to(device)
                tgt_in_ids = tgt_in_ids.to(device)
                tgt_out_ids = tgt_out_ids.to(device)
                batch_size = src_ids.size(0)
                
                self.opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.cfg["amp"]):
                    logits = self.model(src_ids, tgt_in_ids)
                    B, T, V = logits.size()
                    loss = criterion(logits.reshape(B*T, V), tgt_out_ids.reshape(B*T))
                
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
                train_acc  += self.accuracy(logits, tgt_out_ids, pad_id=self.cfg["tgt_pad_id"]) * batch_size
                n += batch_size

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
        total = 0.0
        for src_ids, tgt_in_ids, tgt_out_ids in loader:
            src_ids = src_ids.to(device)
            tgt_in_ids = tgt_in_ids.to(device)
            tgt_out_ids = tgt_out_ids.to(device)
            logits = model(src_ids, tgt_in_ids, src_pad_id=self.cfg["src_pad_id"], tgt_pad_id=self.cfg["tgt_pad_id"])
            B, T, V = logits.size()
            loss = criterion(logits.reshape(B*T, V), tgt_out_ids.reshape(B*T))
            total += loss.item()
        return total / len(loader)


    @torch.no_grad()
    def accuracy(self, logits, targets, pad_id=None):
        """
        logits : (B, T, V)
        targets: (B, T)
        pad_id : PAD 토큰 인덱스 (무시)
        """
        # (B, T)
        preds = logits.argmax(dim=-1)
        
        # pad 제외
        if pad_id is not None:
            mask = (targets != pad_id)
            correct = (preds == targets) & mask
            acc = correct.sum().float() / mask.sum().float()
        else:
            acc = (preds == targets).float().mean()
        
        return acc.item()