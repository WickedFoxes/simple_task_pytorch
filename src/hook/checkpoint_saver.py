import math, os
from src.utils.checkpoint import save
from src.hook.base import Hook

class CheckpointSaver(Hook):
    def __init__(self, save_dir="checkpoints", monitor="val/acc", mode="max",
                 save_best=True, save_last=False, filename="epoch{epoch:03d}.ckpt", min_delta=1e-4):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.filename = filename
        self.best = -math.inf if mode == "max" else math.inf
        self.last_path = None
        self.min_delta = min_delta

    def _is_improved(self, current):
        return (current > self.best + self.min_delta) if self.mode == "max" else (current < self.best - self.min_delta)

    def on_validation_end(self, metric, epoch, model, optimizer, scheduler, logger=None, **kw):
        # metric은 Trainer에서 val/acc과 같은 monitor에 해당하는 값을 넘겨주도록 연결
        current = metric
        # 1) best 저장
        if self.save_best and self._is_improved(current):
            self.best = current
            path = os.path.join(self.save_dir, "best.ckpt")
            save(path, model, optimizer, scheduler, epoch, best_metric=current)
            if logger: logger.log_metrics({"ckpt/best_epoch": epoch, "ckpt/best_metric": current}, step=kw.get("epoch", epoch))
        # 2) last 저장(에폭마다)
        if self.save_last:
            path = os.path.join(self.save_dir, self.filename.format(epoch=epoch))
            save(path, model, optimizer, scheduler, epoch, best_metric=self.best)
            self.last_path = path

    def on_step_end(self, model, optimizer, scheduler, epoch, **kw):
        # last 저장
        if self.save_last:
            path = os.path.join(self.save_dir, "last.ckpt")
            save(path, model, optimizer, scheduler, epoch, best_metric=self.best)
            self.last_path = path

    def on_train_end(self, model, optimizer, scheduler, epoch, **kw):
        # 종료 시점에 마지막 스냅샷을 한 번 더(선택)
        if self.save_last and self.last_path is None:
            path = os.path.join(self.save_dir, "last.ckpt")
            save(path, model, optimizer, scheduler, epoch, best_metric=self.best)
