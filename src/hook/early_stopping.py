from src.hook.base import Hook

class EarlyStopping(Hook):
    def __init__(self, patience, mode="max", min_delta=1e-4):
        self.patience = patience; self.mode = mode
        self.best = -float("inf") if mode=="max" else float("inf")
        self.wait = 0; self.should_stop = False
        self.min_delta = min_delta

    def _is_improved(self, current):
        return (current > self.best + self.min_delta) if self.mode == "max" else (current < self.best - self.min_delta)

    def on_validation_end(self, metric, **kw):
        if self._is_improved(metric):
            self.best = metric; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience: self.should_stop = True