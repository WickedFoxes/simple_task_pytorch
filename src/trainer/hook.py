class Hook:
    def on_train_start(self, **kw): pass
    def on_epoch_end(self, **kw): pass
    def on_validation_end(self, **kw): pass
    def on_train_end(self, **kw): pass

class EarlyStopping(Hook):
    def __init__(self, patience, mode="max"):
        self.patience = patience; self.mode = mode
        self.best = -float("inf") if mode=="max" else float("inf")
        self.wait = 0; self.should_stop = False

    def on_validation_end(self, metric, **kw):
        improved = (metric > self.best) if self.mode=="max" else (metric < self.best)
        if improved:
            self.best = metric; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience: self.should_stop = True
