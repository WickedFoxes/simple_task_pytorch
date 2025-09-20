from src.registry import register
from src.logger.base import LoggerBase

@register("logger", "tensorboard")
class TBLogger(LoggerBase):
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.w = SummaryWriter(log_dir)
    def log_metrics(self, d, step):
        for k,v in d.items(): self.w.add_scalar(k, v, step)
    def log_params(self, d): pass
    def finalize(self): self.w.close()