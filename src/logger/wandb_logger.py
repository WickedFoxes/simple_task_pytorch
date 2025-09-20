from src.registry import register
from src.logger.base import LoggerBase

@register("logger", "wandb")
class WandbLogger(LoggerBase):
    def __init__(self, project, config=None):
        import wandb
        self.wb = wandb.init(project=project, config=config, reinit=True)
    def log_metrics(self, d, step): self.wb.log(d, step=step)
    def log_params(self, d): pass
    def finalize(self): self.wb.finish()