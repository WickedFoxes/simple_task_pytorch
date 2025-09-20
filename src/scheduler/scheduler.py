from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR, CosineAnnealingWarmRestarts
from src.registry import register

@register("scheduler", "cosine")
def cosine(optimizer, max_epochs, min_lr=1e-6):
    # T_max는 에폭 기준, 스텝 기준 사용 시 CosineAnnealingWarmRestarts 등으로 변경
    return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)

@register("scheduler", "step")
def step(optimizer, step_size, gamma=0.1):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

@register("scheduler", "onecycle")
def onecycle(optimizer, max_lr, steps_per_epoch, epochs, pct_start=0.3):
    return OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                      epochs=epochs, pct_start=pct_start)

@register("scheduler", "cosine_warm_restarts")
def cosine_warm_restarts(optimizer, first_cycle_steps, cycle_mult=1, min_lr=1e-6):
    return CosineAnnealingWarmRestarts(optimizer, T_0=first_cycle_steps, T_mult=cycle_mult, eta_min=min_lr)