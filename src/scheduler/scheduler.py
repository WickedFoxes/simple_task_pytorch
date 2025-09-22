from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    OneCycleLR,
    LambdaLR,
    SequentialLR,
    ReduceLROnPlateau,
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
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

@register("scheduler", "linear_schedule_with_warmup")
def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

@register("scheduler", "cosine_schedule_with_warmup")
def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7):
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)


@register("scheduler", "reduce_lr_on_plateau")
def Reduce_lr_on_plateau(
    optimizer, 
    mode='min',
    factor=0.1, 
    patience=3,
):
    return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)