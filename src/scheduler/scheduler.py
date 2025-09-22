from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    OneCycleLR,
    LambdaLR,
    SequentialLR,
    ReduceLROnPlateau,
)
from transformers import get_cosine_schedule_with_warmup
from src.registry import register

@register("scheduler", "cosine")
def cosine(optimizer, max_epochs, min_lr=1e-6):
    # T_max는 에폭 기준, 스텝 기준 사용 시 CosineAnnealingWarmRestarts 등으로 변경
    return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)

@register("scheduler", "cosine_warmup")
def cosine_warmup(optimizer, max_epochs, warmup_epochs=2, min_lr=1e-6):
    def warmup_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # 2단계: cosine decay
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs-warmup_epochs, eta_min=min_lr)

    # 1단계 warmup 끝나면 cosine 이어서 실행
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])
    return scheduler

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


@register("scheduler", "reduce_lr_on_plateau")
def Reduce_lr_on_plateau(
    optimizer, 
    mode='min',
    factor=0.1, 
    patience=3,
):
    return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)