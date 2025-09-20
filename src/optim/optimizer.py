import torch.optim as optim
from src.registry import register

@register("optimizer", "adamw")
def build_adamw(params, lr, weight_decay, betas=(0.9,0.999)):
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)

@register("optimizer", "sgd")
def build_sgd(params, lr, momentum=0.9, weight_decay=0.0, nesterov=True):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

@register("optimizer", "adam")
def build_adam(params, lr, weight_decay):
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)