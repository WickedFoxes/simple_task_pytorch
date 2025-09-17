import torch

def build_scheduler(optimizer : torch.optim.Optimizer,
                    scheduler_type : str ="cosine",
                    epochs : int =50,):
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs*0.5), int(epochs*0.75)],
            gamma=0.1
        )
    else:
        scheduler = None
    return scheduler


