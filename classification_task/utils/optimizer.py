import torch
import torch.nn as nn
def build_optimizer(model : nn.Module, 
                    lr : float = 0.1, 
                    weight_decay : float = 5e-4, 
                    momentum : float = 0.9, 
                    optimizer : str = 'SGD'):    
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer