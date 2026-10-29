import math
import torch


def make_optimizer(model, lr, betas=(0.9, 0.95), weight_decay=0.1):
    """Matrices (ndim >= 2, including the embedding) get weight decay.
    RMSNorm weights (ndim 1) get none. Every parameter lands in exactly one group."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(t, a_max, a_min, t_warm, t_cos):
    if t < t_warm:
        return t / t_warm * a_max
    if t <= t_cos:
        progress = (t - t_warm) / (t_cos - t_warm)
        return a_min + 0.5 * (1 + math.cos(math.pi * progress)) * (a_max - a_min)
    return a_min


def make_scheduler(optimizer, a_max, a_min, t_warm, t_cos):
    """The optimizer must be created with lr=a_max; LambdaLR multiplies it."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: lr_at(t, a_max, a_min, t_warm, t_cos) / a_max
    )