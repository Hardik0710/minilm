import torch.nn.functional as F


def lm_loss(logits, targets):
    """logits: (B, S, V), targets: (B, S) -> scalar mean cross-entropy."""
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))