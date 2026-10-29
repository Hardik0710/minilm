import torch
from minilm.train.loss import lm_loss


def train_step(model, optimizer, scheduler, scaler, x, y, use_amp=False, clip=1.0):
    """Returns (loss, grad_norm). Gradients are unscaled BEFORE clipping."""
    device_type = x.device.type
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
        logits = model(x)
    loss = lm_loss(logits.float(), y)          # loss computed in fp32
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                 # must come before clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item(), grad_norm.item()