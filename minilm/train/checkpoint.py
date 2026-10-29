import torch


def save_checkpoint(path, model, optimizer, scheduler, scaler, step):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, map_location=None):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    return ck["step"]