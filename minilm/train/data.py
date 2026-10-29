import numpy as np
import torch


def get_batch(data, batch_size, context_length, device):
    """data: 1-D array (can be np.load(..., mmap_mode='r')) of token IDs.
    Returns x, y of shape (B, context_length); y is x shifted right by one."""
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)