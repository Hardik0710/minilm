import math
import torch
import torch.nn.functional as F


def slow_attention(q, k, v):
    """softmax(Q K^T / sqrt(d_k)) V with a causal mask.

    q, k, v: (..., seq, d_k). Works for 3-D (batch, seq, d) and
    4-D (batch, heads, seq, d) inputs.
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # (..., seq, seq)
    seq = q.shape[-2]
    # True above the diagonal = future positions
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v