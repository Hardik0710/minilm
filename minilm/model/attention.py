import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from minilm.model.rope import RoPE

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

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope_theta=10000.0, max_len=256):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(rope_theta, self.d_head, max_len)

    def forward(self, x):
        """x: (..., seq, d_model) -> same shape."""
        h = self.num_heads
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=h)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=h)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=h)
        q, k = self.rope(q), self.rope(k)          # RoPE on q and k only, not v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.out_proj(out)