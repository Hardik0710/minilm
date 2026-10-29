import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotates each pair (x[2k], x[2k+1]) at position i by angle i / theta^(2k/d_head)."""

    def __init__(self, theta, d_head, max_len):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even"
        k = torch.arange(d_head // 2, dtype=torch.float32)
        inv_freq = 1.0 / theta ** (2 * k / d_head)              # (d_head/2,)
        pos = torch.arange(max_len, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None, :]                # (max_len, d_head/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions=None):
        """x: (..., seq, d_head). positions: optional int tensor (..., seq)."""
        if positions is None:
            seq = x.shape[-2]
            cos, sin = self.cos[:seq], self.sin[:seq]
        else:
            cos, sin = self.cos[positions], self.sin[positions]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]                      # even / odd entries
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack((out1, out2), dim=-1).flatten(-2)     # re-interleave