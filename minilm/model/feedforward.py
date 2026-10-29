import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU: W2( SiLU(W1 x) * W3 x )."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))