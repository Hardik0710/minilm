import torch
import torch.nn.functional as F
from minilm.model.attention import slow_attention


def test_matches_sdpa_3d():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 10, 32) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(slow_attention(q, k, v), expected, atol=1e-5)


def test_matches_sdpa_4d():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 10, 32) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(slow_attention(q, k, v), expected, atol=1e-5)


def test_cannot_see_future():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 8, 16) for _ in range(3))
    out1 = slow_attention(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, 5:] = torch.randn_like(k2[:, 5:])   # change positions 5 and later
    v2[:, 5:] = torch.randn_like(v2[:, 5:])
    out2 = slow_attention(q, k2, v2)
    assert torch.allclose(out1[:, :5], out2[:, :5])