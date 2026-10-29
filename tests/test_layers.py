import torch
import torch.nn.functional as F
from minilm.model.attention import CausalSelfAttention, slow_attention
from minilm.model.feedforward import FeedForward


def test_attention_shapes_with_extra_batch_dims():
    attn = CausalSelfAttention(64, 4, max_len=32)
    for shape in [(10, 64), (2, 10, 64), (3, 2, 10, 64)]:
        x = torch.randn(*shape)
        assert attn(x).shape == x.shape


def test_attention_cannot_see_future():
    torch.manual_seed(0)
    attn = CausalSelfAttention(64, 4, max_len=32)
    x1 = torch.randn(2, 12, 64)
    x2 = x1.clone()
    t = 6
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])   # change tokens after t
    y1, y2 = attn(x1), attn(x2)
    assert torch.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-6)
    assert not torch.allclose(y1[:, t + 1:], y2[:, t + 1:])


def test_attention_matches_manual_computation():
    torch.manual_seed(0)
    d, h, s = 64, 4, 10
    attn = CausalSelfAttention(d, h, max_len=32)
    x = torch.randn(2, s, d)
    dh = d // h

    def split(t):
        return t.view(2, s, h, dh).transpose(1, 2)

    q = attn.rope(split(attn.q_proj(x)))
    k = attn.rope(split(attn.k_proj(x)))
    v = split(attn.v_proj(x))                      # no RoPE on v
    out = slow_attention(q, k, v).transpose(1, 2).reshape(2, s, d)
    expected = attn.out_proj(out)
    assert torch.allclose(attn(x), expected, atol=1e-5)


def test_attention_has_no_bias_and_4_projections():
    attn = CausalSelfAttention(64, 4)
    names = [n for n, _ in attn.named_parameters()]
    assert len(names) == 4 and not any("bias" in n for n in names)


def test_feedforward_shapes():
    ff = FeedForward(64, 128)
    for shape in [(10, 64), (2, 10, 64), (3, 2, 10, 64)]:
        x = torch.randn(*shape)
        assert ff(x).shape == x.shape


def test_feedforward_matches_formula():
    torch.manual_seed(0)
    ff = FeedForward(32, 64)
    x = torch.randn(5, 32)
    expected = (F.silu(x @ ff.w1.weight.T) * (x @ ff.w3.weight.T)) @ ff.w2.weight.T
    assert torch.allclose(ff(x), expected, atol=1e-5)


def test_feedforward_param_count():
    ff = FeedForward(512, 1344)
    assert sum(p.numel() for p in ff.parameters()) == 3 * 512 * 1344