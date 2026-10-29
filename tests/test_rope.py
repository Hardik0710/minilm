import torch
from minilm.model.rope import RoPE

THETA, D, MAXLEN = 10000.0, 32, 64


def _rotation_matrix(i, theta=THETA, d=D):
    """Slow reference: full block-diagonal rotation matrix for position i."""
    R = torch.zeros(d, d)
    for k in range(d // 2):
        ang = i / theta ** (2 * k / d)
        c, s = torch.tensor(ang).cos(), torch.tensor(ang).sin()
        R[2 * k, 2 * k], R[2 * k, 2 * k + 1] = c, -s
        R[2 * k + 1, 2 * k], R[2 * k + 1, 2 * k + 1] = s, c
    return R


def test_matches_rotation_matrix():
    torch.manual_seed(0)
    rope = RoPE(THETA, D, MAXLEN)
    x = torch.randn(10, D)
    expected = torch.stack([_rotation_matrix(i) @ x[i] for i in range(10)])
    assert torch.allclose(rope(x), expected, atol=1e-5)


def test_shapes_with_extra_batch_dims():
    rope = RoPE(THETA, D, MAXLEN)
    for shape in [(10, D), (2, 10, D), (2, 4, 10, D), (3, 2, 4, 10, D)]:
        x = torch.randn(*shape)
        assert rope(x).shape == x.shape


def test_position_zero_is_identity():
    rope = RoPE(THETA, D, MAXLEN)
    x = torch.randn(1, D)
    assert torch.allclose(rope(x), x, atol=1e-6)


def test_preserves_norm():
    rope = RoPE(THETA, D, MAXLEN)
    x = torch.randn(2, 10, D)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_explicit_positions_match_default():
    rope = RoPE(THETA, D, MAXLEN)
    x = torch.randn(2, 10, D)
    pos = torch.arange(10).expand(2, 10)
    assert torch.allclose(rope(x), rope(x, pos), atol=1e-6)


def test_buffers_not_in_state_dict():
    assert len(RoPE(THETA, D, MAXLEN).state_dict()) == 0