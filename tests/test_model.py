import math
import torch
import torch.nn.functional as F
from minilm.model.transformer import TransformerLM

CFG = dict(vocab_size=10_000, context_length=256, d_model=512,
           num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)


def _small():
    return TransformerLM(vocab_size=100, context_length=32, d_model=64,
                         num_layers=2, num_heads=4, d_ff=128, rope_theta=10000.0)


def test_logits_shape():
    model = TransformerLM(**CFG)
    ids = torch.randint(0, 10_000, (2, 17))
    assert model(ids).shape == (2, 17, 10_000)


def test_cannot_see_future():
    torch.manual_seed(0)
    model = _small().eval()
    a = torch.randint(0, 100, (2, 20))
    b = a.clone()
    t = 9
    b[:, t + 1:] = torch.randint(0, 100, (2, 20 - t - 1))   # change tokens after t
    with torch.no_grad():
        ya, yb = model(a), model(b)
    assert torch.allclose(ya[:, : t + 1], yb[:, : t + 1], atol=1e-6)


def test_initial_loss_near_log_vocab():
    # With near-uniform initial logits, each of the V classes gets probability
    # about 1/V, so cross-entropy = -log(1/V) = ln(V) = ln(10000) ~ 9.21.
    torch.manual_seed(0)
    model = TransformerLM(**CFG).eval()
    ids = torch.randint(0, 10_000, (4, 64))
    with torch.no_grad():
        logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 10_000), ids[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(10_000)) < 0.5


def test_uses_modulelist_and_rmsnorm():
    model = _small()
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert len(model.blocks) == 2
    assert isinstance(model.final_norm, torch.nn.RMSNorm)


def test_param_count_matches_hand_count():
    model = TransformerLM(**CFG)
    V, d, L, ff = 10_000, 512, 4, 1344
    expected = (V * d                       # embedding
                + L * (4 * d * d            # q, k, v, o
                       + 3 * d * ff         # w1, w2, w3
                       + 2 * d)             # two RMSNorm weights per block
                + d                         # final RMSNorm
                + d * V)                    # lm_head
    assert sum(p.numel() for p in model.parameters()) == expected