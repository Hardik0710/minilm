import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from minilm.model.transformer import TransformerLM
from minilm.train.loss import lm_loss
from minilm.train.optim import make_optimizer, make_scheduler, lr_at
from minilm.train.data import get_batch
from minilm.train.step import train_step
from minilm.train.checkpoint import save_checkpoint, load_checkpoint


def small_model():
    return TransformerLM(vocab_size=50, context_length=16, d_model=32,
                         num_layers=2, num_heads=4, d_ff=64, rope_theta=10000.0)


def test_loss_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 7)
    targets = torch.randint(0, 7, (3, 5))
    logp = torch.log_softmax(logits, dim=-1)
    manual = -logp.gather(-1, targets[..., None]).mean()
    assert torch.allclose(lm_loss(logits, targets), manual, atol=1e-6)


def test_optimizer_groups_cover_every_param_once():
    model = small_model()
    opt = make_optimizer(model, lr=1e-3)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.ndim == 1 for p in opt.param_groups[1]["params"])


def test_lr_schedule_hand_values():
    a_max, a_min, tw, tc = 1e-3, 1e-4, 10, 110
    assert lr_at(0, a_max, a_min, tw, tc) == 0.0
    assert lr_at(5, a_max, a_min, tw, tc) == pytest.approx(5e-4)
    assert lr_at(10, a_max, a_min, tw, tc) == pytest.approx(1e-3)
    assert lr_at(60, a_max, a_min, tw, tc) == pytest.approx(5.5e-4)  # midpoint
    assert lr_at(110, a_max, a_min, tw, tc) == pytest.approx(1e-4)
    assert lr_at(500, a_max, a_min, tw, tc) == pytest.approx(1e-4)


def test_scheduler_follows_function():
    model = small_model()
    a_max, a_min, tw, tc = 1e-3, 1e-4, 10, 110
    opt = make_optimizer(model, lr=a_max)
    sch = make_scheduler(opt, a_max, a_min, tw, tc)
    for t in range(130):
        assert opt.param_groups[0]["lr"] == pytest.approx(lr_at(t, a_max, a_min, tw, tc))
        opt.step()
        sch.step()


def test_grad_clipping_bounds_norm():
    torch.manual_seed(0)
    model = small_model()
    x = torch.randint(0, 50, (4, 16))
    y = torch.randint(0, 50, (4, 16))
    (lm_loss(model(x), y) * 1000).backward()          # make gradients huge
    before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert before > 1.0
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
    assert total <= 1.0 + 1e-4


def test_get_batch_shift_and_bounds():
    data = np.arange(1000, dtype=np.uint16)
    for _ in range(200):
        x, y = get_batch(data, 8, 16, "cpu")
        assert x.shape == y.shape == (8, 16)
        assert torch.equal(y[:, :-1], x[:, 1:])        # y is x shifted by one
        assert (y[:, -1] == x[:, -1] + 1).all()        # data[i]=i, so next is +1
        assert y.max() <= 999                          # never past the end


def test_get_batch_works_on_tiny_array():
    data = np.arange(18, dtype=np.uint16)
    x, y = get_batch(data, 4, 16, "cpu")               # only starts 0..1 are valid
    assert y.max() <= 17


def _run_steps(model, opt, sch, scaler, batches):
    return [train_step(model, opt, sch, scaler, x, y)[0] for x, y in batches]


def test_checkpoint_resume_reproduces_losses(tmp_path):
    torch.manual_seed(0)
    batches = [(torch.randint(0, 50, (4, 16)), torch.randint(0, 50, (4, 16)))
               for _ in range(10)]

    def build():
        torch.manual_seed(1)
        m = small_model()
        o = make_optimizer(m, lr=1e-3)
        s = make_scheduler(o, 1e-3, 1e-4, 3, 10)
        sc = torch.amp.GradScaler("cuda", enabled=False)
        return m, o, s, sc

    m, o, s, sc = build()
    _run_steps(m, o, s, sc, batches[:5])
    path = tmp_path / "ck.pt"
    save_checkpoint(path, m, o, s, sc, step=5)
    original = _run_steps(m, o, s, sc, batches[5:])

    m2, o2, s2, sc2 = build()
    assert load_checkpoint(path, m2, o2, s2, sc2) == 5
    resumed = _run_steps(m2, o2, s2, sc2, batches[5:])
    assert np.allclose(original, resumed, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fp16_loss_close_to_fp32():
    torch.manual_seed(0)
    model = small_model().cuda()
    x = torch.randint(0, 50, (4, 16)).cuda()
    y = torch.randint(0, 50, (4, 16)).cuda()
    with torch.no_grad():
        l32 = lm_loss(model(x), y)
        with torch.autocast("cuda", dtype=torch.float16):
            l16 = lm_loss(model(x).float(), y)
    assert abs(l32.item() - l16.item()) < 0.05