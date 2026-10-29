import torch
from minilm.model.transformer import TransformerLM
from minilm.tokenizer.tokenizer import Tokenizer
from minilm.generate import generate


def _setup():
    torch.manual_seed(0)
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], ["<|endoftext|>"])
    model = TransformerLM(257, 32, 32, 2, 4, 64, 10000.0)
    return model, tok


def test_greedy_is_deterministic():
    model, tok = _setup()
    a = generate(model, tok, "Once", 20, temperature=0)
    b = generate(model, tok, "Once", 20, temperature=0)
    assert a == b


def test_seeded_sampling_reproducible():
    model, tok = _setup()
    torch.manual_seed(5)
    a = generate(model, tok, "Once", 20, temperature=0.8, top_p=0.9)
    torch.manual_seed(5)
    b = generate(model, tok, "Once", 20, temperature=0.8, top_p=0.9)
    assert a == b


def test_long_generation_past_context():
    model, tok = _setup()
    out = generate(model, tok, "Once", 60, temperature=0)   # > context 32
    assert isinstance(out, str)