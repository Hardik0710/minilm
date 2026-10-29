import io
from minilm.tokenizer.tokenizer import Tokenizer

SPECIAL = ["<|endoftext|>"]


def _toy():
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
             6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    return Tokenizer(vocab, merges)


def _full():
    vocab = {i: bytes([i]) for i in range(256)}
    return Tokenizer(vocab, [], SPECIAL)


def test_hand_worked_example():
    assert _toy().encode("the cat ate") == [9, 7, 1, 5, 10, 3]


def test_roundtrip():
    tok = _full()
    for s in ["Hello world", "नमस्ते दुनिया", "emoji 😀🎉", "", "line1\nline2\n\n  x"]:
        assert tok.decode(tok.encode(s)) == s


def test_special_token_single_id():
    tok = _full()
    ids = tok.encode("a<|endoftext|>b")
    sid = tok.bytes_to_id[b"<|endoftext|>"]
    assert ids.count(sid) == 1
    assert ids == [ord("a"), sid, ord("b")]
    assert tok.decode(ids) == "a<|endoftext|>b"


def test_overlapping_specials_prefer_longer():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], ["<|a|>", "<|a|><|a|>"])
    ids = tok.encode("<|a|><|a|>")
    assert len(ids) == 1


def test_encode_iterable_matches_encode():
    tok = _full()
    text = "Once upon a time.\nThere was a cat.<|endoftext|>\nNew story here.\n"
    assert list(tok.encode_iterable(io.StringIO(text))) == tok.encode(text)


def test_invalid_utf8_decodes_with_replacement():
    tok = _full()
    assert tok.decode([0xFF]) == "\ufffd"