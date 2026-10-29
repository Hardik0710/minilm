from minilm.tokenizer.bpe import train_bpe

SPECIAL = ["<|endoftext|>"]


def _toy(tmp_path):
    # One word per line so the GPT-2 regex doesn't attach a leading space
    # to each word; "\n" becomes its own pre-token and makes no pairs.
    words = ["low"] * 5 + ["lower"] * 2 + ["widest"] * 3 + ["newest"] * 6
    path = tmp_path / "toy.txt"
    path.write_text("\n".join(words), encoding="utf-8")
    return path


def test_toy_example(tmp_path):
    # 256 bytes + 1 special token + 6 merges = 263
    vocab, merges = train_bpe(_toy(tmp_path), vocab_size=263, special_tokens=SPECIAL)
    expected = [
        (b"s", b"t"), (b"e", b"st"), (b"o", b"w"),
        (b"l", b"ow"), (b"w", b"est"), (b"n", b"e"),
    ]
    assert merges[:6] == expected


def test_vocab_size_and_unique_ids(tmp_path):
    vocab, merges = train_bpe(_toy(tmp_path), vocab_size=263, special_tokens=SPECIAL)
    assert len(vocab) == 263
    assert len(set(vocab.keys())) == len(vocab)
    assert len(merges) == 6
    assert b"<|endoftext|>" in vocab.values()


def test_no_cross_story_merges(tmp_path):
    story1 = "abab abab abab ab ba"
    story2 = "cdcd cdcd cdcd cd dc"
    path = tmp_path / "two.txt"
    path.write_text(story1 + "<|endoftext|>" + story2, encoding="utf-8")
    vocab, merges = train_bpe(path, vocab_size=290, special_tokens=SPECIAL)
    for idx, tok in vocab.items():
        if idx <= 256:
            continue  # base bytes and the special token
        has1 = bool(set(tok) & set(b"ab"))
        has2 = bool(set(tok) & set(b"cd"))
        assert not (has1 and has2), tok