import regex
from collections import defaultdict, Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens):
    """Return (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])."""
    # 1. Base vocab: 256 bytes, then special tokens.
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 2. Split on special tokens so merges never cross stories.
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        sep = "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        chunks = regex.split(sep, text)
    else:
        chunks = [text]

    # 3. Pre-tokenize and count.
    word_counts = Counter()
    for chunk in chunks:
        for m in regex.finditer(PAT, chunk):
            word_counts[m.group().encode("utf-8")] += 1

    # 4. Each pre-token as a list of single-byte tokens; initial pair stats.
    word_syms = {w: [bytes([b]) for b in w] for w in word_counts}
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for w, syms in word_syms.items():
        c = word_counts[w]
        for p in zip(syms, syms[1:]):
            pair_counts[p] += c
            pair_to_words[p].add(w)

    # 5. Merge loop, updating only affected pre-tokens.
    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        new = best[0] + best[1]
        vocab[len(vocab)] = new

        touched = set()
        for w in list(pair_to_words[best]):
            syms = word_syms[w]
            c = word_counts[w]
            for p in zip(syms, syms[1:]):
                pair_counts[p] -= c
                pair_to_words[p].discard(w)
                touched.add(p)

            merged = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == best[0] and syms[i + 1] == best[1]:
                    merged.append(new)
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            word_syms[w] = merged

            for p in zip(merged, merged[1:]):
                pair_counts[p] += c
                pair_to_words[p].add(w)
                touched.add(p)

        for p in touched:
            if pair_counts[p] <= 0:
                del pair_counts[p]
                pair_to_words.pop(p, None)

    return vocab, merges