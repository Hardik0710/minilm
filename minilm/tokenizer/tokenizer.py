import json
import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = regex.compile(PAT)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.bytes_to_id = {b: i for i, b in self.vocab.items()}
        for tok in self.special_tokens:
            tb = tok.encode("utf-8")
            if tb not in self.bytes_to_id:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = tb
                self.bytes_to_id[tb] = new_id
        self.merge_rank = {pair: r for r, pair in enumerate(self.merges)}
        if self.special_tokens:
            # longest first so overlapping special tokens match the longer one
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = regex.compile(
                "(" + "|".join(regex.escape(t) for t in ordered) + ")"
            )
        else:
            self._special_re = None
        self._cache = {}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path) as f:
            vocab = {int(i): bytes.fromhex(h) for i, h in json.load(f).items()}
        merges = []
        with open(merges_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ")
                    merges.append((bytes.fromhex(a), bytes.fromhex(b)))
        return cls(vocab, merges, special_tokens)

    def _encode_word(self, word: bytes):
        if word in self._cache:
            return self._cache[word]
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = self.merge_rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        ids = [self.bytes_to_id[p] for p in parts]
        self._cache[word] = ids
        return ids

    def _encode_chunk(self, text):
        ids = []
        for m in _PAT.finditer(text):
            ids.extend(self._encode_word(m.group().encode("utf-8")))
        return ids

    def encode(self, text):
        if self._special_re is None:
            return self._encode_chunk(text)
        ids = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                ids.append(self.bytes_to_id[piece.encode("utf-8")])
            else:
                ids.extend(self._encode_chunk(piece))
        return ids

    def encode_iterable(self, iterable):
        # iterable yields strings (e.g. an open file yielding lines).
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")