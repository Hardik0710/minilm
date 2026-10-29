import argparse
import json
import os
import time

from minilm.tokenizer.bpe import train_bpe

SPECIAL = ["<|endoftext|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="only use roughly the first N bytes of the file")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    path = args.input

    if args.max_bytes:
        with open(path, "rb") as f:
            data = f.read(args.max_bytes)
        cut = data.rfind(b"<|endoftext|>")  # cut on a story boundary
        if cut != -1:
            data = data[:cut]
        path = "data/_subset.txt"
        with open(path, "wb") as f:
            f.write(data)
        print(f"using {len(data) / 1e6:.0f} MB subset")

    start = time.time()
    vocab, merges = train_bpe(path, args.vocab_size, SPECIAL)
    elapsed = time.time() - start
    print(f"training time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    longest = max(vocab.values(), key=len)
    print("longest token:", longest, "length:", len(longest))

    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump({str(i): b.hex() for i, b in vocab.items()}, f)
    with open(os.path.join(args.out_dir, "merges.txt"), "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()