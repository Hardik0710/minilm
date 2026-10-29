import argparse
import time
import numpy as np

from minilm.tokenizer.tokenizer import Tokenizer

SPECIAL = ["<|endoftext|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_files(f"{args.tok_dir}/vocab.json",
                               f"{args.tok_dir}/merges.txt", SPECIAL)
    start = time.time()
    n_bytes = 0
    chunks = []
    buf = []
    with open(args.input, encoding="utf-8") as f:
        def lines():
            nonlocal n_bytes
            for line in f:
                n_bytes += len(line.encode("utf-8"))
                yield line
        for tid in tok.encode_iterable(lines()):
            buf.append(tid)
            if len(buf) >= 5_000_000:
                chunks.append(np.array(buf, dtype=np.uint16))
                buf = []
    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))
    arr = np.concatenate(chunks)
    np.save(args.output, arr)

    el = time.time() - start
    print(f"tokens: {len(arr):,}  bytes: {n_bytes:,}")
    print(f"bytes/token: {n_bytes / len(arr):.2f}")
    print(f"time: {el:.0f}s  speed: {n_bytes / el:,.0f} bytes/s")


if __name__ == "__main__":
    main()