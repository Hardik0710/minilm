import argparse, json, os, time
import torch

from minilm.model.transformer import TransformerLM
from minilm.tokenizer.tokenizer import Tokenizer
from minilm.generate import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    d = a.model_dir
    cfg = json.load(open(os.path.join(d, "config.json")))
    tok = Tokenizer.from_files(os.path.join(d, "vocab.json"),
                               os.path.join(d, "merges.txt"), ["<|endoftext|>"])
    model = TransformerLM(**cfg)
    model.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                     map_location=a.device))
    model.to(a.device).eval()

    t0 = time.time()
    text = generate(model, tok, a.prompt, a.max_new_tokens,
                    a.temperature, a.top_p, device=a.device)
    dt = time.time() - t0
    n = len(tok.encode(text))
    print(a.prompt + text)
    print(f"\n{n} tokens in {dt:.1f}s ({n / dt:.1f} tokens/s)")


if __name__ == "__main__":
    main()