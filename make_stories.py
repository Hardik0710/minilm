import torch
from minilm.model.transformer import TransformerLM
from minilm.tokenizer.tokenizer import Tokenizer
from minilm.generate import generate

tok = Tokenizer.from_files("tok_train/vocab.json", "tok_train/merges.txt",
                           ["<|endoftext|>"])
model = TransformerLM(10000, 256, 512, 4, 16, 1344, 10000.0)
model.load_state_dict(torch.load("runs/main/ckpt.pt", map_location="cpu")["model"])

for temp, p in [(0.7, 0.9), (1.0, 0.95)]:
    for seed in (1, 2):
        torch.manual_seed(seed)
        text = generate(model, tok, "Once upon a time", 300, temp, p,
                        min_new_tokens=256)
        full = "Once upon a time" + text
        print(f"===== temperature {temp}, top_p {p}, seed {seed}, "
              f"{len(tok.encode(full))} tokens =====")
        print(full, "\n")