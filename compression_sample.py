import random
from minilm.tokenizer.tokenizer import Tokenizer

tok = Tokenizer.from_files("tok_train/vocab.json", "tok_train/merges.txt",
                           ["<|endoftext|>"])
with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
    stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]

random.seed(0)
sample = random.sample(stories, 10)
tb = tt = 0
for s in sample:
    n_b, n_t = len(s.encode("utf-8")), len(tok.encode(s))
    tb, tt = tb + n_b, tt + n_t
    print(f"{n_b} bytes / {n_t} tokens = {n_b / n_t:.2f}")
print(f"overall: {tb / tt:.2f} bytes/token")