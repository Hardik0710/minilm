import json, os, shutil, torch

os.makedirs("release", exist_ok=True)
ck = torch.load("runs/main/ckpt.pt", map_location="cpu")
torch.save(ck["model"], "release/model.pt")          # weights only, no optimizer
shutil.copy("tok_train/vocab.json", "release/vocab.json")
shutil.copy("tok_train/merges.txt", "release/merges.txt")
cfg = dict(vocab_size=10000, context_length=256, d_model=512,
           num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)
json.dump(cfg, open("release/config.json", "w"), indent=2)
print({f: os.path.getsize(f"release/{f}") for f in os.listdir("release")})