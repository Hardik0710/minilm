import torch

torch.manual_seed(0)
d_k = 32
q = torch.randn(1000, d_k)
k = torch.randn(1000, d_k)
raw = (q * k).sum(-1)            # dot products, variance ~ d_k
scaled = raw / d_k ** 0.5

print("std raw   :", raw.std().item())      # about sqrt(32) = 5.66
print("std scaled:", scaled.std().item())   # about 1

scores = torch.randn(8) * d_k ** 0.5        # one row of unscaled scores
print("softmax without scaling:", torch.softmax(scores, -1))
print("softmax with scaling   :", torch.softmax(scores / d_k ** 0.5, -1))