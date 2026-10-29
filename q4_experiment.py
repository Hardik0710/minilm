import torch
from minilm.model.rope import RoPE

torch.manual_seed(0)
D, MAXLEN = 32, 128
rope = RoPE(10000.0, D, MAXLEN)

q = torch.randn(D)
k = torch.randn(D)
# Put the SAME q (and the same k) at every position, then rotate.
Q = rope(q.expand(MAXLEN, D).contiguous())   # Q[i] = q rotated to position i
K = rope(k.expand(MAXLEN, D).contiguous())   # K[j] = k rotated to position j

print("same offset i-j = 3:")
for i, j in [(3, 0), (10, 7), (50, 47), (100, 97)]:
    print(f"  i={i:3d} j={j:3d}  dot = {Q[i] @ K[j]:.6f}")

print("different offsets:")
for i, j in [(5, 5), (5, 4), (5, 0), (20, 0)]:
    print(f"  i={i:3d} j={j:3d} (i-j={i-j:2d})  dot = {Q[i] @ K[j]:.6f}")