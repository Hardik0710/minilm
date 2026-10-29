import torch

for lr in [1, 10, 100, 1000]:
    torch.manual_seed(0)
    w = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = torch.optim.SGD([w], lr=lr)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = (w ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"lr={lr}:", [f"{l:.3g}" for l in losses])