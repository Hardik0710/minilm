import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("runs/main/log.csv")))
steps = [int(r["step"]) for r in rows]
elapsed = [float(r["elapsed"]) for r in rows]
train = [float(r["train_loss"]) for r in rows]
vs = [(int(r["step"]), float(r["elapsed"]), float(r["val_loss"]))
      for r in rows if r["val_loss"] != ""]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(steps, train, alpha=0.5, label="train (single batch)")
ax[0].plot([s for s, _, _ in vs], [v for _, _, v in vs], "o-", label="validation")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].set_ylim(1.5, 5)
ax[0].legend()
ax[1].plot(elapsed, train, alpha=0.5, label="train (single batch)")
ax[1].plot([t for _, t, _ in vs], [v for _, _, v in vs], "o-", label="validation")
ax[1].set_xlabel("elapsed seconds"); ax[1].set_ylabel("loss"); ax[1].set_ylim(1.5, 5)
ax[1].legend()
plt.tight_layout()
plt.savefig("figures/main_loss.png", dpi=150)