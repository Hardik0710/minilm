import sys
import csv
import matplotlib.pyplot as plt

# usage: python plot_runs.py out.png runs/a/log.csv runs/b/log.csv ...
out, paths = sys.argv[1], sys.argv[2:]
for p in paths:
    with open(p) as f:
        rows = list(csv.DictReader(f))
    plt.plot([int(r["step"]) for r in rows],
             [float(r["train_loss"]) for r in rows], label=p)
plt.xlabel("step"); plt.ylabel("train loss"); plt.legend()
plt.savefig(out, dpi=150)