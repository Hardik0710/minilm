import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from minilm.model.transformer import TransformerLM
from minilm.train.checkpoint import load_checkpoint, save_checkpoint
from minilm.train.data import get_batch
from minilm.train.loss import lm_loss
from minilm.train.optim import make_optimizer, make_scheduler
from minilm.train.step import train_step


def parse_args():
    ap = argparse.ArgumentParser()
    # data / output
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--out-dir", required=True)
    # model
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--d-ff", type=int, default=1344)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    # optimisation
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--min-lr", type=float, default=None, help="default lr/10")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # logging / checkpoints
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    # debugging
    ap.add_argument("--single-batch", action="store_true",
                    help="train on one fixed batch repeatedly")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="exit after this step (simulates an interruption)")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, batches, use_amp, device):
    model.eval()
    total = 0.0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        total += lm_loss(logits.float(), y).item()
    model.train()
    return total / len(batches)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.fp16 and device.type == "cuda"
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train = np.load(args.train_data, mmap_mode="r")
    val = np.load(args.val_data, mmap_mode="r")

    # Fixed validation batches: same seed every run, built once, reused every eval.
    np.random.seed(1234)
    val_batches = [get_batch(val, args.batch_size, args.context_length, "cpu")
                   for _ in range(args.eval_batches)]

    torch.manual_seed(args.seed)
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model,
                          args.num_layers, args.num_heads, args.d_ff,
                          args.rope_theta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}  device: {device}  amp: {use_amp}")

    optimizer = make_optimizer(model, args.lr, (args.beta1, args.beta2),
                               args.weight_decay)
    warm = max(1, int(args.warmup_frac * args.steps))
    # Cosine ends at the LAST step index, so lr == min_lr exactly at the last step.
    scheduler = make_scheduler(optimizer, args.lr, min_lr, warm, args.steps - 1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fwd_model = torch.compile(model) if args.compile else model

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    log_path = os.path.join(args.out_dir, "log.csv")
    start_step, time_offset = 0, 0.0
    if args.resume and os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler,
                                     scaler, map_location=device)
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            time_offset = float(rows[-1]["elapsed"])
        print(f"resumed from step {start_step}")
    new_log = not (args.resume and os.path.exists(log_path))
    log_f = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed"])

    # Different data order on resume, but deterministic for a given (seed, step).
    np.random.seed(args.seed + start_step)
    fixed = get_batch(train, args.batch_size, args.context_length, device) \
        if args.single_batch else None

    t0 = time.time()
    model.train()
    for step in range(start_step, args.steps):
        x, y = fixed if fixed is not None else get_batch(
            train, args.batch_size, args.context_length, device)
        lr_now = optimizer.param_groups[0]["lr"]
        loss, gnorm = train_step(fwd_model, optimizer, scheduler, scaler, x, y,
                                 use_amp=use_amp, clip=args.clip)
        done = step + 1

        val_loss = ""
        if done % args.eval_every == 0 or done == args.steps:
            val_loss = evaluate(fwd_model, val_batches, use_amp, device)
        if done % args.log_every == 0 or val_loss != "" or done == 1:
            elapsed = time_offset + time.time() - t0
            writer.writerow([done, loss, val_loss, lr_now, gnorm, f"{elapsed:.1f}"])
            log_f.flush()
            msg = f"step {done} loss {loss:.4f} lr {lr_now:.2e} gnorm {gnorm:.3f}"
            if val_loss != "":
                msg += f" val {val_loss:.4f}"
            print(msg)

        if not np.isfinite(loss):
            print("non-finite loss, stopping")
            break
        if done % args.ckpt_every == 0 or done == args.steps:
            tmp = ckpt_path + ".tmp"
            save_checkpoint(tmp, model, optimizer, scheduler, scaler, done)
            os.replace(tmp, ckpt_path)
        if args.stop_after is not None and done >= args.stop_after:
            print(f"stopping after step {done} (--stop-after)")
            break

    log_f.close()


if __name__ == "__main__":
    main()