import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from model import Config, Transformer, build_config


def get_batch(data, B, T, device):
    ix = np.random.randint(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(it, a):
    if it < a.warmup:
        return a.lr * (it + 1) / a.warmup
    if it >= a.max_iters:
        return a.min_lr
    r = (it - a.warmup) / max(1, a.max_iters - a.warmup)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (a.lr - a.min_lr)


@torch.no_grad()
def estimate(model, data, a, device, ctx):
    model.eval()
    losses = []
    for _ in range(a.eval_iters):
        x, y = get_batch(data, a.batch_size, model.c.max_seq_len, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save(path, model, opt, it, best_val, bad_evals):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": model.c.to_dict(),
                "iter": it, "best_val": best_val, "bad_evals": bad_evals}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out/run")
    ap.add_argument("--preset", default="small", choices=["nano", "small", "base"])
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--patience", type=int, default=0, help="stop after N evals without val improvement (0=off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    train = np.memmap(os.path.join(a.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    assert len(train) > a.max_seq_len + 2, "training split too small for this sequence length"
    assert len(val) > a.max_seq_len + 2, "validation split too small for this sequence length"

    device = a.device
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)

    ckpt_path, best_path = os.path.join(a.out_dir, "ckpt.pt"), os.path.join(a.out_dir, "best.pt")
    it, best_val, bad_evals = 0, float("inf"), 0
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = Transformer(Config(**ck["config"])).to(device)
        model.load_state_dict(ck["model"])
        opt = model.configure_optimizer(a.lr, a.weight_decay)
        opt.load_state_dict(ck["optimizer"])
        it, best_val, bad_evals = ck["iter"], ck["best_val"], ck.get("bad_evals", 0)
        print(f"resumed from step {it}, best val {best_val:.4f}")
    else:
        cfg = build_config(a.preset, meta["vocab_size"], a.max_seq_len, a.dropout)
        model = Transformer(cfg).to(device)
        opt = model.configure_optimizer(a.lr, a.weight_decay)

    c = model.c
    print(f"params {model.num_params():,} | dim {c.dim} layers {c.n_layers} heads {c.n_heads}/{c.n_kv_heads}kv "
          f"hidden {c.hidden_dim} vocab {c.vocab_size} ctx {c.max_seq_len} | device {device} bf16={use_bf16}")
    tok_per_iter = a.batch_size * a.grad_accum * c.max_seq_len
    print(f"tokens/iter {tok_per_iter:,} | train tokens {len(train):,} -> "
          f"{a.max_iters * tok_per_iter / len(train):.1f} epochs at max_iters={a.max_iters}")

    log_path = os.path.join(a.out_dir, "log.csv")
    new_log = not (a.resume and os.path.exists(log_path))
    logf = open(log_path, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["iter", "train_loss", "val_loss", "lr", "tokens_seen", "seconds"])

    t0 = time.time()
    model.train()
    run_loss, run_n = 0.0, 0
    try:
        while it <= a.max_iters:
            lr = lr_at(it, a)
            for g in opt.param_groups:
                g["lr"] = lr

            if it % a.eval_interval == 0:
                tl = estimate(model, train, a, device, ctx)
                vl = estimate(model, val, a, device, ctx)
                gap = vl - tl
                print(f"step {it:6d} | train {tl:.4f} | val {vl:.4f} | gap {gap:+.3f} | lr {lr:.2e} | "
                      f"{time.time() - t0:.0f}s")
                log.writerow([it, f"{tl:.5f}", f"{vl:.5f}", f"{lr:.3e}", it * tok_per_iter, f"{time.time() - t0:.1f}"])
                logf.flush()
                if vl < best_val:
                    best_val, bad_evals = vl, 0
                    save(best_path, model, opt, it, best_val, bad_evals)
                else:
                    bad_evals += 1
                    if a.patience and bad_evals >= a.patience:
                        print(f"early stop: no val improvement for {bad_evals} evals (best {best_val:.4f})")
                        break
                save(ckpt_path, model, opt, it, best_val, bad_evals)
            if it == a.max_iters:
                break

            for _ in range(a.grad_accum):
                x, y = get_batch(train, a.batch_size, c.max_seq_len, device)
                with ctx:
                    _, loss = model(x, y)
                (loss / a.grad_accum).backward()
                run_loss += loss.item()
                run_n += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            it += 1
    except KeyboardInterrupt:
        print("\nCtrl+C - saving checkpoint before exit...")
    finally:
        save(ckpt_path, model, opt, it, best_val, bad_evals)
        logf.close()
        print(f"saved {ckpt_path} (step {it}); best val {best_val:.4f} in {best_path}")


if __name__ == "__main__":
    main()
