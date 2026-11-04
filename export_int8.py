import argparse
import copy
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Config, Transformer

MAGIC = 0x616B3432  # "ak42"
HEADER_SIZE = 256   # fixed by llama2.c version 2


def quantize_q80(w, gs):
    w = w.detach().float().reshape(-1, gs)
    scale = w.abs().amax(dim=1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.reshape(-1), scale


def dequantize(q, scale, shape, gs):
    return (q.float().reshape(-1, gs) * scale[:, None]).reshape(shape)


def fake_quant_act(x, gs):
    shp = x.shape
    xg = x.reshape(-1, gs)
    s = xg.abs().amax(dim=1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.round(xg / s) * s).reshape(shp)


class QLinear(nn.Module):
    """Emulates runq.c matmul: int8 weights (dequantized here) and per-group int8 activations."""
    def __init__(self, w_deq, gs):
        super().__init__()
        self.register_buffer("w", w_deq)
        self.gs = gs

    def forward(self, x):
        return F.linear(fake_quant_act(x, self.gs), self.w)


def weight_list(model):
    L = model.layers
    ws = [model.tok_emb.weight]
    for name in ("wq", "wk", "wv", "wo"):
        ws += [getattr(l.attn, name).weight for l in L]
    for name in ("w1", "w2", "w3"):
        ws += [getattr(l.ffn, name).weight for l in L]
    return ws


def write_model(model, path, gs):
    c = model.c
    f32 = lambda t: t.detach().cpu().float().contiguous().numpy().tobytes()
    with open(path, "wb") as f:
        # llama2.c files are little-endian regardless of the host CPU.
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<i", 2))
        f.write(struct.pack("<iiiiiii", c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
                            c.vocab_size, c.max_seq_len))
        f.write(struct.pack("<B", 1))           # shared classifier (tied embeddings)
        f.write(struct.pack("<i", gs))
        padding = HEADER_SIZE - f.tell()
        assert padding >= 0, f"header is {f.tell()} bytes, larger than {HEADER_SIZE}"
        f.write(b"\0" * padding)
        for l in model.layers:
            f.write(f32(l.attn_norm.weight))
        for l in model.layers:
            f.write(f32(l.ffn_norm.weight))
        f.write(f32(model.norm.weight))
        worst = 0.0
        for w in weight_list(model):
            q, s = quantize_q80(w, gs)
            f.write(q.cpu().numpy().tobytes())
            f.write(f32(s))
            worst = max(worst, (dequantize(q, s, w.shape, gs) - w.detach().float()).abs().max().item())
    return worst


def expected_size(c, gs):
    norms = (2 * c.n_layers + 1) * c.dim * 4
    mats = [c.vocab_size * c.dim] + [c.dim * c.n_heads * c.head_dim] * c.n_layers
    mats += [c.dim * c.n_kv_heads * c.head_dim] * (2 * c.n_layers) + [c.n_heads * c.head_dim * c.dim] * c.n_layers
    mats += [c.dim * c.hidden_dim] * (3 * c.n_layers)
    return HEADER_SIZE + norms + sum(n + 4 * (n // gs) for n in mats)


def check_divisible(c, gs):
    for name, n in (("dim", c.dim), ("hidden_dim", c.hidden_dim), ("n_heads*head_dim", c.n_heads * c.head_dim)):
        if n % gs:
            raise SystemExit(f"{name}={n} is not divisible by group size {gs}; pick another --group-size")


@torch.no_grad()
def val_loss(model, data, T, batches=60, B=16, seed=0):
    rng = np.random.RandomState(seed)
    losses = []
    for _ in range(batches):
        ix = rng.randint(0, len(data) - T - 1, size=B)
        x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
        losses.append(model(x, y)[1].item())
    return float(np.mean(losses))


def emulate_runq(model, gs):
    q = copy.deepcopy(model)
    for l in q.layers:
        for name in ("wq", "wk", "wv", "wo"):
            m = getattr(l.attn, name)
            qq, s = quantize_q80(m.weight, gs)
            setattr(l.attn, name, QLinear(dequantize(qq, s, m.weight.shape, gs), gs))
        for name in ("w1", "w2", "w3"):
            m = getattr(l.ffn, name)
            qq, s = quantize_q80(m.weight, gs)
            setattr(l.ffn, name, QLinear(dequantize(qq, s, m.weight.shape, gs), gs))
    qq, s = quantize_q80(model.tok_emb.weight, gs)
    deq = dequantize(qq, s, model.tok_emb.weight.shape, gs)
    q.tok_emb.weight = nn.Parameter(deq.clone())     # embedding lookup uses dequantized rows
    q.output = QLinear(deq, gs)                      # classifier shares the same int8 table
    return q.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="model_int8.bin")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--data-dir", default=None, help="enables the int8 accuracy check on val.bin")
    ap.add_argument("--bandwidth-mbs", type=float, default=43.0,
                    help="effective PSRAM bandwidth for the speed estimate (MB/s)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model = Transformer(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    c = model.c
    check_divisible(c, a.group_size)

    worst = write_model(model, a.out, a.group_size)
    size = os.path.getsize(a.out)
    assert size == expected_size(c, a.group_size), f"size {size} != expected {expected_size(c, a.group_size)}"
    fp32 = model.num_params() * 4
    print(f"wrote {a.out}: {size / 2**20:.2f} MiB  (fp32 would be {fp32 / 2**20:.2f} MiB, "
          f"{100 * (1 - size / fp32):.0f}% smaller) | worst weight error {worst:.5f}")

    # Memory-bound speed estimate: every weight byte is read once per generated token.
    kv_bytes = c.n_layers * 2 * (c.max_seq_len // 2) * c.n_kv_heads * c.head_dim * 4   # avg half-full fp32 cache
    per_tok = size + kv_bytes   # tied table is read in full for the logits, so ~every stored byte is read per token
    print(f"bytes read per token ~{per_tok / 1e6:.2f} MB -> rough ceiling {a.bandwidth_mbs * 1e6 / per_tok:.1f} tok/s "
          f"at {a.bandwidth_mbs:.0f} MB/s effective (measure your board and pass --bandwidth-mbs)")

    if a.data_dir:
        val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
        T = c.max_seq_len
        l32 = val_loss(model, val, T)
        l8 = val_loss(emulate_runq(model, a.group_size), val, T)
        print(f"val loss fp32 {l32:.4f} (ppl {np.exp(l32):.2f}) | int8 W8A8 {l8:.4f} (ppl {np.exp(l8):.2f}) "
              f"| delta {l8 - l32:+.4f}")


if __name__ == "__main__":
    main()
