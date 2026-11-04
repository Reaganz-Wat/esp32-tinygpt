"""Shared helpers for the test suite (no third-party test framework needed)."""
import os
import struct
from pathlib import Path

import numpy as np
import torch

from export_int8 import HEADER_SIZE
from model import Config, Transformer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

RUNQ_COMMIT = "350e04fe35433e6d2941dce5a1f53308f87058eb"
RUNQ_SHA256 = "ba9cc681afca79f994f815dfeffc4a61c5b3ca8c5d4d4dcb75c8d375856ba548"
RUNQ_URL = f"https://raw.githubusercontent.com/karpathy/llama2.c/{RUNQ_COMMIT}/runq.c"


def tiny_config(vocab_size=128, max_seq_len=32):
    return Config(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=4,
                  n_kv_heads=2, max_seq_len=max_seq_len, dropout=0.0)


def tiny_model(vocab_size=128, max_seq_len=32, seed=0):
    torch.manual_seed(seed)
    return Transformer(tiny_config(vocab_size, max_seq_len)).eval()


def tensor_sizes(c, gs):
    """Sizes of the quantized matrices, in the exact order write_model() emits them."""
    hd = c.head_dim
    sizes = [c.vocab_size * c.dim]
    for size in (c.dim * c.n_heads * hd, c.dim * c.n_kv_heads * hd,
                 c.dim * c.n_kv_heads * hd, c.n_heads * hd * c.dim):
        sizes += [size] * c.n_layers
    for size in (c.dim * c.hidden_dim, c.hidden_dim * c.dim, c.dim * c.hidden_dim):
        sizes += [size] * c.n_layers
    return sizes


def parse_model(path):
    """Read a llama2.c v2 int8 file back into arrays (little-endian)."""
    path = str(path)
    with open(path, "rb") as f:
        magic, version = struct.unpack("<Ii", f.read(8))
        dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack("<iiiiiii", f.read(28))
        shared, gs = struct.unpack("<Bi", f.read(5))
        f.seek(HEADER_SIZE)

        c = Config(vocab_size=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
                   n_kv_heads=n_kv, max_seq_len=seq, dropout=0.0)
        attn_norms = [np.frombuffer(f.read(dim * 4), dtype="<f4").copy() for _ in range(n_layers)]
        ffn_norms = [np.frombuffer(f.read(dim * 4), dtype="<f4").copy() for _ in range(n_layers)]
        final_norm = np.frombuffer(f.read(dim * 4), dtype="<f4").copy()

        mats = []
        for n in tensor_sizes(c, gs):
            q = np.frombuffer(f.read(n), dtype=np.int8).copy()
            s = np.frombuffer(f.read((n // gs) * 4), dtype="<f4").copy()
            mats.append((q, s))
        assert f.tell() == os.path.getsize(path), "unexpected trailing bytes"

    return dict(magic=magic, version=version, shared=shared, gs=gs, config=c,
                attn_norms=attn_norms, ffn_norms=ffn_norms, final_norm=final_norm,
                mats=mats, size=os.path.getsize(path))
