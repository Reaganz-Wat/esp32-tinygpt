import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    #            dim  layers heads kv_heads
    "nano":  dict(dim=128, n_layers=6, n_heads=4, n_kv_heads=2),
    "small": dict(dim=192, n_layers=6, n_heads=6, n_kv_heads=3),
    "base":  dict(dim=256, n_layers=8, n_heads=8, n_kv_heads=4),
}


@dataclass
class Config:
    vocab_size: int = 1024
    dim: int = 192
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 3
    max_seq_len: int = 128
    multiple_of: int = 32
    norm_eps: float = 1e-5
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert (self.dim // self.n_heads) % 2 == 0, "head_dim must be even (RoPE rotates pairs)"

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @property
    def hidden_dim(self):
        h = int(2 * (4 * self.dim) / 3)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)

    def to_dict(self):
        return dict(self.__dict__)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim, max_seq, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_seq).float(), freqs)  # (T, head_dim/2)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: (B, T, H, head_dim). Rotates ADJACENT pairs (x[2i], x[2i+1]) exactly like run.c."""
    xr = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.n_rep = c.n_heads // c.n_kv_heads
        self.wq = nn.Linear(c.dim, c.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(c.n_heads * self.hd, c.dim, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin):  # full sequence (training / evaluation)
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd)
        k = self.wk(x).view(B, T, self.n_kv, self.hd)
        v = self.wv(x).view(B, T, self.n_kv, self.hd)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        k, v = k.repeat_interleave(self.n_rep, dim=2), v.repeat_interleave(self.n_rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

    def step(self, x, pos, cos, sin, kc, vc):  # ONE token with a KV cache (what the MCU does)
        q = self.wq(x).view(1, 1, self.n_heads, self.hd)
        k = self.wk(x).view(1, 1, self.n_kv, self.hd)
        v = self.wv(x).view(self.n_kv, self.hd)
        q = apply_rope(q, cos[pos:pos + 1], sin[pos:pos + 1])[0, 0]
        k = apply_rope(k, cos[pos:pos + 1], sin[pos:pos + 1])[0, 0]
        kc[pos], vc[pos] = k, v
        K = kc[:pos + 1].repeat_interleave(self.n_rep, dim=1)  # (P, n_heads, hd)
        V = vc[:pos + 1].repeat_interleave(self.n_rep, dim=1)
        att = torch.einsum("hd,phd->hp", q, K) / math.sqrt(self.hd)
        y = torch.einsum("hp,phd->hd", att.softmax(-1), V).reshape(-1)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.w1 = nn.Linear(c.dim, c.hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(c.hidden_dim, c.dim, bias=False)  # down
        self.w3 = nn.Linear(c.dim, c.hidden_dim, bias=False)  # up
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        self.ffn = FeedForward(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.dim)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim, c.norm_eps)
        self.output = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.output.weight = self.tok_emb.weight  # tied: stored once
        cos, sin = rope_tables(c.head_dim, c.max_seq_len, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init for residual-output projections
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weight counted once

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.c.max_seq_len, f"sequence {T} > max_seq_len {self.c.max_seq_len}"
        x = self.drop(self.tok_emb(idx))
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    # ---- incremental decoding: identical algorithm to the C engine -------------
    def new_cache(self, device="cpu"):
        c = self.c
        return [(torch.zeros(c.max_seq_len, c.n_kv_heads, c.head_dim, device=device),
                 torch.zeros(c.max_seq_len, c.n_kv_heads, c.head_dim, device=device))
                for _ in self.layers]

    @torch.no_grad()
    def decode_step(self, token, pos, cache):
        x = self.tok_emb.weight[token]
        for layer, (kc, vc) in zip(self.layers, cache):
            x = x + layer.attn.step(layer.attn_norm(x), pos, self.rope_cos, self.rope_sin, kc, vc)
            x = x + layer.ffn(layer.ffn_norm(x))
        return self.output(self.norm(x))  # (vocab,)

    def configure_optimizer(self, lr, weight_decay, betas=(0.9, 0.95)):
        decay = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def build_config(preset, vocab_size, max_seq_len=128, dropout=0.0, **overrides):
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    return Config(vocab_size=vocab_size, max_seq_len=max_seq_len, dropout=dropout, **kw)
