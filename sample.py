import argparse

import sentencepiece as spm
import torch

from model import Config, Transformer


def load(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = Transformer(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


def sample_token(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        keep = (torch.cumsum(sp, 0) - sp) < top_p       # always keeps the first token
        sp = sp * keep
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def generate(model, sp, prompt, max_new=80, temperature=0.7, top_k=40, top_p=0.95, device="cpu"):
    c = model.c
    ids = [sp.bos_id()] + sp.encode(prompt)
    if len(ids) >= c.max_seq_len:
        raise ValueError(f"prompt is {len(ids)} tokens; context is {c.max_seq_len}")
    cache = model.new_cache(device)
    logits = None
    for pos, t in enumerate(ids):                      # prefill, one token at a time like the MCU
        logits = model.decode_step(t, pos, cache)
    out, pos = [], len(ids)
    while pos < c.max_seq_len and len(out) < max_new:
        nxt = sample_token(logits, temperature, top_k, top_p)
        if nxt == sp.eos_id():
            break
        out.append(nxt)
        logits = model.decode_step(nxt, pos, cache)
        pos += 1
    # (full text: prompt + continuation, continuation only)
    return sp.decode(ids[1:] + out), sp.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default=None, help="raw prompt text")
    ap.add_argument("--chat", default=None, help="user message, wrapped in --template")
    ap.add_argument("--template", default="User: {msg} Assistant:")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    model = load(a.ckpt, a.device)
    run = lambda p: generate(model, sp, p, a.max_new, a.temperature, a.top_k, a.top_p, a.device)

    if a.prompt is not None:
        print(run(a.prompt)[0])
    elif a.chat is not None:
        print(run(a.template.format(msg=a.chat))[0])
    else:
        print("interactive mode, empty line to quit")
        while True:
            msg = input("you> ").strip()
            if not msg:
                break
            p = a.template.format(msg=msg)
            print("bot>", run(p)[1].strip())


if __name__ == "__main__":
    main()
