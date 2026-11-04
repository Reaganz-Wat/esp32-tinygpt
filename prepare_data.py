import argparse
import json
import os
import random
import struct

import numpy as np
import sentencepiece as spm


def export_tokenizer_bin(sp, path):
    """llama2.c tokenizer.bin: int32 max_token_length, then per token: float score, int32 len, bytes."""
    tokens, scores = [], []
    for i in range(sp.get_piece_size()):
        t = sp.id_to_piece(i)
        if i == sp.bos_id():
            t = "\n<s>\n"
        elif i == sp.eos_id():
            t = "\n</s>\n"
        tokens.append(t.replace("▁", " ").encode("utf-8"))
        scores.append(sp.get_score(i))
    with open(path, "wb") as f:
        f.write(struct.pack("I", max(len(t) for t in tokens)))
        for b, s in zip(tokens, scores):
            f.write(struct.pack("fI", s, len(b)))
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-byte-fallback", action="store_true",
                    help="saves 256 vocab slots but unknown characters become <unk>")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    lines = [l.strip() for l in open(a.corpus, encoding="utf-8") if l.strip()]
    random.Random(a.seed).shuffle(lines)
    n_val = max(1, int(len(lines) * a.val_frac))
    val_lines, train_lines = lines[:n_val], lines[n_val:]
    if not train_lines:
        raise SystemExit("corpus needs at least 2 non-empty lines (one is used for validation)")
    print(f"{len(lines):,} examples -> {len(train_lines):,} train / {len(val_lines):,} val")

    # Tokenizer is trained on TRAIN lines only, so validation stays honest.
    train_txt = os.path.join(a.out_dir, "tok_train.txt")
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))
    prefix = os.path.join(a.out_dir, "tok")
    spm.SentencePieceTrainer.train(
        input=train_txt, model_prefix=prefix, model_type="bpe", vocab_size=a.vocab_size,
        character_coverage=1.0, split_digits=True,          # digits become single tokens
        byte_fallback=not a.no_byte_fallback,
        normalization_rule_name="identity",                 # keep newlines / exact text
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,            # llama2.c expects exactly this
        self_test_sample_size=0, hard_vocab_limit=False, minloglevel=2)
    os.remove(train_txt)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    V = sp.get_piece_size()
    if V > 65535:
        raise SystemExit(f"vocab size {V} does not fit in the uint16 train/val files")

    def encode(ls):
        out = []
        for l in ls:
            out.append(sp.bos_id())
            out.extend(sp.encode(l))
            out.append(sp.eos_id())
        return np.array(out, dtype=np.uint16)

    tr, va = encode(train_lines), encode(val_lines)
    tr.tofile(os.path.join(a.out_dir, "train.bin"))
    va.tofile(os.path.join(a.out_dir, "val.bin"))
    export_tokenizer_bin(sp, os.path.join(a.out_dir, "tokenizer.bin"))

    chars = sum(len(l) for l in train_lines)
    meta = dict(vocab_size=V, train_tokens=int(len(tr)), val_tokens=int(len(va)),
                chars_per_token=round(chars / max(1, len(tr)), 2), corpus_lines=len(lines))
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)

    print(f"vocab size        : {V}")
    print(f"train / val tokens: {len(tr):,} / {len(va):,}   ({meta['chars_per_token']} chars per token)")
    # Sanity: decode round trip + look at what the tokenizer learned
    sample = train_lines[0]
    ids = sp.encode(sample)
    assert sp.decode(ids) == sample, "tokenizer round trip failed"
    print("example           :", ids[:12], "->", [sp.id_to_piece(i) for i in ids[:12]])
    assert int(tr.max()) < V and int(va.max()) < V
    # A model only learns tokens it actually sees. Dead tokens waste embedding rows and ESP32 flash.
    counts = np.bincount(tr, minlength=V)
    is_byte = np.array([sp.id_to_piece(i).startswith("<0x") for i in range(V)])
    dead = int(((counts == 0) & ~is_byte).sum())
    rare = int(((counts < 5) & ~is_byte).sum())
    print(f"byte-fallback slots: {int(is_byte.sum())} (unused if your text is plain ASCII; --no-byte-fallback frees them)")
    print(f"normal tokens never seen in train: {dead} | seen < 5 times: {rare}  (many = vocab too big for this corpus)")
    print(f"train tokens: {len(tr):,}. Rule of thumb: ~20 tokens per parameter for a well-trained model "
          f"(1M params -> ~20M tokens). With less data keep the model small and watch val loss.")

if __name__ == "__main__":
    main()
