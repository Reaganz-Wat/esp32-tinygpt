# tinychat-esp32

Train a tiny Llama-style transformer in PyTorch, export it to an int8 file, and
run it on an ESP32-S3.

The model is deliberately small: RMSNorm, RoPE, SwiGLU, grouped-query
attention, tied embeddings, and no biases. Its tensor layout matches
[llama2.c](https://github.com/karpathy/llama2.c), so the PyTorch model and the
on-device C engine run the same forward pass.

## What is in the box

- `model.py` — the transformer, including a KV-cache `decode_step` that mirrors the C engine.
- `prepare_data.py` — SentencePiece BPE tokenizer, `train.bin`, `val.bin`, and `tokenizer.bin`.
- `train.py` — validation, best checkpoint, resume, cosine LR, early stopping, CSV log.
- `sample.py` — temperature / top-k / top-p generation that stops at EOS.
- `export_int8.py` — llama2.c v2 int8 export with size, accuracy, and speed checks.
- `esp32/` — ESP32-S3 firmware and a portable C inference engine.
- `make_toy_corpus.py` — generate a tiny synthetic corpus to test the pipeline.

## Requirements

- Python 3.10+
- An ESP32-S3 with PSRAM for the on-device demo (optional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Makefile uses `python3` by default. To use the virtualenv interpreter:

```bash
make PYTHON=.venv/bin/python train
```

## Quick start

The repository already includes a demo corpus and tokenizer, so you can start
training immediately:

```bash
make train                          # small preset -> out/small
make sample Q="What is PSRAM?"      # chat with out/small/best.pt
make export                         # writes model_int8.bin
```

Run the whole pipeline from scratch:

```bash
make data                           # chat_corpus.txt -> data/
make train                          # train the small preset
make sample Q="What is PSRAM?"
make export
```

Run `make help` to see all targets.

## How it works

1. **Prepare data.** Put one training example per line:

   ```text
   User: What is PSRAM? Assistant: PSRAM is an external RAM chip ...
   ```

   `prepare_data.py` trains the tokenizer on the training split only, writes
   `train.bin` / `val.bin`, and exports `tokenizer.bin` in llama2.c format.
   Special tokens are `unk=0`, `bos=1`, `eos=2`.

2. **Train.** `train.py` random-crops batches from a memory-mapped file,
   evaluates on a held-out split, and keeps the best checkpoint. Ctrl+C saves a
   resumable checkpoint; `--resume` continues it.

3. **Sample.** The prompt must match the training template. The default chat
   template is `User: {msg} Assistant:`. Use `--temperature 0` for greedy
   decoding.

4. **Export.** `export_int8.py` writes the llama2.c version-2 int8 format:
   magic `ak42`, group size 32, fp32 norms, tied classifier, no biases. It
   checks the file size and reports int8-vs-fp32 validation loss.

## Presets

All presets use vocab 1024, context 128, and int8 group size 32.

| preset | params | int8 file | KV cache | rough ceiling* |
|--------|--------|-----------|----------|----------------|
| nano   | 1.24M  | 1.33 MiB  | 384 KiB  | ~27 tok/s |
| small  | 2.63M  | 2.83 MiB  | 576 KiB  | ~13 tok/s |
| base   | 6.16M  | 6.63 MiB  | 1024 KiB | ~6 tok/s  |

\*Memory-bandwidth estimate: every weight byte is read once per token at
~43 MB/s effective PSRAM bandwidth. Measure your board and pass
`--bandwidth-mbs`.

A small model needs a narrow dataset. A useful rule of thumb is ~20 training
tokens per parameter. If your corpus is under a million tokens, keep the model
small, keep the domain narrow, and watch the train/validation gap for
memorization.

## Results

The bundled `small` preset on the demo corpus:

| metric | value |
|---|---|
| parameters | 2.63M |
| int8 file | 2.83 MiB |
| KV cache (fp32, full 128-token context) | 576 KiB |
| best validation loss | 0.3249 (ppl 1.38) |

The training curve is in `results/log.csv`.

## ESP32 deployment

The firmware targets an ESP32-S3 **N16R8** (16 MB flash, 8 MB OPI PSRAM). It
loads `model_int8.bin` and `tokenizer.bin` from LittleFS and runs the int8
forward pass from PSRAM.

```bash
make export
cp model_int8.bin data/tokenizer.bin esp32/data/

cd esp32
pio run                            # build
pio run -t upload                  # flash firmware
pio run -t uploadfs                # flash LittleFS image (model + tokenizer)
pio device monitor -b 115200       # chat on the serial console
```

Type a message at the `User:` prompt; the reply streams back over serial.
Expect roughly 2.83 MiB for the weights, 0.75 MiB for the dequantized embedding,
0.56 MiB for the KV cache, plus activations: about 4.4 MiB of PSRAM.

`platformio.ini` sets 16 MB flash and `qio_opi` (QIO flash, OPI PSRAM). If your
board uses QSPI PSRAM, change `board_build.arduino.memory_type` to `qio_qspi`.

### Portable inference engine

`esp32/src/esp32-llm.c` and `esp32/src/esp32-llm.h` are a self-contained C
inference engine adapted from Andrej Karpathy's `runq.c` (MIT). They have no
Arduino or ESP-IDF dependencies.

The integration contract is small:

- load `model_int8.bin` and `tokenizer.bin` into memory and keep both buffers
  alive until `ellm_free()`;
- the model buffer must be 4-byte aligned;
- pass an allocator that returns PSRAM for large models, or `NULL` to use
  `malloc`/`free`;
- build your own prompt string and drive `ellm_begin()` / `ellm_next()`.

`main.cpp` is the example app: LittleFS loading, PSRAM allocator, the
`User: {msg} Assistant:` template, a serial chat loop, and optional SSD1306
OLED output.

### OLED output

`main.cpp` drives a 128x64 SSD1306 over I2C and mirrors the prompt and streaming
reply there. Default wiring is SDA=GPIO8, SCL=GPIO9, address `0x3C`; `0x3D` is
probed automatically. Change `OLED_SDA`, `OLED_SCL`, and `OLED_ADDR` near the top
of `main.cpp` if your module is wired differently. If no OLED is found, the
firmware falls back to serial only.

## Using your own data

`chat_corpus.txt` is a small synthetic chat corpus used for the demo. Replace it
with your own one-example-per-line corpus and rerun `make data`. Make sure you
have the right to redistribute any corpus you publish.

## Repository layout

```text
.
├── chat_corpus.txt
├── data/                  # tokenizer + metadata (train/val .bin are generated)
├── esp32/                 # ESP32-S3 PlatformIO firmware + LittleFS data
├── results/log.csv        # training curve
├── model.py
├── prepare_data.py
├── train.py
├── sample.py
├── export_int8.py
├── make_toy_corpus.py
├── Makefile
└── requirements.txt
```

## Acknowledgements

- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy for the C
  engine and int8 file format.
- [SentencePiece](https://github.com/google/sentencepiece) for tokenization.
- [PyTorch](https://pytorch.org) for training.

## Limitations

This is a demo, not a general assistant. It is small, trained on a narrow
corpus, and can memorize. On an ESP32 expect a few tokens per second; the scalar
int8 matmul is not optimized for the S3.
