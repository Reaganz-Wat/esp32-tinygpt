PYTHON ?= python3
DATA_DIR ?= data
OUT_DIR ?= out/small
CKPT ?= $(OUT_DIR)/best.pt
MODEL ?= model_int8.bin
PRESET ?= small
Q ?= What is PSRAM?

.PHONY: help all data train sample export clean

help:
	@echo "tinychat-esp32"
	@echo ""
	@echo "  make data      tokenizer + train/val from chat_corpus.txt"
	@echo "  make train     train the $(PRESET) preset into $(OUT_DIR)"
	@echo "  make sample    chat with the checkpoint (Q='your question')"
	@echo "  make export    write $(MODEL) (llama2.c v2 int8)"
	@echo "  make clean     remove out/ and Python caches"
	@echo ""
	@echo "Override: make PYTHON=.venv/bin/python ..."

all: data train export

data:
	$(PYTHON) prepare_data.py --corpus chat_corpus.txt --vocab-size 1024 --out-dir $(DATA_DIR)

train:
	$(PYTHON) train.py --data-dir $(DATA_DIR) --preset $(PRESET) --out-dir $(OUT_DIR)

sample:
	$(PYTHON) sample.py --ckpt $(CKPT) --tokenizer $(DATA_DIR)/tok.model --chat "$(Q)"

export:
	$(PYTHON) export_int8.py --ckpt $(CKPT) --out $(MODEL) --data-dir $(DATA_DIR)

clean:
	rm -rf out __pycache__ .pytest_cache
	find . -name '*.pyc' -delete
