PYTHON ?= python3
DATA_DIR ?= data
OUT_DIR ?= out/small
CKPT ?= $(OUT_DIR)/best.pt
MODEL ?= model_int8.bin
TOKENIZER ?= $(DATA_DIR)/tokenizer.bin
PRESET ?= small
Q ?= What is PSRAM?
PROMPT ?= User: Hi Assistant:

.PHONY: help all data train sample export verify test test-c clean

help:
	@echo "tinychat-esp32"
	@echo ""
	@echo "  make data      tokenizer + train/val from chat_corpus.txt"
	@echo "  make train     train the $(PRESET) preset into $(OUT_DIR)"
	@echo "  make sample    chat with the checkpoint (Q='your question')"
	@echo "  make export    write $(MODEL) (llama2.c v2 int8)"
	@echo "  make verify    run $(MODEL) in the pinned llama2.c C engine"
	@echo "  make test      Python unit + integration tests"
	@echo "  make test-c    also compile/run the C engine (needs gcc + network)"
	@echo "  make clean     remove out/, .runq/, and Python caches"
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

verify:
	./verify_runq.sh $(MODEL) $(TOKENIZER) "$(PROMPT)"

test:
	$(PYTHON) -m unittest discover -s tests -v

test-c:
	RUN_C_TEST=1 $(PYTHON) -m unittest tests.test_c_engine -v

clean:
	rm -rf out .runq __pycache__ tests/__pycache__ .pytest_cache
	find . -name '*.pyc' -delete
