import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from make_toy_corpus import FACTS, PREFIX
from tests.helpers import REPO, parse_model


class PipelineTest(unittest.TestCase):
    def run_cli(self, *args):
        cmd = [sys.executable, *map(str, args)]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            self.fail(f"command failed: {' '.join(cmd)}\n"
                      f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        return proc

    def test_toy_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "toy.txt"
            rng = random.Random(0)
            with open(corpus, "w", encoding="utf-8") as f:
                for _ in range(300):
                    q, a = rng.choice(FACTS)
                    f.write(f"User: {rng.choice(PREFIX)}{q} Assistant: {a}\n")

            data = tmp / "data"
            out = tmp / "out"
            model = tmp / "model.bin"

            self.run_cli("prepare_data.py", "--corpus", corpus,
                         "--vocab-size", 320, "--out-dir", data)
            self.run_cli("train.py", "--data-dir", data, "--preset", "nano",
                         "--out-dir", out, "--max-iters", 2, "--eval-interval", 10,
                         "--eval-iters", 1, "--batch-size", 2, "--max-seq-len", 32,
                         "--device", "cpu")
            self.assertTrue((out / "best.pt").exists())

            self.run_cli("sample.py", "--ckpt", out / "best.pt",
                         "--tokenizer", data / "tok.model",
                         "--prompt", "User: Hi Assistant:",
                         "--temperature", 0, "--max-new", 4)

            self.run_cli("export_int8.py", "--ckpt", out / "best.pt",
                         "--out", model, "--data-dir", data)
            info = parse_model(model)
            self.assertEqual(info["magic"], 0x616B3432)
            self.assertEqual(info["version"], 2)
            self.assertEqual(info["config"].max_seq_len, 32)
            self.assertEqual(info["size"], model.stat().st_size)


if __name__ == "__main__":
    unittest.main()
