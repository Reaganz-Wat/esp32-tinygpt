import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from export_int8 import write_model
from tests.helpers import DATA, RUNQ_SHA256, RUNQ_URL, tiny_model

RUN_C_TEST = os.environ.get("RUN_C_TEST") == "1"


@unittest.skipUnless(RUN_C_TEST, "set RUN_C_TEST=1 to compile and run the C engine")
class CEngineTest(unittest.TestCase):
    def test_export_loads_and_runs_in_runq(self):
        if not shutil.which("gcc"):
            self.skipTest("gcc not found")
        tokenizer = DATA / "tokenizer.bin"
        if not tokenizer.exists():
            self.skipTest("data/tokenizer.bin not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "runq.c"
            try:
                urllib.request.urlretrieve(RUNQ_URL, src)
            except (urllib.error.URLError, TimeoutError) as e:
                self.skipTest(f"could not download pinned runq.c: {e}")

            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            self.assertEqual(digest, RUNQ_SHA256, "pinned runq.c checksum mismatch")

            model = tiny_model(vocab_size=1024, max_seq_len=32)
            model_path = tmp / "model.bin"
            write_model(model, model_path, 32)

            runq = tmp / "runq"
            subprocess.run(["gcc", "-O3", "-o", str(runq), str(src), "-lm"],
                           check=True, capture_output=True, text=True)

            proc = subprocess.run(
                [str(runq), str(model_path), "-z", str(tokenizer),
                 "-t", "0", "-n", "16", "-i", "User: Hi Assistant:"],
                cwd=tmp, capture_output=True, text=True, errors="replace")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(proc.stdout.strip(), "runq produced no output")
            self.assertIn("achieved tok/s", proc.stderr)


if __name__ == "__main__":
    unittest.main()
