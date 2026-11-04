import os
import struct
import tempfile
import unittest

import numpy as np
import torch

from export_int8 import (MAGIC, check_divisible, dequantize, emulate_runq,
                         expected_size, val_loss, weight_list, write_model)
from tests.helpers import DATA, parse_model, tiny_model


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = tiny_model(vocab_size=128, max_seq_len=32)
        self.gs = 32
        self.path = os.path.join(self.tmp.name, "model.bin")
        write_model(self.model, self.path, self.gs)
        self.info = parse_model(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header_and_size(self):
        c = self.model.c
        info = self.info
        self.assertEqual(info["magic"], MAGIC)
        self.assertEqual(info["version"], 2)
        self.assertEqual(info["shared"], 1)
        self.assertEqual(info["gs"], self.gs)
        self.assertEqual(info["config"].dim, c.dim)
        self.assertEqual(info["config"].hidden_dim, c.hidden_dim)
        self.assertEqual(info["config"].n_layers, c.n_layers)
        self.assertEqual(info["config"].n_heads, c.n_heads)
        self.assertEqual(info["config"].n_kv_heads, c.n_kv_heads)
        self.assertEqual(info["config"].vocab_size, c.vocab_size)
        self.assertEqual(info["config"].max_seq_len, c.max_seq_len)
        self.assertEqual(info["size"], expected_size(c, self.gs))

    def test_norms_roundtrip_exactly(self):
        for got, layer in zip(self.info["attn_norms"], self.model.layers):
            np.testing.assert_array_equal(got, layer.attn_norm.weight.detach().numpy())
        for got, layer in zip(self.info["ffn_norms"], self.model.layers):
            np.testing.assert_array_equal(got, layer.ffn_norm.weight.detach().numpy())
        np.testing.assert_array_equal(self.info["final_norm"], self.model.norm.weight.detach().numpy())

    def test_quantized_weights_roundtrip_within_half_scale(self):
        for i, (w, (q, s)) in enumerate(zip(weight_list(self.model), self.info["mats"])):
            w = w.detach().float()
            deq = dequantize(torch.from_numpy(q).reshape(w.shape),
                             torch.from_numpy(s), w.shape, self.gs)
            max_err = (deq - w).abs().max().item()
            self.assertLessEqual(max_err, float(s.max()) / 2 + 1e-5, f"matrix {i}")

    def test_scales_are_finite_and_positive(self):
        for q, s in self.info["mats"]:
            self.assertTrue(np.isfinite(s).all())
            self.assertTrue((s > 0).all())
            self.assertTrue(((q >= -127) & (q <= 127)).all())

    def test_group_size_must_divide_model(self):
        with self.assertRaises(SystemExit):
            check_divisible(self.model.c, 48)

    def test_int8_emulation_runs(self):
        val = np.random.RandomState(0).randint(0, 128, size=128).astype(np.uint16)
        l32 = val_loss(self.model, val, self.model.c.max_seq_len, batches=1, B=2)
        l8 = val_loss(emulate_runq(self.model, self.gs), val, self.model.c.max_seq_len, batches=1, B=2)
        self.assertTrue(np.isfinite(l32))
        self.assertTrue(np.isfinite(l8))


class TokenizerTest(unittest.TestCase):
    def test_tokenizer_bin_format(self):
        path = DATA / "tokenizer.bin"
        self.assertTrue(path.exists(), "data/tokenizer.bin is required")
        with open(path, "rb") as f:
            max_len = struct.unpack("<I", f.read(4))[0]
            tokens, scores = [], []
            while True:
                head = f.read(8)
                if not head:
                    break
                score, length = struct.unpack("<fI", head)
                self.assertLessEqual(length, max_len)
                tokens.append(f.read(length))
                scores.append(score)
        self.assertEqual(len(tokens), 1024)
        self.assertEqual(tokens[0], b"<unk>")
        self.assertEqual(tokens[1], b"\n<s>\n")
        self.assertEqual(tokens[2], b"\n</s>\n")
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
