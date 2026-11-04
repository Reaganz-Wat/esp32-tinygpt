import unittest

import torch

from model import PRESETS, build_config
from tests.helpers import tiny_model


class ModelTest(unittest.TestCase):
    def test_forward_shapes_and_loss(self):
        model = tiny_model(vocab_size=128, max_seq_len=16)
        idx = torch.randint(0, 128, (2, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(tuple(logits.shape), (2, 8, 128))
        self.assertTrue(torch.isfinite(loss).item())

    def test_decode_step_matches_full_forward(self):
        model = tiny_model(vocab_size=128, max_seq_len=16)
        idx = torch.randint(0, 128, (1, 8))

        with torch.no_grad():
            full, _ = model(idx)
            cache = model.new_cache()
            for pos in range(idx.shape[1]):
                step = model.decode_step(int(idx[0, pos]), pos, cache)
                torch.testing.assert_close(step, full[0, pos], atol=1e-4, rtol=1e-4)

    def test_tied_classifier(self):
        model = tiny_model()
        self.assertIs(model.output.weight, model.tok_emb.weight)

    def test_presets_are_valid(self):
        for name in PRESETS:
            c = build_config(name, vocab_size=256, max_seq_len=32)
            self.assertEqual(c.dim % c.n_heads, 0, name)
            self.assertEqual(c.n_heads % c.n_kv_heads, 0, name)
            self.assertEqual(c.head_dim % 2, 0, name)
            self.assertEqual(c.hidden_dim % c.multiple_of, 0, name)

    def test_context_limit_is_enforced(self):
        model = tiny_model(vocab_size=128, max_seq_len=8)
        idx = torch.zeros((1, 9), dtype=torch.long)
        with self.assertRaises(AssertionError):
            model(idx)


if __name__ == "__main__":
    unittest.main()
