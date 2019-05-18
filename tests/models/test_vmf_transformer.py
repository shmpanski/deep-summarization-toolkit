import unittest

import torch

from dst.models import VMFTransformer


class TestVMFTransformer(unittest.TestCase):
    def setUp(self):
        self.model_params = {
            "max_seq_len": 32,
            "n_layers": 1,
            "dim_m": 16,
            "dim_i": 32,
            "vocab_size": 100,
            "emb_size": 50,
        }

    def test_init(self):
        VMFTransformer(**self.model_params)

    def test_forward(self):
        input = torch.randint(0, 100, (8, 15))
        output = torch.randint(0, 100, (8, 7))

        model = VMFTransformer(**self.model_params)
        generated = model(input, output)

        self.assertTupleEqual(generated.shape, (8, 6, 50))

    def test_inference(self):
        input = torch.randint(0, 100, (8, 15))

        model = VMFTransformer(**self.model_params)
        generated = model.inference(input, limit=7)
        self.assertTupleEqual(generated.shape, (8, 7))
