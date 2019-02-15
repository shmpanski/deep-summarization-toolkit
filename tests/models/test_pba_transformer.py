import unittest

import torch

from dst.models import PBATransformer


class TestPBATransformerMethods(unittest.TestCase):
    def setUp(self):
        max_seq_len = 42
        num_layers = 3
        emb_size = 100
        dim_i = 64
        n_heads = 4
        head_convs = (1, 2)
        self.vocab_size = 100
        self.dim_m = 32
        self.model = PBATransformer(max_seq_len, self.vocab_size,
                                    num_layers=num_layers,
                                    emb_size=emb_size,
                                    dim_m=self.dim_m,
                                    dim_i=dim_i,
                                    n_heads=n_heads,
                                    head_convs=head_convs)

    def test_forward(self):
        batch_size, source_seq_len, target_seq_len = 8, 17, 7
        source = torch.randint(100, (batch_size, source_seq_len))
        target = torch.randint(100, (batch_size, target_seq_len))

        output = self.model(source, target)
        self.assertTupleEqual(output.shape, (batch_size, target_seq_len, self.vocab_size))

    def test_inference(self):
        batch_size, source_seq_len, limit = 8, 17, 7
        source = torch.randint(100, (batch_size, source_seq_len))

        output, distribution = self.model.inference(source, limit)
        self.assertTupleEqual(output.shape, (batch_size, limit))
        self.assertTupleEqual(distribution.shape, (batch_size, limit, self.vocab_size))
