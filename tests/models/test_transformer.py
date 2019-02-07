import unittest
import torch
from torch import nn

from models import SummarizationTransformer


class TestSummarizationTransformerMethods(unittest.TestCase):
    def setUp(self):
        self.model = SummarizationTransformer(max_seq_len=50,
                                              vocab_size=100,
                                              embedding_weights=None,
                                              n_layers=1,
                                              emb_size=100,
                                              dim_m=64,
                                              n_heads=1,
                                              dim_i=128,
                                              dropout=0.1)
        self.input_seq = torch.LongTensor([[2, 98, 43, 5, 3],
                                           [2, 34, 23, 3, 0],
                                           [2, 52, 32, 7, 0]])
        self.output_seq = torch.LongTensor([[2, 98, 5, 3],
                                            [2, 23, 3, 0],
                                            [2, 32, 7, 0]])

    def test_model(self):
        seq_distr, seq = self.model(self.input_seq, self.output_seq)

        self.assertEqual(seq_distr.shape, (3, 4, 100))
        self.assertEqual(seq.shape, self.output_seq.shape)

    def test_model_inference(self):
        limit = self.output_seq.shape[1]
        seq_distr, seq = self.model.inference(self.input_seq, limit)
        self.assertEqual(seq_distr.shape, (3, 4, 100))
        self.assertEqual(seq.shape, self.output_seq.shape)
