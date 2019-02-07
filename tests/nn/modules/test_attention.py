import unittest
import torch

from dst.nn.modules import LuongAttention


class TestLuongAttentionMethods(unittest.TestCase):
    def setUp(self):
        self.available_scores = ["dot", "general"]

    def test_score(self):
        # `batch, seq, hidden`
        enc_seq = torch.randn(8, 15, 42)
        query = torch.randn(8, 1, 42)
        for score_t in self.available_scores:
            with self.subTest(score=score_t):
                attention = LuongAttention(score_t, query_size=42, key_size=42)
                score = attention.score_function(enc_seq, query)
                self.assertTupleEqual(score.shape, (8, 1, 15))

    def test_attention(self):
        # `batch, seq, hidden`
        enc_seq = torch.randn(8, 15, 42)
        dec_seq = torch.randn(8, 7, 42)

        for score_t in self.available_scores:
            with self.subTest("{} score".format(score_t)):
                attention = LuongAttention(score_t, query_size=42, key_size=42)
                context_vector, attention_distr = attention(enc_seq, enc_seq, dec_seq)
                self.assertTupleEqual(context_vector.shape, (8, 7, 42))
                self.assertTupleEqual(attention_distr.shape, (8, 7, 15))

    def test_attention_batch_first(self):
        enc_seq = torch.randn(15, 8, 42)
        dec_seq = torch.randn(7, 8, 42)
        attention = LuongAttention(batch_first=False)
        context_vector, attention_distr = attention(enc_seq, enc_seq, dec_seq)
        self.assertTupleEqual(context_vector.shape, (8, 7, 42))
        self.assertTupleEqual(attention_distr.shape, (8, 7, 15))
