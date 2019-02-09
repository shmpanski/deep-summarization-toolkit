import unittest
import torch
from torch import nn

from dst.nn.modules import (LuongAttention, MultiHeadPhrasalAttentionBase,
                            MultiHeadHomogeneousAttention, MultiHeadHeterogeneousAttention)


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


class TestMultiHeadPhrasalAttentionBaseMethods(unittest.TestCase):

    def test_stack_heads(self):
        batch_size, seq_len, dim = 8, 5, 32
        tensor = torch.randn(batch_size, seq_len, dim)

        stacked = MultiHeadPhrasalAttentionBase.stack_heads(tensor, 7)
        self.assertTupleEqual(stacked.shape, (7, batch_size, seq_len, dim))

    def test_stack_convolution(self):
        head_convs = (2, 0, 2)
        input_dim, output_dim = 32, 32
        convs = MultiHeadPhrasalAttentionBase.stack_convolutions(head_convs, input_dim, output_dim)
        convs = list(convs)
        self.assertEqual(len(convs), 2)
        self.assertIsInstance(convs[-1], nn.Conv1d)


class TestMultiHeadHomogeneousAttentionMethods(unittest.TestCase):
    def setUp(self):
        self.attention = MultiHeadHomogeneousAttention(dim_m=64,
                                                       dim_proj=32,
                                                       head_convs=(2, 0, 2))

    def test_calculate_conv_heads(self):
        batch_size, inp_seq_len = 8, 10
        value = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)

        heads = self.attention.calculate_conv_heads(value, self.attention.value_projections, self.attention.head_convs)
        self.assertTupleEqual(heads.shape,
                              (self.attention.total_n_heads, batch_size, inp_seq_len, self.attention.dim_proj))

    def test_forward_masked(self):
        batch_size, inp_seq_len, out_seq_len = 8, 10, 10
        value = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        key = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        query = torch.randn(batch_size, out_seq_len, self.attention.dim_m)
        mask = torch.ones((batch_size, out_seq_len, out_seq_len), dtype=torch.uint8)

        result = self.attention(value, key, query, mask)

        self.assertTupleEqual(result.shape, (batch_size, out_seq_len, self.attention.dim_m))

    def test_forward(self):
        batch_size, inp_seq_len, out_seq_len = 8, 10, 7
        value = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        key = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        query = torch.randn(batch_size, out_seq_len, self.attention.dim_m)

        result = self.attention(value, key, query)

        self.assertTupleEqual(result.shape, (batch_size, out_seq_len, self.attention.dim_m))


class TestMultiHeadHeterogeneousAttentionMethods(unittest.TestCase):
    def setUp(self):
        self.attention = MultiHeadHeterogeneousAttention(dim_m=64,
                                                         dim_proj=32,
                                                         head_convs=(1, 2, 3),
                                                         n_heads=8)

    def test_calculate_conv_heads(self):
        batch_size, inp_seq_len = 8, 10
        value = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)

        heads = self.attention.calculate_conv_heads(value, self.attention.value_projections, self.attention.head_convs)
        self.assertTupleEqual(heads.shape,
                              (self.attention.total_n_heads, batch_size, 3 * inp_seq_len - 3, self.attention.dim_proj))

    def test_forward_masked(self):
        batch_size, inp_seq_len, out_seq_len = 8, 10, 10
        value = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        key = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        query = torch.randn(batch_size, out_seq_len, self.attention.dim_m)
        mask = torch.ones((batch_size, out_seq_len, out_seq_len), dtype=torch.uint8)

        result = self.attention(value, key, query, mask)

        self.assertTupleEqual(result.shape, (batch_size, out_seq_len, self.attention.dim_m))

    def test_forward(self):
        batch_size, inp_seq_len, out_seq_len = 8, 10, 7
        value = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        key = torch.randn(batch_size, inp_seq_len, self.attention.dim_m)
        query = torch.randn(batch_size, out_seq_len, self.attention.dim_m)

        result = self.attention(value, key, query)

        self.assertTupleEqual(result.shape, (batch_size, out_seq_len, self.attention.dim_m))
