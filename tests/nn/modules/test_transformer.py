import unittest
import torch
from dst.nn import PBATransformerEncoderLayer, PBATransformerDecoderLayer


class TestPBATransformerEncoderLayer(unittest.TestCase):
    def test_homogeneous_forward(self):
        dim_m, dim_proj, dim_i, dropout = 32, 16, 64, 0.1
        attention = "homogeneous"
        attention_args = dict(head_convs=(3, 2, 3))
        batch_size, seq_len = 8, 7

        input = torch.randn(batch_size, seq_len, dim_m)
        layer = PBATransformerEncoderLayer(dim_m, dim_proj, dim_i, dropout, attention, **attention_args)
        out = layer(input)

        self.assertTupleEqual(out.shape, input.shape)

    def test_heterogeneous_forward(self):
        dim_m, dim_proj, dim_i, dropout = 32, 16, 64, 0.1
        attention = "heterogeneous"
        attention_args = dict(head_convs=(1, 0, 3), n_heads=8)
        batch_size, seq_len = 8, 7

        input = torch.randn(batch_size, seq_len, dim_m)
        layer = PBATransformerEncoderLayer(dim_m, dim_proj, dim_i, dropout, attention, **attention_args)
        out = layer(input)

        self.assertTupleEqual(out.shape, input.shape)


class TestPBATransformerDecoderLayer(unittest.TestCase):
    def test_homogeneous_forward(self):
        dim_m, dim_proj, dim_i, dropout = 32, 16, 64, 0.1
        attention = "homogeneous"
        attention_args = dict(head_convs=(3, 2, 3))
        batch_size, inp_seq_len, enc_seq_len = 8, 7, 9

        input_seq = torch.randn(batch_size, inp_seq_len, dim_m)
        encoder_seq = torch.randn(batch_size, enc_seq_len, dim_m)
        mask = torch.zeros(batch_size, inp_seq_len, inp_seq_len, dtype=torch.uint8)

        layer = PBATransformerDecoderLayer(dim_m, dim_proj, dim_i, dropout, attention, **attention_args)
        out = layer(input_seq, encoder_seq, mask)

        self.assertTupleEqual(out.shape, input_seq.shape)

    def test_heterogeneous_forward(self):
        dim_m, dim_proj, dim_i, dropout = 32, 16, 64, 0.1
        attention = "heterogeneous"
        attention_args = dict(head_convs=(1, 0, 3), n_heads=8)
        batch_size, inp_seq_len, enc_seq_len = 8, 7, 9

        input_seq = torch.randn(batch_size, inp_seq_len, dim_m)
        encoder_seq = torch.randn(batch_size, enc_seq_len, dim_m)
        mask = torch.zeros(batch_size, inp_seq_len, inp_seq_len, dtype=torch.uint8)

        layer = PBATransformerDecoderLayer(dim_m, dim_proj, dim_i, dropout, attention, **attention_args)
        out = layer(input_seq, encoder_seq, mask)

        self.assertTupleEqual(out.shape, input_seq.shape)
