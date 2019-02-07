import unittest
import torch

from dst.nn.modules import RNNEncoder, RNNDecoder


class TestRNNEncoderMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(ValueError):
            RNNEncoder(10, 10, rnn="another")

        gru_encoder = RNNEncoder(10, 10, rnn="GRU")
        lstm_encoder = RNNEncoder(10, 10, rnn="LSTM")

        self.assertEqual(gru_encoder.rnn.mode, "GRU")
        self.assertEqual(lstm_encoder.rnn.mode, "LSTM")

    def test_forward(self):
        batch_size = 7
        input_seq_len = 17
        input_seq_lens = [input_seq_len] * batch_size
        input_size, hidden_size = 100, 200

        encoder = RNNEncoder(input_size, hidden_size)
        input_seq = torch.randn(input_seq_len, batch_size, input_size)
        output, h = encoder(input_seq, input_seq_lens)

        self.assertTupleEqual(output.shape, (input_seq_len, batch_size, hidden_size))
        self.assertTupleEqual(h.shape, (3, batch_size, hidden_size))


class TestRNNDecoderMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        with self.assertRaises(ValueError):
            RNNDecoder(10, 10, rnn="another")

        with self.assertRaises(ValueError):
            RNNDecoder(10, 10, attention="another")

    def test_forward(self):
        batch_size = 7
        input_seq_len = 17
        output_seq_len = 9
        input_seq_lens = [input_seq_len] * batch_size
        output_seq_lens = [output_seq_len] * batch_size
        input_size, hidden_size = 100, 200

        encoder = RNNEncoder(input_size, hidden_size)
        decoder = RNNDecoder(input_size, hidden_size)

        input_seq = torch.randn(input_seq_len, batch_size, input_size)
        output_seq = torch.randn(output_seq_len, batch_size, input_size)

        encoder_output, h_e = encoder(input_seq, input_seq_lens)
        output, _, _ = decoder(output_seq, encoder_output, h_e, output_seq_lens)
        self.assertTupleEqual(output.shape, (output_seq_len, batch_size, hidden_size))
