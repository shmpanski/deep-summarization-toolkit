import unittest

import torch
import ignite.engine

from dst.models import SummarizationRNN
from torch.optim import Adam


class TestSummarizationRNNMethods(unittest.TestCase):
    def setUp(self):
        self.model = SummarizationRNN(vocab_size=100, hidden_size=200,
                                      embedding_size=100)
        self.optimizer = Adam(self.model.learnable_parameters())
        self.device = torch.device("cpu")

        self.input_seq = torch.LongTensor([[2, 98, 43, 5, 3],
                                           [2, 34, 23, 3, 0],
                                           [2, 52, 32, 7, 0]])
        self.output_seq = torch.LongTensor([[2, 98, 5, 3],
                                            [2, 23, 3, 0],
                                            [2, 32, 7, 0]])
        self.input_seq_length = torch.LongTensor([5, 4, 4])
        self.output_seq_length = torch.LongTensor([4, 3, 3])

    def test_model(self):
        seq_distr, att_distr = self.model(self.input_seq, self.output_seq,
                                          self.input_seq_length, self.output_seq_length)
        self.assertTupleEqual(seq_distr.shape, (3, 3, 100))
        self.assertTupleEqual(att_distr.shape, (3, 4, 5))

    def test_model_without_seq_lengths(self):
        seq_distr, att_distr = self.model(self.input_seq, self.output_seq)
        self.assertTupleEqual(seq_distr.shape, (3, 3, 100))
        self.assertTupleEqual(att_distr.shape, (3, 4, 5))

    def test_model_inference(self):
        seq_distr, seq = self.model.inference(self.input_seq, self.output_seq.shape[1], self.input_seq_length)
        self.assertTupleEqual(seq_distr.shape, (3, 3, 100))
        self.assertTupleEqual(seq.shape, (3, 3))

    def test_model_inference_without_seq_lengths(self):
        seq_distr, seq = self.model.inference(self.input_seq, self.output_seq.shape[1])
        self.assertTupleEqual(seq_distr.shape, (3, 3, 100))
        self.assertTupleEqual(seq.shape, (3, 3))

    def test_model_create_trainer(self):
        trainer = self.model.create_trainer(self.optimizer, self.device)
        self.assertIsInstance(trainer, ignite.engine.Engine)

    def test_model_create_evaluator(self):
        evaluator = self.model.create_evaluator(self.device)
        self.assertIsInstance(evaluator, ignite.engine.Engine)
