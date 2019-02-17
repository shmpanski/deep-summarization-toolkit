import unittest

import torch

from dst.data.decoding import BeamSearch


class TestBeamSearchMethods(unittest.TestCase):
    def setUp(self):
        self.k = 5
        self.beam_search = BeamSearch(self.k)

    def test_initial_update(self):
        vocab = 100
        output = torch.randn(vocab).softmax(-1)
        self.beam_search.update(output)

        score_shape = self.beam_search.scores.shape
        sequences = self.beam_search.sequences
        self.assertTupleEqual(score_shape, (self.k, 1))
        self.assertEqual(len(sequences), self.k)
        self.assertEqual(len(sequences[0]), 1)

    def test_multiple_update(self):
        vocab = 100
        length = 10

        for t in range(length):
            if t == 0:
                output = torch.randn(vocab).softmax(-1)
            else:
                output = torch.randn(self.k, vocab).softmax(-1)
            self.beam_search.update(output)

        score_shape = self.beam_search.scores.shape
        sequences = self.beam_search.sequences
        self.assertTupleEqual(score_shape, (self.k, 1))
        self.assertEqual(len(sequences), self.k)
        self.assertEqual(len(sequences[0]), length)

    def test_search(self):
        vocab = 100
        length = 10

        for t in range(length):
            if t == 0:
                output = torch.randn(vocab).softmax(-1)
            else:
                output = torch.randn(self.k, vocab).softmax(-1)
            self.beam_search.update(output)
        sequence = self.beam_search.search()

        self.assertTupleEqual(sequence.shape, (self.k, length))
