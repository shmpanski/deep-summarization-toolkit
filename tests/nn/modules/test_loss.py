import unittest

import torch

from dst.nn.modules import NLLvMF


class TestNLLvMFLoss(unittest.TestCase):
    def test_forward(self):
        loss_func = NLLvMF()
        target = torch.randn((16, 150))
        predicted = torch.randn(((16, 150)))
        loss = loss_func(predicted, target)
        self.assertTupleEqual(loss.shape, ())
