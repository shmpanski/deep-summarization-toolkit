import unittest

import torch

from dst.nn.functional import ive, logC


class TestVMF(unittest.TestCase):
    def test_ive_call(self):
        v = torch.Tensor([1])
        z = torch.Tensor([2])
        result = ive(v, z).item()
        self.assertAlmostEqual(result, 0.2, places=1)

    def test_logC(self):
        k = torch.Tensor([15])
        m = torch.Tensor([300])
        result = logC(k, m).item()
        self.assertAlmostEqual(result, 34.92, places=1)

    def test_logC_backward(self):
        m = torch.full((1,), 2, requires_grad=True)
        k = torch.Tensor([1])
        logC(k, m).backward()
        self.assertAlmostEqual(m.grad.item(), -0.96, places=1)
