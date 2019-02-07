import unittest

import torch
from torch.optim import Adam
from ignite.engine import Engine

from dst.models import ConvTransformer


class TestConvTransformerMethods(unittest.TestCase):
    def setUp(self):
        self.model = ConvTransformer(100, 100, emb_size=100, num_layers=1, dim_m=64, n_heads=2)

    def test_model_init(self):
        self.assertIsInstance(self.model, ConvTransformer)

    def test_model_forward(self):
        input_seq = torch.randint(0, 100, (5, 42))
        output_seq = torch.randint(0, 100, (5, 7))

        distr = self.model(input_seq, output_seq)
        self.assertTupleEqual(distr.shape, (5, 6, 100))

    def test_model_inference(self):
        input_seq = torch.randint(0, 100, (5, 42))

        generated_seq, generated_distr = self.model.inference(input_seq, 10)
        self.assertTupleEqual(generated_seq.shape, (5, 10))
        self.assertTupleEqual(generated_distr.shape, (5, 10, 100))

    def test_create_trainer(self):
        optimizer = Adam(self.model.learnable_parameters())
        trainer = self.model.create_trainer(optimizer, torch.device("cpu"))
        self.assertIsInstance(trainer, Engine)

    def test_create_evaluator(self):
        evaluator = self.model.create_evaluator(torch.device("cpu"))
        self.assertIsInstance(evaluator, Engine)
