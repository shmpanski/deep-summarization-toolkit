import os
import tempfile
import unittest

import torch
from jsonschema import ValidationError

from dst import train


class TestSummarizationPipeline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.dirname = self.directory.name
        train.WORKBENCH_DIR = os.path.join(self.directory.name, "workbench")

        random_senteces = [
            "I have writen some code\tWrite code\r\n",
            "She advised him to come back at once.\tSome words\r\n",
            "We need to rent a room for our party.\tIt's random sentece\r\n",
        ]

        with open(self.dirname + "/train.tsv", "w+") as train_file:
            train_file.writelines(random_senteces)

        with open(self.dirname + "/test.tsv", "w+") as dev_file:
            dev_file.writelines(random_senteces)

        with open(self.dirname + "/dev.tsv", "w+") as test_file:
            test_file.writelines(random_senteces)

        self.config = {
            "prefix": "test-model",
            "model": {
                "name": "SummarizationRNN",
                "args": {
                    "vocab_size": 42,
                    "hidden_size": 8,
                    "embedding_size": 16,
                    "num_layers": 1,
                },
            },
            "dataset": {
                "args": {
                    "init": {"prefix": "test-dataset", "directory": self.dirname},
                    "preprocess": {"vocab_size": 42, "embedding_size": 16},
                },
                "name": "BPEDataset",
            },
            "optimizer": {"name": "Adam"},
            "training": {"name": "SummarizationTrainer", "epochs": 1},
        }

    def tearDown(self):
        self.directory.cleanup()

    def test_train_mode_init(self):
        pipeline = train.SummarizationPipeline(self.config)
        self.assertEqual(pipeline.config["training"]["batches"]["train_size"], 16)
        self.assertEqual(len(pipeline.datasets), 2)
        self.assertEqual(len(pipeline.datasets["train"]), 3)
        self.assertEqual(len(pipeline.datasets["test"]), 3)
        self.assertIsNotNone(pipeline.model)
        self.assertIsNotNone(pipeline.optimizer)

    def test_train(self):
        pipeline = train.SummarizationPipeline(self.config)
        pipeline.train()


