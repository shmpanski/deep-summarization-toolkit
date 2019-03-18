import os
import tempfile
import unittest

import torch
from jsonschema import ValidationError

from dst import train


class TestSummarizationTrainer(unittest.TestCase):
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

    def test_load_trainer(self):
        trainer = train.load_trainer(self.config)
        self.assertDictEqual(self.config, trainer.config)

    def test_load_trainer_validation(self):
        invalid_conf = {"prefix": "pref", "model": {"name": 42}}
        with self.assertRaises(ValidationError):
            train.load_trainer(invalid_conf)

    def test_instantiate_dataset(self):
        train_dataset, test_dataset = train.SummarizationTrainer.instantiate_datasets(
            self.config["dataset"]
        )
        self.assertEqual(len(train_dataset), 3)
        self.assertEqual(len(test_dataset), 3)

    def test_instantiate_model(self):
        train_dataset, _ = train.SummarizationTrainer.instantiate_datasets(
            self.config["dataset"]
        )
        model, args = train.SummarizationTrainer.instantiate_model(
            self.config["model"], train_dataset
        )
        self.assertTrue(hasattr(model, "forward"))
        self.assertIsInstance(args, dict)

    def test_instantiate_optimizer(self):
        train_dataset, _ = train.SummarizationTrainer.instantiate_datasets(
            self.config["dataset"]
        )
        model, args = train.SummarizationTrainer.instantiate_model(
            self.config["model"], train_dataset
        )
        optim = train.SummarizationTrainer.instantiate_optimizer(
            self.config["optimizer"], model.learnable_parameters()
        )
        self.assertTrue(hasattr(optim, "step"))

    def test_load_device(self):
        trainer = train.SummarizationTrainer(self.config)
        device = trainer.load_device({}, move_model=True)
        self.assertIsInstance(device, torch.device)

    def test_instantiate_dataloader(self):
        trainer = train.SummarizationTrainer(self.config)
        training_config = {"batches": {"train_size": 8, "eval_size": 16}}
        train_loader, test_loader = trainer.instantiate_dataloaders(training_config)
        self.assertEqual(train_loader.batch_size, 8)
        self.assertEqual(test_loader.batch_size, 16)

        train_loader, test_loader = trainer.instantiate_dataloaders({})
        self.assertEqual(train_loader.batch_size, 16)
        self.assertEqual(test_loader.batch_size, 32)

    def test_get_intervals(self):
        trainer = train.SummarizationTrainer(self.config)
        training_config = {"intervals": {"checkpoint": 100, "log": 10}}
        log_interval, checkpoint_interval = trainer.get_intervals(training_config)
        self.assertEqual(log_interval, 10)
        self.assertEqual(checkpoint_interval, 100)

        log_interval, checkpoint_interval = trainer.get_intervals({})
        self.assertEqual(log_interval, 100)
        self.assertEqual(checkpoint_interval, 1000)

    def test_run(self):
        trainer = train.SummarizationTrainer(self.config)
        trainer.run()
