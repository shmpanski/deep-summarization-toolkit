import logging
import os
import typing

import torch
import yaml
from ignite.engine import Events, Engine
from torch.utils.data import DataLoader
from tqdm import tqdm

from dst import data, models
from dst.utils.metrics import RougeMetric

from .train import WORKBENCH_DIR

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, model_state_path):
        self.config = yaml.load(config)
        self.model_name = self.config["model"]
        self.margs = self.config["margs"]
        self.targs = self.config["targs"]
        self.dataset_name = self.config["dataset"]
        self.dargs = self.config["dargs"]
        self.eargs = self.config.get("eargs", dict())
        self.dump_directory = os.path.join(WORKBENCH_DIR, self.targs["prefix"])

        logger.info("Configuration for `%s` and `%s` dataset with launch prefix `%s` have been loaded",
                    self.model_name, self.dataset_name, self.targs["prefix"])

        self.test_dataset = self.load_dataset(self.dataset_name, self.dargs)

        logger.info("Test part loaded. Total size: %d", len(self.test_dataset))

        self.device = torch.device(self.targs["device"])
        logger.info("Device `%s` selected for evalution", self.device)

        # Create model
        self.model = self.instantiate_model(self.model_name, self.margs, model_state_path)
        self.model.to(self.device)
        logger.info("Model loaded.\nArguments: %s.\nTotal parameters: %d ",
                    str(self.margs),
                    sum(p.numel() for p in self.model.learnable_parameters()))

        self.pbar = tqdm(initial=0, leave=False, desc="Evaluation")

        # Choose evaluation metrics:
        self.eval_metrics = {
            "rouge": RougeMetric(output_transform=self.get_rouge_process_function(),
                                 metrics=["rouge-1", "rouge-2", "rouge-l"],
                                 stats=["f", "p", "r"])
        }

    def run(self):
        evaluator = self.model.create_evaluator(self.device, **self.eargs)
        val_loader = DataLoader(self.test_dataset,
                                self.targs.get("test_batch_size", 8),
                                shuffle=True,
                                collate_fn=self.test_dataset.collate_function,
                                drop_last=True)
        self.pbar.total = len(val_loader)

        evaluator.add_event_handler(Events.ITERATION_COMPLETED, self.get_evaluator_logger())

        for name, metric in self.eval_metrics.items():
            metric.attach(evaluator, name)

        logger.info("Start evaluation process")

        metrics = evaluator.run(val_loader).metrics
        metrics = {m: metrics["rouge"][m]["f"] for m in ["rouge-1", "rouge-2", "rouge-l"]}
        message = "Evaluation result: Rouge-1-f: {:.4f} | Rouge-2-f: {:.4f} | Rouge-l-f: {:.4f}"
        message = message.format(*metrics.values())
        logger.info(message)

        logger.info("Evaluation completed.")
        self.pbar.close()

    def get_evaluator_logger(self):
        def _evaluator_logger(engine: Engine):
            self.pbar.update(1)
        return _evaluator_logger

    def load_dataset(self, name, dargs):
        """Load dataset parts

        Args:
            name (str): Name of dataset class from `data` module.
            dargs (dict): Dataset arguments.

        Returns:
            tuple: pair of selected dataset for `train` and `test` parts.
        """

        init_args = dargs["init"]
        dataset_class = getattr(data, name)

        test_dataset = dataset_class(part="test", **init_args)

        return test_dataset

    def instantiate_model(self, name, marg, state_path):
        """Instantiate selected model.

        Args:
            name (string): Name of any model from `model` module.
            marg (dict): Model arguments.
            state_path (str): Path to model state.

        Returns:
            model.BaseSummarizationModel: Instantiated model.
        """
        model_class = getattr(models, name)
        instance, _ = model_class.create(self.test_dataset, marg)
        instance.load_state_dict(torch.load(state_path))
        return instance

    def get_rouge_process_function(self):
        def _rouge_process_function(output):
            generated, target = output
            generated_str = self.test_dataset.decode(generated)
            target_str = self.test_dataset.decode(target)
            return generated_str, target_str
        return _rouge_process_function
