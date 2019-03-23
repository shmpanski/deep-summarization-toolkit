"""Pipelines descriptions.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import jsonschema
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dst import data, models
from dst.utils import fill_dict_default_values
from dst.utils.metrics import RougeMetric

# Use JSON schema for validation and typing of necessary fields.
MODELRUN_SCHEMA = "dst/schemas/modelrun.schema.json"
WORKBENCH_DIR = "workbench"
logger = logging.getLogger(__name__)


class SummarizationPipeline:
    """Pipline for training, evaluating and sampling summarization models.

    Args:
        config (Dict): Pipeline configuration.
            See schema-JSON and :attr:`DEFAULT_CONFIGURATION` for more details.
        mode (str, optional): Defaults to "train". Running mode.
            Available: `train`, `eval` and `sample`.

    Raises:
        ValueError: In way of using invalid running mode.

    Notes:
        You can use train, evaluate and sample methods only in certain modes.
        Otherwise raise RuntimeError.

    """

    AVAIALABLE_MODES = ["train", "eval", "sample"]
    # Use this dictionary config to describe default values.
    DEFAULT_CONFIGURATION = {
        "prefix": str(),
        "model": {"name": str(), "args": dict()},
        "dataset": {
            "name": str(),
            "args": {
                "init": {"directory": str(), "prefix": str()},
                "preprocess": dict(),
            },
        },
        "optimizer": {"name": str(), "args": dict()},
        "training": {
            "name": str(),
            "device": "cpu",
            "epochs": 7,
            "batches": {"train_size": 16, "eval_size": 32},
            "intervals": {"checkpoint": 1000, "log": 100},
        },
        "evaluation": {"args": dict()},
        "sample": {"args": dict()},
    }

    def __init__(self, config: Dict, mode="train", **kwargs):
        if mode not in self.AVAIALABLE_MODES:
            raise ValueError(
                mode,
                " is invalid. Pipeline supports only `train`, `eval` or `sample` mode.",
            )

        fill_dict_default_values(config, self.DEFAULT_CONFIGURATION)
        self.mode = mode
        self.config = config
        self.dump_directory = os.path.join(WORKBENCH_DIR, config["prefix"])
        self.tensorboard_directory = os.path.join(self.dump_directory, "tensorboard")
        self.datasets = {}
        self.device = torch.device(config["training"]["device"])
        self.model = None
        self.model_args = {}
        self.optimizer = None
        self.tensorboard = SummaryWriter(self.tensorboard_directory)

        self.ModelClass = getattr(models, config["model"]["name"])
        self.OptimizerClass = getattr(torch.optim, config["optimizer"]["name"])
        self.DatasetClass = getattr(data, config["dataset"]["name"])

        if torch.cuda.is_available() and config["training"]["device"] == "cpu":
            logger.warning(
                "Your machine has cuda device. Use it to speed up training and evaluation processes."
            )

        if mode == "train":
            self._init_train(config)
        else:
            if "dump_file" not in kwargs:
                raise ValueError("Missing `dump_file` argument")
            if mode == "eval":
                self._init_eval(config, kwargs["dump_file"])
            else:
                self._init_sample(config, kwargs["dump_file"])

        self.model.to(self.device)

    def _init_train(self, config):
        # Prepare all required parameters:
        dataset_init_args = config["dataset"]["args"]["init"]
        dataset_preprocess_args = config["dataset"]["args"]["preprocess"]
        model_args = config["model"]["args"]
        optimizer_args = config["optimizer"]["args"]

        # Load train and test dataset parts:
        train_dataset = self.DatasetClass(
            part="train", **dataset_init_args, **dataset_preprocess_args
        )
        test_dataset = self.DatasetClass(
            part="test", **dataset_init_args, spm=train_dataset.get_spm()
        )
        self.datasets = {"train": train_dataset, "test": test_dataset}
        logger.info(
            "Dataset has been loaded. Train part size: %d; test part size: %d.",
            len(train_dataset),
            len(test_dataset),
        )

        # Instantiate model
        self.model, self.model_args = self.ModelClass.create(train_dataset, model_args)
        logger.info(
            "Model %s has been instantiated. Total parameters count: %d; Initial arguments: %s",
            self.ModelClass,
            sum(p.numel() for p in self.model.learnable_parameters()),
            self.model_args,
        )

        # Instantiate optimizer
        self.optimizer = self.OptimizerClass(
            self.model.learnable_parameters(), **optimizer_args
        )
        logger.info(
            "Optimizer %s has been instantiated. Initial arguments: %s",
            self.OptimizerClass,
            optimizer_args,
        )

    def _init_eval(self, config: Dict, model_dump_filename: str):
        dataset_init_args = config["dataset"]["args"]["init"]
        model_args = config["model"]["args"]

        # Load test dataset part:
        test_dataset = self.DatasetClass(part="test", **dataset_init_args)
        self.datasets = {"test": test_dataset}
        logger.info("Dataset has been loaded. Test part size: %d.", len(test_dataset))

        # Instantiate model
        self.model, self.model_args = self.ModelClass.create(test_dataset, model_args)
        logger.info(
            "Model %s has been instantiated. Total parameters count: %d; Initial arguments: %s",
            self.ModelClass,
            sum(p.numel() for p in self.model.learnable_parameters()),
            self.model_args,
        )

        # Load model state from checkpoint
        checkpoint = torch.load(model_dump_filename)
        self.model.load_state_dict(checkpoint)
        logger.info("Model state dictionary loaded from %s", model_dump_filename)

    def _init_sample(self, config: Dict, model_dump_filename: str):
        dataset_init_args = config["dataset"]["args"]["init"]
        model_args = config["model"]["args"]

        sample_dataset = self.DatasetClass(part=None, **dataset_init_args)
        self.datasets = {"sample": sample_dataset}
        self.model, self.model_args = self.ModelClass.create(sample_dataset, model_args)
        logger.info(
            "Model %s has been instantiated. Total parameters count: %d; Initial arguments: %s",
            self.ModelClass,
            sum(p.numel() for p in self.model.learnable_parameters()),
            self.model_args,
        )

        # Load model state from checkpoint
        checkpoint = torch.load(model_dump_filename)
        self.model.load_state_dict(checkpoint)
        logger.info("Model state dictionary loaded from %s", model_dump_filename)

    def train(self):
        """Run train process.
        """
        if self.mode != "train":
            raise RuntimeError("You cannot `train` in `%s` mode.", self.mode)

        train_bs = self.config["training"]["batches"]["train_size"]
        eval_bs = self.config["training"]["batches"]["eval_size"]
        checkpoint_interval = self.config["training"]["intervals"]["checkpoint"]
        log_interval = self.config["training"]["intervals"]["log"]
        train_loader = self._get_dataloader(self.datasets["train"], train_bs)
        eval_loader = self._get_dataloader(self.datasets["test"], eval_bs)

        trainer_engine = self.model.create_trainer(self.optimizer, self.device)
        evaluation_engine = self._evaluation_engine(log_tensorboard=True)
        trainer_saver = ModelCheckpoint(
            self.dump_directory + "/checkpoints",
            "checkpoint",
            save_interval=checkpoint_interval,
            save_as_state_dict=True,
        )
        best_state_saver = ModelCheckpoint(
            self.dump_directory + "/best_models",
            "best",
            score_name="rouge",
            score_function=lambda e: e.state.metrics["rouge"]["rouge-1"]["f"],
            n_saved=3,
            save_as_state_dict=True,
        )
        checkpoint_objects = {"model": self.model, "optim": self.optimizer}
        best_model_objects = {"model": self.model}

        trainer_engine.add_event_handler(
            Events.ITERATION_COMPLETED, trainer_saver, checkpoint_objects
        )
        evaluation_engine.add_event_handler(
            Events.COMPLETED, best_state_saver, best_model_objects
        )

        @trainer_engine.on(Events.ITERATION_COMPLETED)
        def log_trainer(e: Engine):
            iteration = (e.state.iteration - 1) % len(train_loader) + 1
            epoch = e.state.epoch
            loss = e.state.output

            if iteration % log_interval == 0:
                pattern = "Epoch[{}] | Iteration [{}/{}] | Loss: {:.4f}"
                message = pattern.format(epoch, iteration, len(train_loader), loss)
                logger.info(message)
                self.tensorboard.add_scalar("training/loss", loss, iteration)

        # Evaluate model.
        @trainer_engine.on(Events.EPOCH_COMPLETED)
        def evaluate(e: Engine):
            logger.info("Start model evaluation.")

            evaluation_engine.run(eval_loader)

        # Sample one batch
        @evaluation_engine.on(Events.ITERATION_COMPLETED)
        def sample(e: Engine):
            if e.state.iteration == 1:
                samples, targets = e.state.output
                articles = e.state.batch["src"]
                articles_str, samples_str, targets_str = [
                    self.datasets["train"].decode(s)
                    for s in [articles, samples, targets]
                ]
                for i, d in enumerate(zip(articles_str, targets_str, samples_str)):
                    article, target, sample = d
                    pattern = "Sample [%d]: \nText: %s\nOriginal summary: %s\nGenerated summary: %s"
                    logging.info(pattern, i, article, target, sample)

        @trainer_engine.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine: Engine, e: Exception):
            if isinstance(e, KeyboardInterrupt) and engine.state.iteration > 1:
                logger.warning("KeyboardInterapt caught. Exiting.")
                engine.terminate()
            else:
                raise e

        self.attach_progress_bar(trainer_engine, lambda x: {"loss": x})
        trainer_engine.run(train_loader, self.config["training"]["epochs"])

    def evaluate(self) -> Dict:
        """Run evalution process.

        Returns:
            Dict: Evaluation metrics values.
        """
        if self.mode != "eval":
            raise RuntimeError("You cannot `evaluate` in `%s` mode.", self.mode)

        eval_bs = self.config["training"]["batches"]["eval_size"]
        eval_loader = self._get_dataloader(self.datasets["test"], eval_bs)
        evaluation_engine = self._evaluation_engine(log_tensorboard=False)

        logger.info("Start model evaluation")
        evaluation_engine.run(eval_loader)

        return evaluation_engine.state.metrics

    def sample(self, input: List[str]) -> List[List[str]]:
        """Sample example.

        Args:
            input (List[str]): Input text, need to summarize.

        Returns:
            List[List[str]]: Summarizations.
        """
        if self.mode != "sample":
            raise RuntimeError("You cannot `sample` in `%s` mode.", self.mode)

        if isinstance(input, str):
            input = [input]
        dataset = self.datasets["sample"]
        input = dataset.encode(input).to(self.device)
        summaries, _ = self.model.inference(input, **self.config["sample"]["args"])
        summaries_strs = dataset.decode(summaries)

        return summaries_strs

    def _evaluation_engine(self, log_tensorboard=False) -> Engine:
        evaluation_args = self.config["evaluation"]["args"]
        evaluation_engine = self.model.create_evaluator(self.device, **evaluation_args)

        self.attach_metrics(evaluation_engine)
        self.attach_progress_bar(evaluation_engine, desc="Evaluating ")

        @evaluation_engine.on(Events.COMPLETED)
        def _log_eval(e: Engine):
            logger.info("Evaluation completed.")
            metrics = e.state.metrics
            for metric_name, metric in metrics.items():
                logger.info("Evalutaion metric %s: %s", metric_name, str(metric))
            if log_tensorboard:
                to_tensorboard = {m: metrics["rouge"][m]["f"] for m in metrics["rouge"]}
                self.tensorboard.add_scalars("evaluating/", to_tensorboard)

        return evaluation_engine

    @staticmethod
    def _get_dataloader(dataset, batch_size):
        loader = DataLoader(
            dataset, batch_size, True, collate_fn=dataset.collate_function
        )
        return loader

    @staticmethod
    def attach_progress_bar(
        e: Engine, output_transform=None, **tqdm_kwargs
    ) -> ProgressBar:
        """Attach progress bar to engine.

        Args:
            e (engine.Engine): Engine.
            output_transform (lambda, optional): Engine output transformation.

        Returns:
            ProgressBar: Created progress bar.
        """

        pbar = ProgressBar(
            bar_format=(
                "{desc}[{n_fmt}/{total_fmt}]"
                "{percentage:3.0f}%|{bar}{postfix} "
                "[{elapsed}<{remaining},{rate_fmt}]"
            ),
            **tqdm_kwargs
        )
        pbar.attach(e, output_transform=output_transform)

    def attach_metrics(self, e: Engine):
        """Attach metrics to engine.

        Args:
            e (engine.Engine): Engine.
        """
        # TODO: add custom metrics.

        def _rouge_process_function(output):
            generated, target = output
            generated_str = self.datasets["test"].decode(generated)
            target_str = self.datasets["test"].decode(target)
            return generated_str, target_str

        metrics = {"rouge": RougeMetric(_rouge_process_function)}

        for metric_name, metric in metrics.items():
            metric.attach(e, metric_name)


def validate_config(config: Dict):
    """Validate config with JSON schema.

    Args:
        config (Dict): Configuration dictionary.
    """

    with open(MODELRUN_SCHEMA) as schema_file:
        schema = json.load(schema_file)
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            logger.error(e.message)
            raise


def load_pipeline(config: Dict, mode="train", **kwargs):
    """Load pipeline with required config.

    Args:
        config (Dict): Pipeline configuration.

    Returns:
        object: Pipeline object.
    """

    validate_config(config)

    pipeline_class = globals()[config["training"]["name"]]
    return pipeline_class(config=config, mode=mode, **kwargs)
