"""Training tools.
"""

import json
import logging
import os
from typing import Dict, Iterable, Tuple

import jsonschema
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dst import data, models
from dst.utils.metrics import RougeMetric

MODELRUN_SCHEMA = "dst/schemas/modelrun.schema.json"
WORKBENCH_DIR = "workbench"
logger = logging.getLogger(__name__)


class SummarizationTrainer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        logger.info("Configuration %s has been loaded", config["prefix"])

        self.dump_directory = os.path.join(WORKBENCH_DIR, config["prefix"])
        self.tensorboard_directory = os.path.join(self.dump_directory, "tensorboard")
        self.training = config.get("training", {})
        self.evaluation = config.get("evaluation", {})

        # Instantiate dataset, model and optimizer:
        self.train_dataset, self.test_dataset = self.instantiate_datasets(
            config["dataset"]
        )
        self.model, _ = self.instantiate_model(config["model"], self.train_dataset)
        self.optimizer = self.instantiate_optimizer(
            config["optimizer"], self.model.learnable_parameters()
        )

        self.device = self.load_device(config["training"], move_model=True)
        self.tensorboard = SummaryWriter(self.tensorboard_directory)

    def run(self):
        train_loader, test_loader = self.instantiate_dataloaders(self.training)
        log_interval, checkpoint_interval = self.get_intervals(self.training)

        # Describe training  pipeline
        trainer_engine = self.model.create_trainer(self.optimizer, self.device)
        evaluate_engine = self.model.create_evaluator(
            self.device, **self.evaluation.get("args", {})
        )
        trainer_saver = ModelCheckpoint(
            self.dump_directory, "checkpoint", save_interval=checkpoint_interval
        )
        checkpoint_objects = {"model": self.model, "optim": self.optimizer}
        best_state_saver = ModelCheckpoint(
            self.dump_directory,
            "best",
            score_name="rouge",
            score_function=lambda e: e.state.metrics["rouge"]["rouge-1"]["f"],
            n_saved=3,
        )

        self.attach_metrics(evaluate_engine)
        self.attach_progress_bar(evaluate_engine, desc="Evaluation ")

        # Attach event handlers
        trainer_engine.add_event_handler(
            Events.ITERATION_COMPLETED, trainer_saver, checkpoint_objects
        )
        evaluate_engine.add_event_handler(
            Events.COMPLETED, best_state_saver, {"model": self.model}
        )

        # Log loss and process during training.
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

            evaluate_engine.run(test_loader)
            metrics = evaluate_engine.state.metrics
            for metric_name, metric in metrics.items():
                logger.info("Evalutaion metric %s: %s", metric_name, str(metric))
            to_tensorboard = {m: metrics["rouge"][m]["f"] for m in metrics["rouge"]}
            self.tensorboard.add_scalars("evaluating/", to_tensorboard)

        # Sample one batch
        @evaluate_engine.on(Events.ITERATION_COMPLETED)
        def sample(e: Engine):
            if e.state.iteration == 1:
                samples, targets = e.state.output
                articles = e.state.batch["src"]
                articles_str, samples_str, targets_str = [
                    self.train_dataset.decode(s) for s in [articles, samples, targets]
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
        trainer_engine.run(train_loader, self.training.get("epochs", 7))

    @staticmethod
    def instantiate_datasets(dataset_config: Dict) -> Tuple[data.SummarizationDataset]:
        """Instatiate train and test datasets.

        Args:
            dataset_config (Dict): Dataset configuration.

        Returns:
            Tuple[data.SummarizationDataset]: train and test dataset objects.
        """

        dataset_name = dataset_config["name"]
        init_args = dataset_config["args"].get("init", {})
        preprocess_args = dataset_config["args"].get("preprocess", {})
        dataset_class = getattr(data, dataset_name)

        # For now, we support only BPE dataset. Reuse tokenizer for both dataset parts.
        train_dataset = dataset_class(part="train", **init_args, **preprocess_args)
        test_dataset = dataset_class(
            part="test", **init_args, spm=train_dataset.get_spm()
        )

        logger.info(
            "Dataset %s with has been instantiated. Train size: %d; Test size: %d",
            dataset_name,
            len(train_dataset),
            len(test_dataset),
        )
        return train_dataset, test_dataset

    @staticmethod
    def instantiate_model(
        model_config: Dict, dataset
    ) -> Tuple[models.BaseSummarizationModel, dict]:
        """Instantiate model.

        Args:
            model_config (Dict): Model configuration.
            dataset (SummarizationDataset): Reference dataset.
        Notes:
            During creating model it's necessary to know some info about used dataset.
            For example: pretrained embeddings, data lengths and etc.
            To serve dataset needs we pass training dataset for model create function.

        Returns:
            Tuple[models.BaseSummarizationModel, dict]: Model instance and initialization it's args.
        """

        model_name = model_config["name"]
        init_args = model_config.get("args", {})
        model_class = getattr(models, model_name)

        model, model_args = model_class.create(dataset, init_args)

        logger.info(
            "Model %s has been instantiated. Total parameters count: %d; Initial arguments: %s",
            model_name,
            sum(p.numel() for p in model.learnable_parameters()),
            str(model_args),
        )
        return model, model_args

    @staticmethod
    def instantiate_optimizer(
        optim_config: Dict, learnable_parameters: Iterable
    ) -> torch.optim.Optimizer:
        """Instantiate optimizer

        Args:
            optim_config (Dict): Optimizer configuration.
            learnable_parameters (Iterable): Model learnable parameters.

        Returns:
            torch.optim.Optimizer: Instantiated optimizer.
        """

        optim_name = optim_config["name"]
        init_args = optim_config.get("args", {})
        optim_class = getattr(torch.optim, optim_name)

        logger.info(
            "Optimizer %s has been instantiated. Initial arguments: %s",
            optim_name,
            init_args,
        )
        return optim_class(learnable_parameters, **init_args)

    def load_device(self, training_config: Dict, move_model=False) -> torch.device:
        """Load torch device.

        Args:
            training_config (Dict): Training configuration.
            move_model (bool, optional): Defaults to False. Whether to move model to selected device.

        Raises:
            RuntimeError: Raises if system has no required device.

        Returns:
            torch.device: Selected device.
        """

        device_name = training_config.get("device", "cpu")
        if torch.cuda.is_available():
            if device_name == "cpu":
                logger.warning(
                    "You have cuda device. Change `training.device` config to use GPU acceleration."
                )
        else:
            if device_name != "cpu":
                logger.error(
                    "You have no cuda device. Change `training.device` config to `cpu` to run your model"
                )
                raise RuntimeError("You have no cuda device")

        device = torch.device(device_name)
        logger.info("Selected %s device for training.", device_name)
        if move_model:
            self.model.to(device)
        return device

    def instantiate_dataloaders(self, training_config: Dict) -> Tuple[DataLoader]:
        """Instantiate data loaders.
        Args:
            training_config (Dict): Training configuration. Needs to determine batch sizes.

        Returns:
            Tuple[torch.utils.data.DataLoader]: Train and test data loaders.
        """

        assert all(
            [self.train_dataset is not None, self.test_dataset is not None]
        ), "Dataloader requires instantiated dataset"

        if "batches" in training_config:
            train_batch_size = training_config["batches"].get("train_size", 16)
            test_batch_size = training_config["batches"].get("eval_size", 32)
        else:
            train_batch_size = 16
            test_batch_size = 32

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_function,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            collate_fn=self.test_dataset.collate_function,
        )

        return train_loader, test_loader

    @staticmethod
    def get_intervals(training_config: Dict) -> Tuple[int]:
        """Get interval values from config. Return default values if they are not provided.

        Args:
            training_config (Dict): Training configuration. Needs to determine intervals.

        Returns:
            Tuple[int]: Train logging interval and checkpoint interval.
        """

        if "intervals" in training_config:
            intervals = training_config["intervals"]
            log_interval = intervals.get("log", 100)
            checkpoint_interval = intervals.get("checkpoint", 1000)
        else:
            log_interval = 100
            checkpoint_interval = 1000
        return log_interval, checkpoint_interval

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
            generated_str = self.train_dataset.decode(generated)
            target_str = self.train_dataset.decode(target)
            return generated_str, target_str

        metrics = {"rouge": RougeMetric(_rouge_process_function)}

        for metric_name, metric in metrics.items():
            metric.attach(e, metric_name)


def load_trainer(config: Dict):
    """Load trainer with required config.

    Args:
        config (Dict): Model training configuration.

    Returns:
        object: Trainer object, inherited from :class:`BaseTrainer`.
    """

    # Validate config with JSON schema
    with open(MODELRUN_SCHEMA) as schema_file:
        schema = json.load(schema_file)
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            logger.error(e.message)
            raise

    trainer_class = globals()[config["training"]["name"]]
    return trainer_class(config=config)
