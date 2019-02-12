import importlib
import logging
import os

import torch
import yaml
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from dst import data
from dst import models
from dst.utils.metrics import RougeMetric

from tensorboardX import SummaryWriter


WORKBENCH_DIR = "./workbench"


class Trainer:
    """Simple trainer class.

    TODO:
        - Default arguments
        - Custom metrics with transform functions
        - Sampler and evaluator
    """

    def __init__(self, config):
        # Load config from yaml file
        self.config = yaml.load(config)
        self.model_name = self.config["model"]
        self.margs = self.config["margs"]
        self.dataset_name = self.config["dataset"]
        self.dargs = self.config["dargs"]
        self.optimizer_name = self.config["optimizer"]
        self.oargs = self.config["oargs"]
        self.targs = self.config["targs"]
        self.dump_directory = os.path.join(WORKBENCH_DIR, self.targs["prefix"])
        self.tensorboard_log_dir = os.path.join(WORKBENCH_DIR, "logs", self.targs["prefix"])
        logging.info("Configuration for `%s` and `%s` dataset with launch prefix `%s` have been loaded",
                     self.model_name, self.dataset_name, self.targs["prefix"])

        # Load selected datasets
        self.train_dataset, self.test_dataset = self.load_dataset(self.dataset_name,
                                                                  self.dargs)
        logging.info("Dataset loaded. Train part size: %d. Test part size: %d",
                     len(self.train_dataset), len(self.test_dataset))

        # Select device
        self.device = torch.device(self.targs["device"])
        logging.info("Device `%s` selected for training", self.device)

        # Create model
        self.model, self.model_args = self.instantiate_model(self.model_name, self.margs)
        self.model.to(self.device)
        logging.info("Model loaded.\nArguments: %s.\nTotal parameters: %d ",
                     str(self.model_args),
                     sum(p.numel() for p in self.model.learnable_parameters()))

        # Create optimizer
        self.optimizer = self.instantiate_optim(self.optimizer_name, self.oargs, self.model.learnable_parameters())
        logging.info("Create `%s` optimizer with parameters: %s", self.optimizer_name, self.oargs)

        self.tb_writer = SummaryWriter(self.tensorboard_log_dir)
        self.pbar_descr = "Epoch[{}] | Loss[{:.2f}]"
        self.pbar = tqdm(initial=0, leave=False, desc=self.pbar_descr)

        # Choose evaluation metrics:
        self.eval_metrics = {
            "rouge": RougeMetric(output_transform=self.get_rouge_process_function(),
                                 metrics=["rouge-1", "rouge-2", "rouge-l"],
                                 stats=["f", "p", "r"])
        }

    def load_dataset(self, name, dargs):
        """Load dataset parts

        Args:
            name (str): Name of dataset class from `data` module.
            dargs (dict): Dataset arguments.

        Returns:
            tuple: pair of selected dataset for `train` and `test` parts.
        """

        init_args = dargs["init"]
        preprocess_args = dargs["preprocess"]
        dataset_class = getattr(data, name)

        train_dataset = dataset_class(part="train", **init_args, **preprocess_args)
        test_dataset = dataset_class(part="test", **init_args, spm=train_dataset.get_spm())

        return train_dataset, test_dataset

    def instantiate_model(self, name, marg):
        """Instantiate selected model.

        Args:
            name (string): Name of any model from `model` module.
            marg (dict): Model arguments.

        Returns:
            model.BaseSummarizationModel: Instantiated model.
        """
        model_class = getattr(models, name)
        instance = model_class.create(self.train_dataset, marg)
        return instance

    def instantiate_optim(self, name, oargs, lp):
        """Instantiate selected optimizer.

        Args:
            name (str): Optimizer class name from torch.optim
            oargs (dict): Optimizer arguments.
            lp (iterator): Learnable parameters.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """

        optim_class = getattr(torch.optim, name)
        instance = optim_class(lp, **oargs)
        return instance

    def run(self):
        # Create trainer engine
        trainer = self.model.create_trainer(self.optimizer, self.device)
        train_loader = DataLoader(self.train_dataset,
                                  self.targs.get("train_batch_size", 8),
                                  shuffle=True,
                                  collate_fn=self.train_dataset.collate_function)
        trainer_saver = ModelCheckpoint(self.dump_directory,
                                        filename_prefix="checkpoint",
                                        save_interval=self.targs.get("checkpoint_interval", 1000),
                                        n_saved=1,
                                        save_as_state_dict=True,
                                        create_dir=True,
                                        require_empty=False)
        best_model_saver = ModelCheckpoint(self.dump_directory,
                                           filename_prefix="best_model",
                                           score_name="rouge",
                                           score_function=lambda engine: engine.state.metrics["rouge"]["rouge-1"]["f"],
                                           n_saved=3,
                                           save_as_state_dict=True,
                                           create_dir=True,
                                           require_empty=False)
        to_save = {"model": self.model, "optimizer": self.optimizer}
        best_model_to_save = {"best_model": self.model}

        # Create evaluator engine
        evaluator = self.model.create_evaluator(self.device)
        val_loader = DataLoader(self.test_dataset,
                                self.targs.get("test_batch_size", 8),
                                shuffle=True,
                                collate_fn=self.test_dataset.collate_function,
                                drop_last=True)

        for name, metric in self.eval_metrics.items():
            metric.attach(evaluator, name)

        # Event handlers
        trainer.add_event_handler(Events.ITERATION_COMPLETED, trainer_saver, to_save)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.get_log_trainer(train_loader))
        trainer.add_event_handler(Events.EPOCH_STARTED, self.get_pbar_initializer(train_loader))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.get_pbar_destructor())
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.get_val_evaluator(evaluator, val_loader))
        evaluator.add_event_handler(Events.COMPLETED, best_model_saver, best_model_to_save)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, self.get_sample_evaluator())

        trainer.run(train_loader, self.targs.get("epochs", 5))
        self.pbar.close()

    def get_log_trainer(self, train_loader):
        def _log_trainer(engine: Engine):
            # TODO: Add supporting of multiple metrics.
            epoch = engine.state.epoch
            iteration = (engine.state.iteration - 1) % len(train_loader) + 1
            loss = engine.state.output

            if iteration % 10 == 0:
                self.pbar.desc = self.pbar_descr.format(epoch, loss)
                self.pbar.update(10)

            if iteration % self.targs.get("train_log_interval", 100) == 0:
                message = "Epoch[{}] | Iteration[{}/{}] | Loss: {:.4f}"
                message = message.format(engine.state.epoch, iteration, len(train_loader), loss)
                logging.info(message)
                self.tb_writer.add_scalar("training/loss", loss, engine.state.iteration)
                # TODO: use tqdm to log progress?
        return _log_trainer

    def get_sample_evaluator(self,):
        def _sample_evaluator(engine: Engine):
            iteration = engine.state.iteration
            if iteration == 1:
                generated, target = engine.state.output
                texts_str = self.train_dataset.decode(engine.state.batch["src"])
                generated_str = self.train_dataset.decode(generated)
                target_str = self.train_dataset.decode(target)
                for i, d in enumerate(zip(texts_str, generated_str, target_str)):
                    text, generated, original = d
                    logging.info("Evaluation sample[%d]: \nText: %s\nOriginal summary: %s\nGenerated summary: %s",
                                 i, text, original, generated)
        return _sample_evaluator

    def get_val_evaluator(self, evaluator, val_loader):
        def _val_evaluator(engine: Engine):
            # TODO: Add supporting of multiple metrics.
            epoch = engine.state.epoch
            logging.info("Start compute validation metrics.")
            metrics = evaluator.run(val_loader).metrics
            metrics = {m: metrics["rouge"][m]["f"] for m in ["rouge-1", "rouge-2", "rouge-l"]}
            message = "Validation result - Epoch[{}] Rouge-1-f: {:.4f} | Rouge-2-f: {:.4f} | Rouge-l-f: {:.4f}"
            message = message.format(epoch, *metrics.values())
            logging.info(message)
            self.tb_writer.add_scalars("evaluating/", metrics, epoch)
            # TODO: use tqdm to log progress?
        return _val_evaluator

    def get_pbar_initializer(self, loader):
        def _pbar_initializer(enigine: Engine):
            self.pbar.total = len(loader)
            self.pbar.unpause()
        return _pbar_initializer

    def get_pbar_destructor(self):
        def _pbar_destructor(engine: Engine):
            self.pbar.total = None
            self.pbar.n = self.pbar.last_print_n = 0
            self.pbar.desc = "Epoch[{}] | Evaluating".format(engine.state.epoch)
        return _pbar_destructor

    def get_rouge_process_function(self):
        def _rouge_process_function(output):
            generated, target = output
            generated_str = self.train_dataset.decode(generated)
            target_str = self.train_dataset.decode(target)
            return generated_str, target_str
        return _rouge_process_function
