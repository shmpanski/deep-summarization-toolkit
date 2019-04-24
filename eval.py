import argparse
import logging
import os

import yaml

from dst import train
from dst.utils import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", nargs=1, type=str, help="evaluation configuration file"
    )
    parser.add_argument(
        "model_state", nargs=1, type=str, help="file, containing model state"
    )
    args = parser.parse_args()

    config_filename = args.config_file[0]
    model_state = args.model_state[0]
    with open(config_filename, "r") as config_file:
        logger.info("Loaded configurations from %s", config_file.name)
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        evaluator = train.load_pipeline(config, "eval", dump_file=model_state)
        evaluator.evaluate()
