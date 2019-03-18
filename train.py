import argparse
import logging
import os

import yaml

from dst import train
from dst.utils import setup_logging


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", nargs=1, type=str, help="runinig configuration file"
    )
    args = parser.parse_args()

    config_filename = args.config_file[0]
    with open(config_filename, "r") as config_file:
        logging.info("Loaded configurations from %s", config_file.name)
        config = yaml.load(config_file)
        trainer = train.load_trainer(config)
        trainer.run()
