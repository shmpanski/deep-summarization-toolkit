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
        "config_file", nargs=1, type=str, help="sampler configuration file"
    )
    parser.add_argument(
        "model_state", nargs=1, type=str, help="file, containing model state"
    )
    parser.add_argument("input", nargs=1, type=str, help="input file with texts")
    parser.add_argument("output", nargs=1, type=str, help="output file")
    args = parser.parse_args()

    config_filename = args.config_file[0]
    model_state = args.model_state[0]
    input_filename = args.input[0]
    output_filename = args.output[0]
    with open(config_filename, "r") as config_file:
        logger.info("Loaded sampling configurations from %s", config_file.name)
        config = yaml.load(config_file)
        sampler = train.load_pipeline(config, "sample", dump_file=model_state)

    with open(input_filename, "r") as input_file, open(
        output_filename, "w"
    ) as output_file:
        for text in input_file:
            summary = sampler.sample(text.lower())
            output_file.write(summary[0] + os.linesep)
