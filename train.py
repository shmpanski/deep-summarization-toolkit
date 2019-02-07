import argparse
import logging
import os

import yaml

from dst import Trainer


def setup_logging(default_path='logging.yml', default_level=logging.INFO, env_key='LOG_CFG'):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs=1, type=str, help="runinig configuration file")
    args = parser.parse_args()

    config_filename = args.config_file[0]
    with open(config_filename, "r") as config_file:
        logging.info("Loaded configurations from %s", config_file.name)
        trainer = Trainer(config_file)
        trainer.run()
