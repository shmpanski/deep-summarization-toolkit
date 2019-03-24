import json
import logging
import os

import jsonschema
import tqdm
import yaml


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def setup_logging(
    default_path="logging.yml", default_level=logging.INFO, env_key="LOG_CFG"
):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def fill_dict_default_values(source: dict, default_dict: dict):
    """Fill dictionary with default values.

    Args:
        source (dict): Source dictionary.
        default_dict (dict): Default dictionary values.
    """

    for key in default_dict:
        if isinstance(default_dict[key], dict):
            fill_dict_default_values(source.setdefault(key, {}), default_dict[key])
        else:
            source.setdefault(key, default_dict[key])


def validate_yaml(yaml_data: dict, schema_file: str):
    """Validate yaml data with json schema file.

    Args:
        yaml_data (dict): YAML data.
        schema_file (str): Schema file name.
    """

    with open(schema_file) as schema_file:
        schema = json.load(schema_file)
        jsonschema.validate(yaml_data, schema)
