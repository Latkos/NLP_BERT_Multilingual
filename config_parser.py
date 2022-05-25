import yaml


def get_config(config_path):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)
