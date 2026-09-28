import yaml


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.
    """

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    return data