import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    """Resolve project paths independently of the current working directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.
    """

    with resolve_path(file_path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data
