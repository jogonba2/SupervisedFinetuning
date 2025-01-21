import json
from pathlib import Path
from uuid import uuid4

import _jsonnet


def read_json(path: str | Path) -> dict:
    """
    Reads a json file.

    Args:
        path (str | Path): path to the json file

    Returns:
        dict: loaded json
    """
    with open(path, "r") as fr:
        return json.load(fr)


def save_json(path: str | Path, json_dict: dict) -> None:
    """
    Saves a dict into a json file.

    Args:
        path (str | Path): path to the json file.
        json_dict (dict): dictionary to be saved.
    """
    with open(path, "w") as fw:
        json.dump(json_dict, fw, indent=4)


def load_configs(path: str | Path) -> list[dict]:
    """
    Load configs from disk.

    Args:
        path (str | Path): path to the config file (json or jsonnet)

    Returns:
        list[dict]: a list containing just one config if the provided path
                    is a json file or a list of configs if a jsonnet file.
    """
    path = Path(path)
    if path.suffix == ".json":
        return [read_json(path)]
    elif path.suffix == ".jsonnet":
        json_str = _jsonnet.evaluate_file(path.as_posix())
        return list(json.loads(json_str)["configs"].values())
    else:
        raise ValueError("Only .json and .jsonnet files are allowed")


def get_results_path(base_path: str | Path, extension: str = ".json") -> Path:
    """
    Appends to a base path a random file name.

    Args:
        base_path (str | Path): the base path of the file
        extension (str): extension of the file

    Returns:
        Path: path to the file
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / f"{uuid4()}{extension}"
