from typing import Dict
from pathlib import Path

import json


def load_config(config_fpath: str) -> Dict:
    """
    Loads configuration from a JSON file.

    Args:
        config_fpath (str): Path to the JSON configuration file.

    Returns:
        Dict: Configuration parameters.
    """
    with open(config_fpath, "r") as file:
        config = json.load(file)

    # Extracts config name
    config_name = get_experiment_name(config_fpath)

    # Extend config file with known relations and deterministic values
    config = extend_config(config, config_name)

    return config


def extend_config(config: Dict, config_name: str) -> Dict:
    """
    Extends the given configuration dictionary by adding new keys based on
    deterministic calculations and relations from existing fields.

    Args:
        config (Dict): The original configuration dictionary.
        config_name (str): Name of the config file.

    Returns:
        Dict: The extended configuration dictionary with new keys and values.

    Raises:
        KeyError: If any expected key is missing from the config.
        ValueError: If a field value is invalid for calculations.
    """

    # Check whether all the required fields are present
    check_required_keys(config)

    # Add new data parameters to config
    config["data"]["image_size"] = config["model"]["image_size"]

    # Define new model parameters
    read_depth = config["data"].get("read_depth", False)
    estimate_depth = config["data"].get("estimate_depth", False)
    window_size = config["data"]["window_size"]
    angles_format = config["data"].get("rotation_format", "euler")
    num_channels = 3  # 4 if (read_depth or estimate_depth) else 3

    if angles_format == "euler":
        num_classes = 6 * (window_size - 1)
    elif angles_format == "quaternion":
        num_classes = 7 * (window_size - 1)

    # Add new model parameters to config
    config["model"]["num_channels"] = num_channels
    config["model"]["num_frames"] = window_size
    config["model"]["num_classes"] = num_classes

    # Model size (DEIT tiny, small or base)
    config = set_model_size(config)

    # Update checkpoint directory path with exp name
    config["checkpoint"]["checkpoint_dpath"] = (
        Path(config["checkpoint"]["checkpoint_dpath"]) / config_name
    )

    return config


def check_required_keys(config: Dict) -> None:
    """
    Checks if all the required keys, including nested keys, are present in the configuration dictionary.

    Args:
        config (Dict): The configuration dictionary.

    Raises:
        KeyError: If any required key is missing from the config.
    """
    required_keys = [
        "data.window_size",
        "model.DEIT_size",
        "model.image_size",
        "checkpoint.checkpoint_dpath",
    ]
    for key in required_keys:
        keys = key.split(".")  # Split the key by '.' to access nested keys
        current_dict = config
        for sub_key in keys:
            if sub_key not in current_dict:
                raise KeyError(f"Key '{key}' is required in the configuration file.")
            current_dict = current_dict[sub_key]  # Drill down to the next level


def set_model_size(config: Dict) -> None:
    """
    Sets the model size parameters (patch_size, embed_dim, depth, num_heads)
    based on the configuration for the DEIT_size.

    Args:
        config (Dict): The configuration dictionary that contains the model size info.
    """
    # Initialize model size parameters (only if 'tiny', 'small' or 'base')
    if config["model"]["DEIT_size"] == "tiny":
        config["model"]["patch_size"] = 16
        config["model"]["embed_dim"] = 192
        config["model"]["depth"] = 12
        config["model"]["num_heads"] = 3

    elif config["model"]["DEIT_size"] == "small":
        config["model"]["patch_size"] = 16
        config["model"]["embed_dim"] = 384
        config["model"]["depth"] = 12
        config["model"]["num_heads"] = 6

    elif config["model"]["DEIT_size"] == "base":
        config["model"]["patch_size"] = 16
        config["model"]["embed_dim"] = 768
        config["model"]["depth"] = 12
        config["model"]["num_heads"] = 12

    return config


def get_experiment_name(config_fpath: str) -> str:
    """
    Extracts the experiment name from a given configuration file path.

    The experiment name is defined as the part of the file name without the
    extension. For example, if the input is "configs/exp1.json", the function
    will return "exp1".

    Args:
        config_fpath (str): The path to the configuration file. It is expected
                            to be a string in the format 'path/exp_name.extension'.

    Returns:
        str: The experiment name extracted from the file path.
    """
    # Get config file name
    config_fname = config_fpath.split("/")[-1]

    # Remove .json extension
    exp_name = config_fname.split(".")[0]

    return exp_name


if __name__ == "__main__":
    from pprint import pprint as pp

    config_fpath = "configs/exp1.json"
    config = load_config(config_fpath)
    pp(config)
