"""
config/settings.py

Loads and validates the BehaveGuard run configuration from YAML. Centralizing
config loading here (rather than scattering `open("config.yaml")` calls
across modules) means there's exactly one place that knows the config schema
and one place that fails fast if it's malformed.
"""

from pathlib import Path

import yaml

from utils.logger import get_logger
from utils.exceptions import ConfigError

logger = get_logger(__name__)

REQUIRED_TOP_LEVEL_KEYS = [
    "data", "preprocessing", "behavioral", "tfidf", "embeddings",
    "model", "training", "output",
]


def load_config(config_path: str) -> dict:
    """
    Load and validate the YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Parsed configuration dict.

    Raises:
        ConfigError: If the file is missing, malformed, or missing required
            top-level sections.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config at {config_path}: {e}") from e

    if not isinstance(cfg, dict):
        raise ConfigError(f"Config at {config_path} did not parse to a dictionary")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"Config at {config_path} is missing required sections: {missing}")

    logger.info(f"Loaded config from {config_path}")
    return cfg


def get_output_path(cfg: dict, key: str) -> str:
    """
    Convenience accessor for an output path, with a clear error if missing.

    Args:
        cfg: Full configuration dict.
        key: Key within cfg['output'] to retrieve (e.g. 'model_path').

    Returns:
        The configured path string.

    Raises:
        ConfigError: If cfg['output'][key] is not set.
    """
    try:
        return cfg["output"][key]
    except KeyError as e:
        raise ConfigError(f"Missing required output path config: output.{key}") from e
