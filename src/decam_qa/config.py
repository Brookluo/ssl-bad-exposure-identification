"""Configuration loading with YAML support and per-stage defaults."""
from pathlib import Path
from typing import Any, Dict
import yaml


_REQUIRED_KEYS = {
    "embed": ("model", "data", "image_roots"),
    "train": ("embeddings", "pipeline", "cv", "model_path"),
    "inference": ("pipeline_path", "embeddings_dir", "dataset_csv"),
}

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "embed": {
        "model": {"size": "base", "use_register": True},
        "data": {
            "batch_size": 1, "crop_size": [2352, 1176],
            "num_workers": 10, "pin_memory": False,
        },
        "distributed": {"backend": "nccl", "gpu_bind": "none"},
        "scratch_dir": "",
    },
    "train": {
        "cv": {"n_splits": 3, "test_size": 0.3, "random_state": 42},
    },
    "inference": {
        "probability_threshold": 0.7,
        "top_n_per_class": 200,
        "metadata": {"ccd_cuts_enabled": False},
    },
}


def _deep_merge(defaults: Dict, cfg: Dict) -> None:
    """Merge default values into cfg in-place, recursing into nested dicts."""
    for key, def_val in defaults.items():
        if key not in cfg:
            cfg[key] = def_val
        elif isinstance(def_val, dict) and isinstance(cfg.get(key), dict):
            _deep_merge(def_val, cfg[key])


def load_config(config_path: str, stage: str) -> Dict[str, Any]:
    """Load and validate a stage-specific YAML config.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    stage : str
        One of 'embed', 'train', 'inference'.

    Returns
    -------
    Dict[str, Any]
        Validated config dictionary with defaults filled.

    Raises
    ------
    FileNotFoundError
        If config_path does not exist.
    ValueError
        If stage is unknown or required keys are missing.
    """
    if stage not in _REQUIRED_KEYS:
        raise ValueError(
            f"Unknown stage '{stage}'. Must be one of: {list(_REQUIRED_KEYS.keys())}"
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for key in _REQUIRED_KEYS[stage]:
        if key not in cfg:
            raise ValueError(
                f"Missing required key '{key}' in config file: {config_path}"
            )

    defaults = _DEFAULTS.get(stage, {})
    _deep_merge(defaults, cfg)

    return cfg


def get_default_config(stage: str) -> Dict[str, Any]:
    """Return the default configuration for a given stage.

    Parameters
    ----------
    stage : str
        One of 'embed', 'train', 'inference'.

    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary.

    Raises
    ------
    ValueError
        If stage is unknown.
    """
    if stage not in _REQUIRED_KEYS:
        raise ValueError(
            f"Unknown stage '{stage}'. Must be one of: {list(_REQUIRED_KEYS.keys())}"
        )
    cfg = dict(_DEFAULTS.get(stage, {}))
    for key in _REQUIRED_KEYS[stage]:
        if key not in cfg:
            cfg[key] = {}
    return cfg
