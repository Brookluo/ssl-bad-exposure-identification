"""Config dataclasses for the embedding pipeline.

Loading order (last wins): YAML file < env vars < direct instantiation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Config for the embedding generation pipeline (master_eval + parallel_eval)."""
    dset_path: Path | None = None
    output_dir: Path | None = None
    image_dir: Path | None = None
    dr: str = "dr11"
    cont: bool = False
    model_size: str = "base"
    batch_size: int = 1
    crop_size: tuple[int, int] = (2352, 1176)
    num_workers: int = 2


@dataclass
class VisualizationConfig:
    """Config for visualization paths (decam_postage_stamps)."""
    image_dir: str = ""
    blob_dir: str = ""
    surveyccd_path: str = ""
    surveyccd_path_dr8: str = ""


def _env_or_none(key: str) -> str | None:
    return os.environ.get(key) or None


def load_config(yaml_path: str | Path | None = None) -> tuple[PipelineConfig, VisualizationConfig]:
    """Load config from YAML, then overlay with env vars.

    Pipeline env vars:  DECAM_IMAGE_DIR (image_dir)
    Visualization env:  DECAM_IMAGE_DIR, DECAM_BLOB_DIR, SURVEYCCD_PATH, SURVEYCCD_PATH_DR8
    """
    pipe = PipelineConfig()
    viz = VisualizationConfig()

    # Layer 1: YAML file
    if yaml_path and Path(yaml_path).exists():
        try:
            import yaml as _yaml
        except ImportError:
            import json as _yaml
            _yaml.safe_load = _yaml.load
        try:
            with open(yaml_path) as f:
                raw: dict[str, Any] = _yaml.safe_load(f)
            if raw:
                raw_pipe = raw.get("pipeline", {})
                raw_viz = raw.get("visualization", {})
                for k in ("dset_path", "output_dir", "image_dir", "dr", "cont", "model_size", "batch_size", "num_workers"):
                    if k in raw_pipe and raw_pipe[k] is not None:
                        setattr(pipe, k, raw_pipe[k])
                for k in ("image_dir", "blob_dir", "surveyccd_path", "surveyccd_path_dr8"):
                    if k in raw_viz and raw_viz[k] is not None:
                        setattr(viz, k, raw_viz[k])
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Failed to load config from %s", yaml_path)

    # Layer 2: Environment variables (override YAML)
    if _env_or_none("DECAM_IMAGE_DIR"):
        pipe.image_dir = Path(os.environ["DECAM_IMAGE_DIR"])
        viz.image_dir = os.environ["DECAM_IMAGE_DIR"]
    if _env_or_none("DECAM_BLOB_DIR"):
        viz.blob_dir = os.environ["DECAM_BLOB_DIR"]
    if _env_or_none("SURVEYCCD_PATH"):
        viz.surveyccd_path = os.environ["SURVEYCCD_PATH"]
    if _env_or_none("SURVEYCCD_PATH_DR8"):
        viz.surveyccd_path_dr8 = os.environ["SURVEYCCD_PATH_DR8"]

    return pipe, viz
