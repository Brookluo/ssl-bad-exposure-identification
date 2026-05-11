"""Tests for decam_qa.config — YAML config loading, validation, defaults."""
import pytest
from pathlib import Path
from decam_qa.config import load_config, get_default_config


def _write_yaml(path, content):
    path.write_text(content)


@pytest.fixture
def valid_embed_yaml(tmp_path):
    p = tmp_path / "embed.yaml"
    _write_yaml(p, """\
model:
  size: base
  use_register: true
data:
  batch_size: 4
  crop_size: [2352, 1176]
  num_workers: 10
  pin_memory: false
image_roots:
  dr10: /data/images/dr10
  dr11: /data/images/dr11
distributed:
  backend: nccl
  gpu_bind: none
scratch_dir: /scratch
""")
    return str(p)


@pytest.fixture
def valid_train_yaml(tmp_path):
    p = tmp_path / "train.yaml"
    _write_yaml(p, """\
embeddings:
  train_dir: /data/train
  test_dir: /data/test
pipeline:
  scalers: [StandardScaler, MinMaxScaler]
  pca_components: [5, 10, 15]
  variance_threshold: [0, 0.001]
  knn:
    n_neighbors: [1, 3, 5]
    p: [1, 2]
    leaf_size: [1, 5]
cv:
  n_splits: 3
  test_size: 0.3
  random_state: 42
model_path: /output/model.pkl
""")
    return str(p)


@pytest.fixture
def valid_inference_yaml(tmp_path):
    p = tmp_path / "inference.yaml"
    _write_yaml(p, """\
pipeline_path: /data/model.pkl
embeddings_dir: /data/embeds
dataset_csv: /data/sample.csv
probability_threshold: 0.7
top_n_per_class: 200
metadata:
  dr10_tab: /data/meta.fits.gz
  ccd_cuts_enabled: true
results_dir: /output/results
plot_dir: /output/plots
""")
    return str(p)


class TestLoadValidConfigs:
    def test_load_valid_embed_config(self, valid_embed_yaml):
        cfg = load_config(valid_embed_yaml, "embed")
        assert cfg["model"]["size"] == "base"
        assert cfg["data"]["crop_size"] == [2352, 1176]
        assert cfg["image_roots"]["dr10"] == "/data/images/dr10"

    def test_load_valid_train_config(self, valid_train_yaml):
        cfg = load_config(valid_train_yaml, "train")
        assert isinstance(cfg["pipeline"]["pca_components"], list)
        assert cfg["cv"]["random_state"] == 42

    def test_load_valid_inference_config(self, valid_inference_yaml):
        cfg = load_config(valid_inference_yaml, "inference")
        assert cfg["probability_threshold"] == 0.7
        assert cfg["metadata"]["ccd_cuts_enabled"] is True


class TestValidation:
    def test_load_missing_required_key(self, tmp_path):
        p = tmp_path / "bad.yaml"
        _write_yaml(p, """\
image_roots:
  dr10: /data
""")
        with pytest.raises(ValueError, match="model"):
            load_config(str(p), "embed")

    def test_load_invalid_stage_name(self, valid_embed_yaml):
        with pytest.raises(ValueError, match="nonexistent"):
            load_config(valid_embed_yaml, "nonexistent")

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml", "embed")


class TestDefaults:
    def test_get_default_config_embed(self):
        cfg = get_default_config("embed")
        assert "model" in cfg
        assert "data" in cfg
        assert "image_roots" in cfg

    def test_get_default_config_train(self):
        cfg = get_default_config("train")
        assert "pipeline" in cfg

    def test_get_default_config_inference(self):
        cfg = get_default_config("inference")
        assert "probability_threshold" in cfg

    def test_embed_config_defaults_fill_missing(self, tmp_path):
        p = tmp_path / "minimal.yaml"
        _write_yaml(p, """\
model:
  size: base
data:
  batch_size: 2
  crop_size: [1000, 500]
  num_workers: 4
image_roots:
  dr10: /data
""")
        cfg = load_config(str(p), "embed")
        assert "pin_memory" in cfg["data"]


class TestFieldTypes:
    def test_embed_crop_size_is_ints(self, valid_embed_yaml):
        cfg = load_config(valid_embed_yaml, "embed")
        cs = cfg["data"]["crop_size"]
        assert isinstance(cs, list)
        assert all(isinstance(x, int) for x in cs)

    def test_train_pca_components_is_ints(self, valid_train_yaml):
        cfg = load_config(valid_train_yaml, "train")
        pca = cfg["pipeline"]["pca_components"]
        assert all(isinstance(x, int) for x in pca)

    def test_inference_threshold_is_float(self, valid_inference_yaml):
        cfg = load_config(valid_inference_yaml, "inference")
        assert isinstance(cfg["probability_threshold"], float)
