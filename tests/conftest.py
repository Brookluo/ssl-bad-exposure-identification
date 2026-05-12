"""Shared pytest fixtures for decam_qa tests."""
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import pytest
import h5py
import torch
import torch.nn as nn
from astropy.io import fits


@pytest.fixture
def synthetic_image():
    """A (1, 2048, 4096) float32 array simulating a DECam CCD image."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (1, 2048, 4096)).astype(np.float32)


@pytest.fixture
def tmp_fits_image(tmp_path, synthetic_image):
    """Create a temporary FITS file from a synthetic image."""
    fpath = tmp_path / "test_image.fits.fz"
    hdu_primary = fits.PrimaryHDU(synthetic_image[0])
    hdu1 = fits.ImageHDU(synthetic_image[0])
    hdu2 = fits.ImageHDU(synthetic_image[0])
    hdul = fits.HDUList([hdu_primary, hdu1, hdu2])
    hdul.writeto(fpath)
    return fpath


@pytest.fixture
def sample_csv(tmp_path, tmp_fits_image):
    """A small pd.DataFrame with columns matching DECamImageDataset expectations."""
    import numpy as np
    rng = np.random.default_rng(42)

    expnum = np.array([1001, 1001, 1002, 1002])
    ccdnum = np.array([25, 26, 27, 28])
    image_hdu = np.array([1, 2, 1, 2])
    filter_col = np.array([1, 1, 2, 2])
    image_filename = [str(tmp_fits_image)] * 4
    reasons = np.array([0, 1, 4, 5], dtype=int)
    vi_source = np.array([0, 1, 2, 3], dtype=int)

    df = pd.DataFrame({
        "image_filename": image_filename,
        "expnum": expnum,
        "ccdnum": ccdnum,
        "image_hdu": image_hdu,
        "filter": filter_col,
        "reasons": reasons,
        "vi_source": vi_source,
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def small_embeddings():
    """(100, 768) synthetic embeddings with known 3-class structure."""
    rng = np.random.default_rng(42)
    n_per_class = 33
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X2d = rng.normal(0, 0.5, (n_per_class * 3, 2))
    for i in range(3):
        X2d[i * n_per_class:(i + 1) * n_per_class] += centers[i]
    random_proj = rng.normal(0, 1, (2, 768))
    return X2d @ random_proj + rng.normal(0, 0.1, (n_per_class * 3, 768))


@pytest.fixture
def small_labels():
    """100 labels with 3 classes for classifier testing."""
    n_per_class = 33
    labels = np.array([1] * n_per_class + [2] * n_per_class + [4] * n_per_class)
    return labels.astype(int)


@pytest.fixture
def temp_h5_dir(tmp_path):
    """Temporary directory with mock HDF5 embedding files (4 workers)."""
    rng = np.random.default_rng(42)
    embeds_dir = tmp_path / "embeds_out"
    embeds_dir.mkdir()
    total = 20
    for w in range(4):
        fpath = embeds_dir / f"{w}_worker_embeds.h5"
        with h5py.File(fpath, 'w') as h5f:
            grp = h5f.create_group("images")
            for i in range(5):
                key = f"idx_{w * 5 + i}_label_{(i % 3) + 1}"
                grp.create_dataset(key, data=rng.normal(0, 1, (1, 768)).astype(np.float32))
    return embeds_dir


@pytest.fixture
def mock_model(mocker):
    """Mocked DINOv2 model returning fixed 768-dim embeddings."""
    mock = mocker.patch("torch.hub.load")
    model = mocker.MagicMock()
    model.forward.return_value = mocker.MagicMock()
    model.forward.return_value.numpy.return_value = np.zeros((1, 768), dtype=np.float32)
    model.eval.return_value = model
    mock.return_value = model
    return model


@pytest.fixture
def fake_dino_model():
    """A minimal mock DINOv2 model with a real nn.Conv2d patch_embed.proj.

    Returns a real nn.Sequential as patch_embed so tests can verify
    weight manipulation. Does NOT mock torch.hub.load — use mock_torch_hub
    for tests that need to intercept model loading.
    """
    class FakePatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14, bias=True)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = FakePatchEmbed()
            self.embed_dim = 768

    return FakeModel()
