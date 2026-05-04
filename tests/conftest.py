from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def sample_csv(tmp_data_dir: Path) -> str:
    """Create a minimal sample CSV for testing CSV-dependent code."""
    import pandas as pd
    df = pd.DataFrame({
        "expnum": [123456, 123457, 123458, 123459, 123460],
        "image_filename": [
            "c4d_200101_000000_ooi_g_ls9.fits.fz",
            "c4d_200101_000001_ooi_r_ls9.fits.fz",
            "c4d_200101_000002_ooi_i_ls9.fits.fz",
            "c4d_200101_000003_ooi_z_ls9.fits.fz",
            "c4d_200101_000004_ooi_Y_ls9.fits.fz",
        ],
        "image_hdu": ["S1", "S2", "S3", "S4", "N1"],
        "reasons": [0, 1, 2, 4, 8],
        "vi_source": [0, 1, 2, 1, 0],
    })
    path = tmp_data_dir / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_npz_path(tmp_data_dir: Path) -> str:
    """Create a synthetic NPZ file that mimics read_image output."""
    half_imdata = np.zeros((4, 2046, 2046), dtype=np.float32)
    ccdnames = np.array(["S1", "S2"])
    path = tmp_data_dir / "sample.npz"
    np.savez(path, ccdnames=ccdnames, half_imdata=half_imdata)
    return str(path)


@pytest.fixture
def sample_h5_path(tmp_data_dir: Path) -> str:
    """Create a synthetic HDF5 file that mimics worker embed output."""
    import h5py
    path = str(tmp_data_dir / "0_worker_embeds.h5")
    with h5py.File(path, 'w') as f:
        grp = f.create_group("images")
        grp.create_dataset("idx_0_label_1", data=np.ones((1, 768), dtype='float'), dtype='float')
        grp.create_dataset("idx_5_label_2", data=np.ones((1, 768), dtype='float'), dtype='float')
    return str(tmp_data_dir)


@pytest.fixture
def sample_html_path(tmp_data_dir: Path) -> str:
    """Create a minimal table HTML file."""
    html = (
        '<table>\n'
        '<td>c4d_200101_000000_ooi_g_ls9.fits.fz<br>123456<br>other<br><td><img src="./images/c4d_200101_000000_ooi_g_ls9.jpg"></tr>\n'
        '<td>c4d_200101_000001_ooi_r_ls9.fits.fz<br>123457<br><td><img src="./images/c4d_200101_000001_ooi_r_ls9.jpg"></tr>\n'
        '</table>\n'
    )
    path = tmp_data_dir / "inspection.html"
    with open(path, 'w') as f:
        f.write(html)
    return str(path)
