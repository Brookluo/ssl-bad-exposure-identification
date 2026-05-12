"""Tests for decam_qa.dataset — DECamExposureDataset."""
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from pathlib import Path
from decam_qa.dataset import DECamExposureDataset


@pytest.fixture
def exposure_fits(tmp_path):
    """FITS file with 2 named CCD HDUs."""
    rng = np.random.default_rng(42)
    fpath = tmp_path / "exp.fits.fz"
    hdus = [fits.PrimaryHDU()]
    for name in ["N1", "N2"]:
        data = rng.normal(100, 5, (2046, 4094)).astype(np.float32)
        hdus.append(fits.ImageHDU(data, name=name))
    fits.HDUList(hdus).writeto(fpath)
    return fpath


@pytest.fixture
def exposure_csv(tmp_path, exposure_fits):
    """CSV with 2 CCD rows from same exposure."""
    df = pd.DataFrame({
        "image_filename": [str(exposure_fits), str(exposure_fits)],
        "expnum": [1, 1],
        "ccdnum": [32, 33],
        "image_hdu": [1, 2],
        "filter": [1, 1],
        "reasons": [1, 4],
        "vi_source": [0, 0],
    })
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    return path


class TestExposureDataset:
    def test_len_equals_exposure_count(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                    binsize=120)
        assert len(ds) == 1

    def test_getitem_returns_dict(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                    binsize=120)
        item = ds[0]
        assert isinstance(item, dict)
        assert "expnum" in item
        assert item["expnum"] == 1

    def test_label_is_exposure_reason_bitmask(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                    binsize=120)
        item = ds[0]
        assert item["reason_bitmask"] == 5

    def test_global_stamp_present(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                    binsize=120)
        item = ds[0]
        assert item["global_stamp"] is not None
        assert item["global_stamp"].ndim == 3
        assert item["global_stamp"].shape[0] == 1

    def test_selected_ccds_present(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                    binsize=120, top_k=8)
        item = ds[0]
        assert "selected_ccds" in item
        assert "selected_images" in item
        assert len(item["selected_ccds"]) == 2

    def test_two_exposures(self, tmp_path):
        """Two distinct exposures, each with one CCD."""
        rng = np.random.default_rng(42)
        dfs = []
        for eid, hdu_name in enumerate(["N1", "N2"], 1):
            fpath = tmp_path / f"exp{eid}.fits.fz"
            data = rng.normal(100, 5, (2046, 4094)).astype(np.float32)
            fits.HDUList([fits.PrimaryHDU(),
                          fits.ImageHDU(data, name=hdu_name)]).writeto(fpath)
            df = pd.DataFrame({
                "image_filename": [str(fpath)],
                "expnum": [eid],
                "ccdnum": [32],
                "image_hdu": [1],
                "filter": [1],
                "reasons": [0],
                "vi_source": [0],
            })
            dfs.append(df)
        csv_path = tmp_path / "two_exp.csv"
        pd.concat(dfs).to_csv(csv_path, index=False)

        ds = DECamExposureDataset(str(csv_path), str(tmp_path), binsize=120)
        assert len(ds) == 2
        e1 = ds[0]
        e2 = ds[1]
        assert e1["expnum"] == 1
        assert e2["expnum"] == 2

    def test_missing_ccd_does_not_crash(self, tmp_path):
        """CCD with bad HDU index should be skipped, not crash."""
        rng = np.random.default_rng(42)
        fpath = tmp_path / "exp.fits.fz"
        data = rng.normal(100, 5, (2046, 4094)).astype(np.float32)
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(data, name="N1")]).writeto(fpath)

        df = pd.DataFrame({
            "image_filename": [str(fpath), str(fpath)],
            "expnum": [1, 1],
            "ccdnum": [32, 99],
            "image_hdu": [1, 999],
            "filter": [1, 1],
            "reasons": [0, 0],
            "vi_source": [0, 0],
        })
        csv_path = tmp_path / "missing_ccd.csv"
        df.to_csv(csv_path, index=False)

        ds = DECamExposureDataset(str(csv_path), str(tmp_path), binsize=120)
        item = ds[0]
        assert item["num_readable_ccds"] == 1

    def test_completely_unreadable_exposure_raises(self, tmp_path):
        """All HDUs invalid -> raise error."""
        fpath = tmp_path / "bad.fits.fz"
        fits.HDUList([fits.PrimaryHDU()]).writeto(fpath)

        df = pd.DataFrame({
            "image_filename": [str(fpath)],
            "expnum": [1],
            "ccdnum": [32],
            "image_hdu": [999],
            "filter": [1],
            "reasons": [0],
            "vi_source": [0],
        })
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        ds = DECamExposureDataset(str(csv_path), str(tmp_path), binsize=120)
        with pytest.raises(RuntimeError, match="No readable CCDs"):
            ds[0]
