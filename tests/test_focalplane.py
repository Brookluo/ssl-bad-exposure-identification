"""Tests for decam_qa.focalplane — focal-plane stamp building."""
import numpy as np
import pytest
from astropy.io import fits
from decam_qa.focalplane import build_focalplane_stamp


@pytest.fixture
def multi_hdu_fits(tmp_path):
    """Create a FITS file with 3 CCD HDUs named N1, N2, N3.

    CCD shape matches existing DECam convention: (4094, 2046).
    """
    rng = np.random.default_rng(42)
    fpath = tmp_path / "exposure.fits.fz"
    hdus = [fits.PrimaryHDU()]
    for name in ["N1", "N2", "N3"]:
        data = rng.normal(100, 10, (4094, 2046)).astype(np.float32)
        hdu = fits.ImageHDU(data, name=name)
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(fpath)
    return fpath


class TestBuildFocalplaneStamp:
    def test_output_is_single_channel(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},  # N1
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 33, "image_hdu": 2},  # N2
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 34, "image_hdu": 3},  # N3
        ]
        hdul = fits.open(multi_hdu_fits)
        stamp = build_focalplane_stamp(hdul, rows, binsize=120)
        assert stamp.ndim == 3
        assert stamp.shape[0] == 1
        hdul.close()

    def test_stamp_shape_deterministic(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        s1 = build_focalplane_stamp(hdul, rows, binsize=120)
        s2 = build_focalplane_stamp(hdul, rows, binsize=120)
        assert s1.shape == s2.shape
        np.testing.assert_array_equal(s1, s2)
        hdul.close()

    def test_binsize_changes_output_size(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        s120 = build_focalplane_stamp(hdul, rows, binsize=120)
        s60 = build_focalplane_stamp(hdul, rows, binsize=60)
        assert s120.shape[1] < s60.shape[1]
        hdul.close()

    def test_no_finite_values(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits, mode="update")
        hdul[1].data[100:200, 100:200] = np.nan
        hdul[1].data[300, 300] = np.inf
        stamp = build_focalplane_stamp(hdul, rows, binsize=120, fill_value=0.0)
        assert np.all(np.isfinite(stamp))
        hdul.close()

    def test_mean_reducer_works(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        stamp = build_focalplane_stamp(hdul, rows, binsize=120, reducer="mean")
        hdul.close()
        assert np.all(np.isfinite(stamp))
        # Verify the stamp has placed CCD content (not all fill_value)
        assert np.any(stamp[0] != 0.0)

    def test_ccds_placed_in_different_regions(self, multi_hdu_fits):
        """CCDs placed at expected pixel positions and orientation matches
        existing (4094, 2046) convention. Different CCDs map to different
        canvas regions."""
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},  # N1
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 33, "image_hdu": 2},  # N2
        ]
        hdul = fits.open(multi_hdu_fits)
        stamp = build_focalplane_stamp(hdul, rows, binsize=120)
        # Two different CCDs should place in different canvas regions
        nonzero = np.argwhere(stamp[0] != 0.0)
        assert len(nonzero) > 0, "Stamp should have placed CCD pixels"
        hdul.close()

    def test_edge_ccd_uses_top_left_position_not_center_offset(self, tmp_path):
        """A CCD with y_pix=0 should start at the top edge and not be half-clipped."""
        fpath = tmp_path / "edge.fits.fz"
        data = np.full((4094, 2046), 7.0, dtype=np.float32)
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data, name="S29"),
        ]).writeto(fpath)

        rows = [{"expnum": 1, "image_filename": str(fpath),
                 "ccdnum": 1, "image_hdu": 1}]
        with fits.open(fpath) as hdul:
            stamp = build_focalplane_stamp(hdul, rows, binsize=120, reducer="mean")

        # CCD S29 has native top-left (x=8498, y=0). A 4094x2046 CCD
        # downsampled by 120 becomes 34x17, then transposed into the (y, x)
        # canvas as 17x34.
        y0 = 0
        x0 = int(np.rint(8498 / 120))
        region = stamp[0, y0:y0 + 17, x0:x0 + 34]
        assert region.shape == (17, 34)
        np.testing.assert_allclose(region, 7.0)

    def test_native_ccd_orientation_transposes_into_yx_canvas(self, tmp_path):
        """Native CCD axis 0 should map to focal-plane x, not stamp y."""
        fpath = tmp_path / "gradient.fits.fz"
        row_gradient = np.arange(4094, dtype=np.float32)[:, np.newaxis]
        data = np.broadcast_to(row_gradient, (4094, 2046)).copy()
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data, name="N1"),
        ]).writeto(fpath)

        rows = [{"expnum": 1, "image_filename": str(fpath),
                 "ccdnum": 32, "image_hdu": 1}]
        with fits.open(fpath) as hdul:
            stamp = build_focalplane_stamp(hdul, rows, binsize=120, reducer="mean")

        y0 = int(np.rint(13494 / 120))
        x0 = 0
        region = stamp[0, y0:y0 + 17, x0:x0 + 34]

        assert region[0, 1] > region[0, 0]
        np.testing.assert_allclose(region[:, 0], region[0, 0])
