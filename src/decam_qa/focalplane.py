"""DECam focal-plane stamp builder for multi-scale exposure analysis.

Builds low-resolution single-channel stamps from exposure HDUs by downsampling
CCDs and placing them in focal-plane layout.
"""
import numpy as np
from .info import ccdname2num


# Focal-plane layout: x_pix and y_pix are CCD top-left positions in the native
# pixel grid (29590 x 26787). Indices are ordered by ccdnum_list:
# [1,2,3,...,60,62] (missing 61).
_CCD_NUM_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62]

_X_PIX = np.array([8498., 12747., 16996., 6373.5, 10622.5, 14871.5, 19120.5,
                   4249., 8498., 12747., 16996., 21245., 2124.5, 6373.5,
                   10622.5, 14871.5, 19120.5, 23369.5, 2124.5, 6373.5,
                   10622.5, 14871.5, 19120.5, 23369.5, 0., 4249., 8498.,
                   12747., 16996., 21245., 25494., 0., 4249., 8498., 12747.,
                   16996., 21245., 25494., 2124.5, 6373.5, 10622.5, 14871.5,
                   19120.5, 23369.5, 2124.5, 6373.5, 10622.5, 14871.5,
                   19120.5, 23369.5, 4249., 8498., 12747., 16996., 21245.,
                   6373.5, 10622.5, 14871.5, 19120.5, 8498., 16996.])

_Y_PIX = np.array([0, 0, 0, 2249, 2249, 2249, 2249, 4498, 4498, 4498, 4498,
                   4498, 6747, 6747, 6747, 6747, 6747, 6747, 8996, 8996,
                   8996, 8996, 8996, 8996, 11245, 11245, 11245, 11245,
                   11245, 11245, 11245, 13494, 13494, 13494, 13494, 13494,
                   13494, 13494, 15743, 15743, 15743, 15743, 15743, 15743,
                   17992, 17992, 17992, 17992, 17992, 17992, 20241, 20241,
                   20241, 20241, 20241, 22490, 22490, 22490, 22490, 24739,
                   24739.])

_NATIVE_X = 29590
_NATIVE_Y = 26787
# Pre-build mapping: ccdnum -> focal-plane row index
_CCDNUM_TO_IDX = {num: i for i, num in enumerate(_CCD_NUM_LIST)}


def build_focalplane_stamp(hdul, exposure_rows, binsize=120,
                           subtract_median_sky=False, reducer="median",
                           fill_value=0.0):
    """Build a low-resolution focal-plane stamp from exposure HDUs.

    For each CCD referenced in exposure_rows: read from the open HDUList,
    subtract sky, trim to a multiple of binsize, downsample, and place
    into a focal-plane canvas.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS HDUList for the exposure (caller manages open/close).
    exposure_rows : list of dict or pd.DataFrame
        Must have keys: ccdnum or ccdname, and image_hdu.
    binsize : int
        Downsampling factor. A native 4094x2046 CCD becomes roughly
        4094/binsize by 2046/binsize before placement.
    subtract_median_sky : bool
        If True, subtract median sky from each CCD before downsampling.
    reducer : str
        'median' or 'mean'. Downsampling reducer.
    fill_value : float
        Value for empty regions and non-finite pixels.

    Returns
    -------
    np.ndarray
        Single-channel stamp with shape (1, out_h, out_w).
    """
    stamp_h = int(np.ceil(_NATIVE_Y / binsize))
    stamp_w = int(np.ceil(_NATIVE_X / binsize))
    stamp = np.full((stamp_h, stamp_w), fill_value, dtype=np.float32)

    # Normalize input: DataFrame -> list of dicts
    if hasattr(exposure_rows, "columns"):  # pd.DataFrame
        exposure_rows = exposure_rows.to_dict("records")

    for row in exposure_rows:
        if "ccdnum" in row:
            ccdnum = row["ccdnum"]
        elif "ccdname" in row:
            ccdname = row["ccdname"].decode() if isinstance(row["ccdname"], bytes) else row["ccdname"]
            ccdnum = ccdname2num[ccdname]
        else:
            raise ValueError("Each row must have 'ccdnum' or 'ccdname' key")
        hdu_idx = row["image_hdu"]

        if ccdnum not in _CCDNUM_TO_IDX:
            continue

        try:
            img = np.asarray(hdul[hdu_idx].data, dtype=np.float32)
        except (IndexError, KeyError, OSError):
            continue

        if subtract_median_sky:
            finite = img[np.isfinite(img)]
            if len(finite) > 0:
                img = img - np.median(finite)

        # Trim to multiple of binsize
        trim_h = (img.shape[0] // binsize) * binsize
        trim_w = (img.shape[1] // binsize) * binsize
        img = img[:trim_h, :trim_w]

        # Downsample via reshaping
        if reducer == "median":
            reshaped = img.reshape(trim_h // binsize, binsize,
                                   trim_w // binsize, binsize)
            downsampled = np.median(reshaped, axis=(1, 3))
        else:  # mean
            reshaped = img.reshape(trim_h // binsize, binsize,
                                   trim_w // binsize, binsize)
            downsampled = np.mean(reshaped, axis=(1, 3))

        # Place into a (y, x) canvas. Native DECam CCD arrays are indexed as
        # (x, y), so transpose after downsampling, matching decam_postage_stamps.
        stamp_ccd = downsampled.T
        fp_idx = _CCDNUM_TO_IDX[ccdnum]
        x0 = int(np.rint(_X_PIX[fp_idx] / binsize))
        y0 = int(np.rint(_Y_PIX[fp_idx] / binsize))
        cc_h = stamp_ccd.shape[0]
        cc_w = stamp_ccd.shape[1]

        # Clamp to canvas boundaries (handle both negative and overflow)
        src_y0 = max(0, -y0)
        src_x0 = max(0, -x0)
        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(y0 + cc_h, stamp_h)
        dst_x1 = min(x0 + cc_w, stamp_w)
        copy_h = dst_y1 - dst_y0
        copy_w = dst_x1 - dst_x0

        if copy_h > 0 and copy_w > 0:
            stamp[dst_y0:dst_y1, dst_x0:dst_x1] = stamp_ccd[
                src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]

    # Replace non-finite values
    stamp[~np.isfinite(stamp)] = fill_value
    # to follow sky orientation, we need to rotate 180 degree
    # see https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
    return stamp[np.newaxis, ::-1, ::-1]
