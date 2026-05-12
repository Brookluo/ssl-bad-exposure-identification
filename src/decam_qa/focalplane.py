"""DECam focal-plane stamp builder for multi-scale exposure analysis.

Builds low-resolution single-channel stamps from exposure HDUs by downsampling
CCDs and placing them in focal-plane layout.
"""
import numpy as np


# Focal-plane layout: x_pix and y_pix are CCD center positions in native
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
                           subtract_median_sky=True, reducer="median",
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
        Must have keys: ccdnum, image_hdu.
    binsize : int
        Downsampling factor. Native CCD -> ~(CCD_WIDTH/binsize, CCD_HEIGHT/binsize).
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
    stamp_h = _NATIVE_Y // binsize + 1
    stamp_w = _NATIVE_X // binsize + 1
    stamp = np.full((stamp_h, stamp_w), fill_value, dtype=np.float32)

    # Normalize input: DataFrame -> list of dicts
    if hasattr(exposure_rows, "columns"):  # pd.DataFrame
        exposure_rows = exposure_rows.to_dict("records")

    for row in exposure_rows:
        ccdnum = row["ccdnum"]
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

        # Place into canvas
        fp_idx = _CCDNUM_TO_IDX[ccdnum]
        x0 = int(_X_PIX[fp_idx] // binsize)
        y0 = int(_Y_PIX[fp_idx] // binsize)
        cc_h = downsampled.shape[0]
        cc_w = downsampled.shape[1]

        # Clip to canvas boundaries
        y_end = min(y0 + cc_h, stamp_h)
        x_end = min(x0 + cc_w, stamp_w)
        cc_h = y_end - y0
        cc_w = x_end - x0

        stamp[y0:y_end, x0:x_end] = downsampled[:cc_h, :cc_w]

    # Replace non-finite values
    stamp[~np.isfinite(stamp)] = fill_value

    return stamp[np.newaxis, :, :]
