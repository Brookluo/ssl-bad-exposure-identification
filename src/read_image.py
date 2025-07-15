from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from typing import Union


logger = logging.getLogger(__name__)


def read_image(image_path: Union[Path, str], padding):
    """Read an image from the given file path. The file format must be either .fits.gz or .npz.
    Note the .npz file is constructed from a .fits.gz exposure file.

    Args:
        image_path (Union[Path, str]): The path to the image file.
        padding (bool): the padding parameter passed on to the read_img_fits function.

    Returns:
        numpy.ndarray: The image data.

    Raises:
        ValueError: If the image format is unknown.
    """
    image_path = Path(image_path)
    if "fits" in image_path.name:
        logger.debug("Reading FITS image from %s", image_path)
        return read_img_fits(image_path, padding)
    elif "npz" in image_path.name:
        logger.debug("Reading NPZ image from %s", image_path)
        return read_img_npz(image_path)
    else:
        raise ValueError(f"Unknown image format: {image_path.suffix}")


def read_img_fits(image_path, padding=False):
    """Read a FITS image file and extract image data for each CCD. The image data is divided in half
    and formatted to be square either via trimming or padding.

    Args:
        image_path (Union[Path, str]): The path to the FITS image file.
        padding (bool): if true, adding a row/column of empty pixel to each side to expand the image to (4096, 2048).
            Else, trimming the image -> removing a row from bottom and top
    Returns:
        tuple: A tuple containing the half-sized image data for each CCD and a list of CCD names.

    """
    img_shape = (4094, 2046)
    half_size = 2046 if not padding else 2048
    ccdnames = []
    with fits.open(image_path, memmap=True) as hdul:
        # remove a row from top and bottom of the image to keep half image a square
        num_ccd = len(hdul) - 1
        half_imdata = np.zeros(
            (num_ccd * 2, half_size, half_size), dtype=np.float32
        )
        for i in range(1, len(hdul)):
            ccdnames.append(hdul[i].name)
            if padding:
                half_imdata[(i - 1) * 2, 1:, 1:-1] = hdul[i].data[:img_shape[0]//2]
                half_imdata[(i - 1) * 2 + 1, :-1, 1:-1] = hdul[i].data[img_shape[0]//2:]
            else:
                half_imdata[(i - 1) * 2] = hdul[i].data[1 : half_size + 1]
                half_imdata[(i - 1) * 2 + 1] = hdul[i].data[half_size + 1 : -1]
    return half_imdata, ccdnames


def save_img_npz(image_path: Union[Path, str], outdir: Union[Path, str]):
    image_path = Path(image_path)
    outdir = Path(outdir)
    half_imdata, ccdnames = read_img_fits(image_path)
    new_img_name = str(image_path.name).replace(".fits.fz", ".npz")
    np.savez(outdir / new_img_name, ccdnames=ccdnames, half_imdata=half_imdata)
    return outdir / new_img_name


def read_img_npz(image_path):
    """Read image data from a .npz file.

    Args:
        image_path (str): The path to the .npz file.

    Returns:
        half_imdata (numpy.ndarray): The image data where each CCD is divided in half to be square.
        img_names (numpy.ndarray): The names of the images.
    """
    data = np.load(image_path)
    ccdnames = data["ccdnames"]
    half_imdata = data["half_imdata"]
    return half_imdata, ccdnames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a directory of FITS images to NPZ."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="The root directory (eg., decam) containing the FITS images.",
    )
    parser.add_argument(
        "--remove-fits",
        "-rm",
        action="store_true",
        help="Remove the FITS files after conversion.",
    )
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    logger.info("Converting FITS images to NPZ in %s", args.image_dir)
    for fp in args.image_dir.rglob("*.fits.fz"):
        save_img_npz(fp, fp.parent)
        if args.remove_fits:
            fp.unlink()
