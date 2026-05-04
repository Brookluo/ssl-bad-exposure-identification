from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
from astropy.io import fits


logger = logging.getLogger(__name__)


def read_image(image_path: str | Path, padding: bool) -> tuple[np.ndarray, list[str]]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if "fits" in image_path.name:
        logger.debug("Reading FITS image from %s", image_path)
        return read_img_fits(image_path, padding)
    elif "npz" in image_path.name:
        logger.debug("Reading NPZ image from %s", image_path)
        return read_img_npz(image_path)
    else:
        raise ValueError(f"Unknown image format: {image_path.suffix}")


def read_img_fits(image_path: str | Path, padding: bool = False) -> tuple[np.ndarray, list[str]]:
    img_shape = (4094, 2046)
    half_size = 2046 if not padding else 2048
    ccdnames: list[str] = []
    with fits.open(image_path, memmap=True) as hdul:
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


def save_img_npz(image_path: str | Path, outdir: str | Path) -> Path:
    image_path = Path(image_path)
    outdir = Path(outdir)
    half_imdata, ccdnames = read_img_fits(image_path)
    new_img_name = str(image_path.name).replace(".fits.fz", ".npz")
    np.savez(outdir / new_img_name, ccdnames=ccdnames, half_imdata=half_imdata)
    return outdir / new_img_name


def read_img_npz(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(image_path)
    ccdnames: np.ndarray = data["ccdnames"]
    half_imdata: np.ndarray = data["half_imdata"]
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
