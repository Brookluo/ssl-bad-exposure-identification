from __future__ import annotations

"""Information for decam focal plane CCD positions and plotting
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import fitsio
from astropy.visualization import ZScaleInterval
from . import decam_info
import logging

logger = logging.getLogger(__name__)


ccd_ra = [-0.31244368,-0.00214103, 0.30855858,-0.46789986,-0.1573787 , 0.15336207,
  0.4637642 ,-0.62325889,-0.312972  ,-0.00212455, 0.30866507, 0.61908193,
 -0.77859061,-0.46870955,-0.15780883, 0.15334942, 0.46418217, 0.77441054,
 -0.77876058,-0.46892617,-0.15799484, 0.15333136, 0.46448109, 0.77444204,
 -0.93389515,-0.624237  ,-0.31362077,-0.00213867, 0.30892024, 0.61974856,
  0.92929411,-0.93410772,-0.62439031,-0.31379523,-0.00251046, 0.30860373,
  0.61929563, 0.92907893,-0.77928668,-0.46927775,-0.15819325, 0.15315534,
  0.464108  , 0.77408146,-0.7791703 ,-0.46938561,-0.15825837, 0.15269545,
  0.46382537, 0.77383443,-0.6239286 ,-0.31363566,-0.00262614, 0.30814956,
  0.61848423,-0.46862823,-0.15833137, 0.15254403, 0.46295505,-0.31333245,
  0.30765903]
ccd_dec = [ 0.90299039, 0.90274404, 0.90285652, 0.73894001, 0.73933177, 0.73919444,
  0.73865878, 0.5745655 , 0.57508801, 0.57510357, 0.57486577, 0.57414278,
  0.41001556, 0.41059824, 0.41088721, 0.41057117, 0.41032572, 0.40963196,
  0.24595122, 0.24597951, 0.24624207, 0.24619019, 0.24582139, 0.24534302,
  0.08128957, 0.08150002, 0.08130657, 0.08138846, 0.0810964 , 0.08093379,
  0.08089282,-0.08302691,-0.08319348,-0.08340522,-0.08351659,-0.08366242,
 -0.08355805,-0.08365399,-0.24756494,-0.2479717 ,-0.24812127,-0.24835309,
 -0.2482645 ,-0.2480924 ,-0.41173856,-0.41236738,-0.41281328,-0.41296242,
 -0.41270174,-0.41225407,-0.57638265,-0.57687683,-0.57711492,-0.57725814,
 -0.57674114,-0.74071528,-0.74115162,-0.74130891,-0.74095896,-0.9049206 ,
 -0.90515532]

full_x_size, full_y_size = 29590, 26787 # Pixel size if all the CCDs are stitched into one image
x_pix = [ 8498. , 12747. , 16996. ,  6373.5, 10622.5, 14871.5, 19120.5,
                   4249. ,  8498. , 12747. , 16996. , 21245. ,  2124.5,  6373.5,
                  10622.5, 14871.5, 19120.5, 23369.5,  2124.5,  6373.5, 10622.5,
                  14871.5, 19120.5, 23369.5,    -0. ,  4249. ,  8498. , 12747. ,
                  16996. , 21245. , 25494. ,    -0. ,  4249. ,  8498. , 12747. ,
                  16996. , 21245. , 25494. ,  2124.5,  6373.5, 10622.5, 14871.5,
                  19120.5, 23369.5,  2124.5,  6373.5, 10622.5, 14871.5, 19120.5,
                  23369.5,  4249. ,  8498. , 12747. , 16996. , 21245. ,  6373.5,
                  10622.5, 14871.5, 19120.5,  8498. , 16996. ]
y_pix = [    0,     0,     0,  2249,  2249,  2249,  2249,  4498,  4498,
                  4498,  4498,  4498,  6747,  6747,  6747,  6747,  6747,  6747,
                  8996,  8996,  8996,  8996,  8996,  8996, 11245, 11245, 11245,
                 11245, 11245, 11245, 11245, 13494, 13494, 13494, 13494, 13494,
                 13494, 13494, 15743, 15743, 15743, 15743, 15743, 15743, 17992,
                 17992, 17992, 17992, 17992, 17992, 20241, 20241, 20241, 20241,
                 20241, 22490, 22490, 22490, 22490, 24739, 24739]

img_shape = (4094, 2046)
pix_size = 0.262/3600

fp_ccd_pos = pd.DataFrame(zip(ccd_ra, ccd_dec, x_pix, y_pix),
             columns=['ra', 'dec', 'x_pix', 'y_pix'],
             index=decam_info.ccdnum_li_m1)


def plot_decam_exposure(
    exp_path: str | Path, fig,
    vrange: tuple[float, float] | None = None,
    cmap: str = 'gray',
    ood_mask: bool = False,
    median: bool = False,
    binsize: int = 20,
) -> None:
    # header = fitsio.read_header(exp_path)
    # exptime = header['EXPTIME']
    with fitsio.FITS(exp_path) as imgfits:
        name_imdata = [(imgfits[i].get_extname(), imgfits[i].read()) for i in range(1, len(imgfits))]
        num_ccds = len(imgfits) - 1
    img_ccdnames, imdata = list(zip(*name_imdata))
    imdata = np.stack(imdata)
    if vrange is None:
        # first calculate the zscale for the image
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(imdata)
        vrange = (vmin, vmax)
    ccdnum_li = decam_info.ccdnum_li_m1 if num_ccds == 61 else decam_info.ccdnum_li_m2
    ax = fig.gca()
    for i, ccdnum in enumerate(ccdnum_li):
        ccdname = decam_info.ccdnum2name[ccdnum]
        # img = imgfits[ccdname].read()
        one_ccd_pos = fp_ccd_pos.loc[ccdnum]
        img = imdata[img_ccdnames.index(ccdname)]
        img_mask = np.ones(img.shape, dtype=bool)
        if ood_mask:
            ood_path = str(exp_path).replace('_ooi_', '_ood_')
            ood = fitsio.read(ood_path, ext=ccdname)
            img_mask &= (ood==0)
        img[~img_mask] = np.nan
        trim_size_x = img.shape[1] % binsize
        trim_size_y = img.shape[0] % binsize
        img = img[:(img.shape[0]-trim_size_y), :(img.shape[1]-trim_size_x)]
        # subdivide the CCD image to smaller patches to better estimate the background
        func_contract = np.nanmedian if median else np.nanmean
        img = func_contract(
                func_contract(
                    img.reshape(
                        (
                            img.shape[0]//binsize,
                            binsize,
                            img.shape[1]//binsize,
                            -1
                        )
                    ),
                    axis=3
                ),
                axis=1
        )
        img[~np.isfinite(img)] = 0
        ysize, xsize = img.shape
        ra = one_ccd_pos['ra']
        dec = one_ccd_pos['dec']
        # ra, dec = ccd_ra[i], ccd_dec[i]
        ax.imshow(img.T, cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                  extent=(
                      ra-ysize*pix_size*binsize/2,
                      ra+ysize*pix_size*binsize/2,
                      dec-xsize*pix_size*binsize/2,
                      dec+xsize*pix_size*binsize/2
                  )
        )
    # imgfits.close()
    ax.axis([1.07, -1.07, -1.0, 1.0])
    ax.axis('off')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # fig.tight_layout()
    # return fig


def assemble_focal_plane(
    exp_path: str | Path, outfile: str | Path | None = None,
) -> np.ndarray:
    # exp_path: full path to the exposure
    logger.info("Loading exposure: %s", exp_path)
    with fitsio.FITS(exp_path) as imgfits:
        name_imdata = [(imgfits[i].get_extname(), imgfits[i].read()) for i in range(1, len(imgfits))]
        num_ccds = len(imgfits) - 1
    img_ccdnames, imdata = list(zip(*name_imdata))
    imdata = np.stack(imdata)
    logger.info("Start to assemble the full focal plane into single array")
    full_fp = np.zeros((full_x_size, full_y_size), dtype=np.float32)
    ccdnum_li = decam_info.ccdnum_li_m1 if num_ccds == 61 else decam_info.ccdnum_li_m2
    for i, ccdnum in enumerate(ccdnum_li):
        ccdname = decam_info.ccdnum2name[ccdnum]
        one_ccd_pos = fp_ccd_pos.loc[ccdnum]
        img = imdata[img_ccdnames.index(ccdname)]
        y_pix = int(one_ccd_pos['y_pix'])
        x_pix = int(one_ccd_pos['x_pix'])
        xsize, ysize = img.shape
        full_fp[x_pix:x_pix+xsize, y_pix:y_pix+ysize] = img
    if outfile:
        np.save(outfile, full_fp)
    return full_fp
