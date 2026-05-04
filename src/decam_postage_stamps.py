# Postage stamp + full-focal-plane visualization for DECam exposures.
# Originally from Rongpu Zhou (01/25/2024).
#
# Paths are configurable via environment variables:
#   DECAM_IMAGE_DIR        — image staging directory
#   DECAM_BLOB_DIR         — CCD blob mask directory
#   SURVEYCCD_PATH         — survey-ccds catalog (dr9)
#   SURVEYCCD_PATH_DR8     — survey-ccds catalog (dr8)

from __future__ import division, print_function
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import fitsio
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter

from . import decam_info
from .config import VisualizationConfig, load_config
from .decam_focalplane import (
    ccd_ra, ccd_dec, x_pix, y_pix,
    full_x_size as x_size, full_y_size as y_size,
    img_shape, pix_size,
)

params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'figure.facecolor': 'w'}
plt.rcParams.update(params)

ccdnamenumdict_inv = decam_info.ccdnum2name
ccdnum_list = decam_info.ccdnum_li_m1

_viz_config, _ = load_config()
image_dir = _viz_config.image_dir
blob_dir = _viz_config.blob_dir
surveyccd_path = _viz_config.surveyccd_path
surveyccd_path_dr8 = _viz_config.surveyccd_path_dr8

image_vrange = {'u':[-5, 5], 'g':[-5, 5], 'r':[-7, 7], 'i':[-10, 10], 'z':[-30, 30], 'Y':[-30, 30]}  # per 100s exposure time

################################################################################


def decam_plot(exposure, plot_path=None, figsize=(13, 12), vrange=None, cmap='seismic', dr8=False, binsize=20, median=True,
               blob_mask=False, ood_mask=False, gaussian_sigma=None, subtract_median_sky=True, show=False):
    '''
    Create high-resolution DECam images.

    Example:
        decam_plot(781475, 'tmp_mask_median.jpeg', binsize=20, blob_mask=True, ood_mask=True, median=True)
        decam_plot('<IMAGE_DIR>/decam/CP/V4.8.2a/CP20141219/c4d_141220_012420_ooi_g_ls9.fits.fz')
    '''

    if plot_path is not None and os.path.isfile(plot_path):
        return None

    if (type(exposure)==str) or (type(exposure)==np.str_):
        image_path = os.path.join(image_dir, exposure.strip())
    elif isinstance(exposure, int) or isinstance(exposure, np.integer):
        if not dr8:
            ccd = Table.read(surveyccd_path)
        else:
            ccd = Table.read(surveyccd_path_dr8)
        ccd_index = np.where(ccd['expnum']==exposure)[0][0]
        # band = ccd['image_filename'][ccd_index][ccd['image_filename'][ccd_index].find('_ooi_')+5]
        image_path = os.path.join(image_dir, ccd['image_filename'][ccd_index].strip())

    print(image_path)

    if not os.path.isfile(image_path):
        print('File does not exist:', image_path)
        return None

    header = fitsio.read_header(image_path)
    exptime = header['EXPTIME']

    if vrange is None:
        band = image_path[image_path.find('_ooi_')+5]
        vrange = np.array(image_vrange[band])
        vrange = vrange*exptime/100.
    print('vrange:', vrange)

    if blob_mask:
        str_loc = str.find(ccd['image_filename'][ccd_index].strip(), '.fits')
        img_filename_base = ccd['image_filename'][ccd_index].strip()[:str_loc]
        blob_path = os.path.join(blob_dir, 'blob_mask', img_filename_base+'-blobmask.npz')
        if (not os.path.isfile(blob_path)) or (os.stat(blob_path).st_size==0):
            print(blob_path+' is empty or does not exist!')
            return None
        blob_data = np.load(blob_path)

    plt.figure(figsize=figsize)

    hdu = fits.open(image_path)

    for ii, ccdnum in enumerate(ccdnum_list):

        ccdname = ccdnamenumdict_inv[ccdnum]

        try:
            # img = fitsio.read(image_path, ext=ccdname)
            img = hdu[ccdname].data
        except (KeyError, OSError):
            if ccdname!='S30':  # mute S30
                print('{} does not exist in image ({})!'.format(ccdname, exposure))
            continue

        if ood_mask:
            ood_path = image_path.replace('_ooi_', '_ood_')
            ood_hdu = fits.open(ood_path)
            # ood = fitsio.read(ood_path, ext=ccdname)
            ood = ood_hdu[ccdname].data

        if blob_mask:
            try:
                with fitsio.FITS(image_path) as f:
                    hdu_index = f.movnam_ext(ccdname)
                blob = blob_data['hdu'+str(hdu_index).zfill(2)]
            except (OSError, KeyError):
                print(blob_path+' hdu'+str(hdu_index)+' does not exist!')
                continue

        if ood_mask or blob_mask:
            img_mask = np.ones(img.shape, dtype=bool)
            if ood_mask:
                img_mask &= (ood==0)
            if blob_mask:
                img_mask &= (blob==True)
            img[~img_mask] = np.nan

        # Only keep the good half of the S7
        if ccdname=='S7':
            img_original = img.copy()
            half = img_shape[1] // 2
            img = img[:, :half]

        # Remove constant background
        if subtract_median_sky:
            if not blob_mask:
                # naive sky estimation
                mask = (img<np.nanpercentile(img.flatten(), 95))
                median_sky = np.nanmedian(img[mask].flatten())
            else:
                median_sky = np.nanmedian(img)
            img = img - median_sky

        # Add back the other half
        if ccdname=='S7':
            tmp = img_original
            half = img_shape[1] // 2
            tmp[:, :half] = img
            if tmp.dtype==float:
                tmp[:, half:] = np.nan
            elif tmp.dtype==int:
                tmp[:, half:] = 0
            img = tmp

        ################ downsize image ################

        # trim edges to enable downsizing
        # trimmed image size need to be multiples of binsize
        trim_size_x = img.shape[1] % binsize
        trim_size_y = img.shape[0] % binsize
        img = img[:(img.shape[0]-trim_size_y), :(img.shape[1]-trim_size_x)]

        # to ignore NAN values, use np.nanmean or np.nanmedian
        if not median:
            img = np.nanmean(np.nanmean(img.reshape((img.shape[0]//binsize, binsize, img.shape[1]//binsize,-1)), axis=3), axis=1)
        else:
            img = np.nanmedian(np.nanmedian(img.reshape((img.shape[0]//binsize, binsize, img.shape[1]//binsize,-1)), axis=3), axis=1)

        img[~np.isfinite(img)] = 0

        ################################################

        if gaussian_sigma is not None:
            img = gaussian_filter(img, gaussian_sigma, mode='reflect', truncate=3)

        ysize, xsize = img.shape
        ra, dec = ccd_ra[ii], ccd_dec[ii]

        fig = plt.imshow(img.T, cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                         extent=(ra-ysize*pix_size*binsize/2, ra+ysize*pix_size*binsize/2, dec-xsize*pix_size*binsize/2, dec+xsize*pix_size*binsize/2))

    plt.axis([1.07, -1.07, -1.0, 1.0])
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # plt.colorbar(fraction=0.04, pad=0.04)
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path)
    if show or plot_path is None:
        plt.show()
    else:
        plt.close()


def decam_postage_stamp(exposure, binsize=120, plot_path=None, save_path=None, vrange=None, cmap='seismic', dr8=False, median=True,
                        blob_mask=True, ood_mask=True, show=False):
    '''
    Create low-resolution postage stamps.

    Examples:
        decam_postage_stamp(781475, plot_path='tmp_stamp.png', save_path='stamp.npz')
        decam_postage_stamp('<IMAGE_DIR>/decam/CP/V4.8.2a/CP20181006/c4d_181007_074137_ooi_g_ls9.fits.fz',
                            plot_path='tmp_stamp.png', save_path='stamp.npz')
    '''

    if plot_path is not None and os.path.isfile(plot_path):
        return None

    if type(exposure)==str:
        image_path = exposure
    elif isinstance(exposure, int) or isinstance(exposure, np.integer):
        if not dr8:
            ccd = Table.read(surveyccd_path)
        else:
            ccd = Table.read(surveyccd_path_dr8)
        ccd_index = np.where(ccd['expnum']==exposure)[0][0]
        image_path = os.path.join(image_dir, ccd['image_filename'][ccd_index].strip())
    else:
        raise ValueError('exposure can either be string or integer!')

    print(image_path)
    band = image_path[image_path.find('_ooi_')+5]

    if save_path is not None and os.path.isfile(save_path):

        fullimg = np.load(save_path)['data']

    else:

        if not os.path.isfile(image_path):
            print('File does not exist:', image_path)
            return None

        header = fitsio.read_header(image_path)
        exptime = header['EXPTIME']

        if blob_mask:
            str_loc = str.find(image_path, '.fits')
            img_filename_base = image_path[len(image_dir)+1:str_loc]
            blob_path = os.path.join(blob_dir, 'blob_mask', img_filename_base+'-blobmask.npz')
            try:
                blob_data = np.load(blob_path)
            except FileNotFoundError:
                print(blob_path+' does not exist!')
                return None

        x_pix_small, y_pix_small = np.rint(x_pix/binsize).astype(int), np.rint(y_pix/binsize).astype(int)
        x_size_small, y_size_small = int(np.ceil(x_size/binsize)), int(np.ceil(y_size/binsize))
        fullimg = np.zeros((y_size_small, x_size_small))
        fullimg[:] = np.nan

        hdu = fits.open(image_path)

        for ii, ccdnum in enumerate(ccdnum_list):

            ccdname = ccdnamenumdict_inv[ccdnum]

            try:
                # img = fitsio.read(image_path, ext=ccdname)
                img = hdu[ccdname].data
            except (KeyError, OSError):
                print(ccdname+' does not exist in image!')
                continue

            if ood_mask:
                ood_path = image_path.replace('_ooi_', '_ood_')
                ood_hdu = fits.open(ood_path)
                # ood = fitsio.read(ood_path, ext=ccdname)
                ood = ood_hdu[ccdname].data

            if blob_mask:
                try:
                    with fitsio.FITS(image_path) as f:
                        hdu_index = f.movnam_ext(ccdname)
                    blob = blob_data['hdu'+str(hdu_index).zfill(2)]
                except (OSError, KeyError):
                    print(blob_path+' hdu'+str(hdu_index)+' does not exist!')
                    continue

            if ood_mask or blob_mask:
                img_mask = np.ones(img.shape, dtype=bool)
                if ood_mask:
                    img_mask &= (ood==0)
                if blob_mask:
                    img_mask &= (blob==True)
                img[~img_mask] = np.nan

            # Only keep the good half of the S7
            if ccdname=='S7':
                img_original = img.copy()
                half = img_shape[1] // 2
                img = img[:, :half]

            # Remove constant background
            if not blob_mask:
                # naive sky estimation
                mask = (img<np.nanpercentile(img.flatten(), 95))
                median_sky = np.nanmedian(img[mask].flatten())
            else:
                median_sky = np.nanmedian(img)
            img = img - median_sky

            # Add back the other half
            if ccdname=='S7':
                tmp = img_original
                half = img_shape[1] // 2
                tmp[:, :half] = img
                tmp[:, half:] = np.nan
                img = tmp

            ################ downsize image ################

            # trim edges to enable downsizing
            # trimmed image size need to be multiples of binsize
            trim_size_x = img.shape[1] % binsize
            trim_size_y = img.shape[0] % binsize
            img = img[:(img.shape[0]-trim_size_y), :(img.shape[1]-trim_size_x)]
            nanmask = np.isfinite(img)

            # to ignore NAN values, use np.nanmean or np.nanmedian
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not median:
                    img = np.nanmean(np.nanmean(img.reshape((img.shape[0]//binsize, binsize, img.shape[1]//binsize,-1)), axis=3), axis=1)
                else:
                    img = np.nanmedian(np.nanmedian(img.reshape((img.shape[0]//binsize, binsize, img.shape[1]//binsize,-1)), axis=3), axis=1)
                nanmask = np.mean(np.mean(nanmask.reshape((nanmask.shape[0]//binsize, binsize, nanmask.shape[1]//binsize,-1)), axis=3), axis=1)
            mask = nanmask<0.01  # require at least 1% of the pixels to be unmasked
            img[mask] = np.nan

            ################################################
            img = img.T
            img_y_size, img_x_size = img.shape
            fullimg[y_pix_small[ii]:y_pix_small[ii]+img_y_size, x_pix_small[ii]:x_pix_small[ii]+img_x_size] = img

        fullimg = np.flip(fullimg, 1)

        if save_path is not None:
            np.savez_compressed(save_path, data=fullimg)

    fullimg[~np.isfinite(fullimg)] = 0

    if (plot_path is not None) or show:

        if vrange is None:
            vrange = np.array(image_vrange[band])
            vrange = vrange*exptime/100.
        ax = create_image(fullimg, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        plt.savefig(plot_path)

        if show:
            plt.show()
        else:
            plt.close()
