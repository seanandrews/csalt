import os, sys, importlib
import numpy as np
import scipy.constants as sc
from astropy.io import fits
import matplotlib.pyplot as plt


def cubestats(imgname):

    # load the image cube
    chdu = fits.open(imgname+'.image.fits')
    cube, hd = np.squeeze(chdu[0].data), chdu[0].header
    chdu.close()

    # load the mask (True = disk, False = sky)
    mhdu = fits.open(imgname+'.mask.fits')
    mask = np.squeeze(mhdu[0].data) != 0	# converts to bool
    mhdu.close()

    # channel "width"
    dnu, nu_rest = np.abs(hd['CDELT3']), hd['RESTFRQ']
    dV = sc.c * (dnu / nu_rest)		# in m/s

    # beam properties
    bmaj, bmin, bpa = hd['BMAJ'], hd['BMIN'], hd['BPA']
    beam_area = (np.pi / 180)**2 * np.pi * bmaj * bmin / (4 * np.log(2))
    pix_area = (np.pi / 180)**2 * np.abs(hd['CDELT1'] * hd['CDELT2'])
    
    # compute the integrated flux in the mask
    flux = 1e-3 * dV * np.sum(cube[mask] * pix_area / beam_area) # in Jy km/s

    # compute the peak intensity in the mask
    peak = np.max(cube[mask])

    # compute the RMS noise outside the mask
    RMS = 1e3 * np.std(cube[~mask])

    # compute the RMS in each channel and make a histogram
    RMS_per_channel = np.empty(cube.shape[0])
    for i in range(len(RMS_per_channel)):
        chanmap, chanmask = cube[i,:,:], mask[i,:,:]
        RMS_per_channel[i] = 1e3 * np.std(chanmap[~chanmask])
    print(np.median(RMS_per_channel), np.mean(RMS_per_channel))
    plt.hist(1.0 * RMS_per_channel)
    plt.show()
    

    print(' ')
    print('# %s' % imgname+'.image')
    print('# Beam = %.3f arcsec x %.3f arcsec (%.2f deg)' % \
          (bmaj * 3600, bmin * 3600, bpa))
    print('# channel spacing = %i m/s' % np.int(np.round(dV)))
    print('# Flux in mask = %.3f Jy km/s' % flux)
    print('# Peak intensity in mask = %.3f Jy/beam/channel' % peak)
    print('# mean RMS noise level = %.2f mJy/beam/channel' % RMS)
    print(' ')
    

    return RMS
