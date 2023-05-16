import os
import sys
import warnings
import importlib
import numpy as np
import scipy.constants as sc
from csalt.keplerian_mask import *
from casatasks import tclean

warnings.filterwarnings("ignore")


""" TCLEAN WRAPPER """
def imagecube(msfile, outfile,
              mk_kepmask=True, kepmask_kwargs=None, tclean_kwargs=None):

    # Populate tclean keywords
    kw = {} if tclean_kwargs is None else tclean_kwargs
    if 'specmode' not in kw: 
        kw['specmode'] = 'cube' 
    if 'datacolumn' not in kw: 
        kw['datacolumn'] = 'data'
    if 'outframe' not in kw: 
        kw['outframe'] = 'LSRK'
    if 'veltype' not in kw: 
        kw['veltype'] = 'radio'
    if 'start' not in kw:
        kw['start'] = '-5.0km/s'
    if 'width' not in kw:
        kw['width'] = '0.16km/s'
    if 'nchan' not in kw:
        kw['nchan'] = 70
    if 'restfreq' not in kw:
        kw['restfreq'] = '230.538GHz'
    if 'imsize' not in kw:
        kw['imsize'] = 512
    if 'cell' not in kw:
        kw['cell'] = '0.02arcsec'
    if 'deconvolver' not in kw:
        kw['deconvolver'] = 'multiscale'
    if 'scales' not in kw:
        kw['scales'] = [0, 10, 30, 50]
    if 'weighting' not in kw:
        kw['weighting'] = 'briggs'
    if 'robust' not in kw:
        kw['robust'] = 0.5
    if 'threshold' not in kw:
        kw['threshold'] = '10mJy'
    if 'restoringbeam' not in kw:
        kw['restoringbeam'] = 'common'
    if 'niter' not in kw:
        kw['niter'] = 100000
    niter_hold = kw['niter']

    # Prepare workspace by removing old files
    ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt']
    [os.system('rm -rf '+outfile+'.'+j) for j in ext]

    # Assign masking
    if mk_kepmask:
        # Prepare workspace by removing old files
        ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt', 'mask']
        [os.system('rm -rf '+outfile+'.'+j) for j in ext]

        # First make a dirty image as a guide
        kw['niter'] = 0
        kw['mask']  = ''
        os.system('rm -rf '+outfile+'.mask')
        tclean(msfile, imagename=outfile, **kw)

        # Parse the mask keywords
        mkw = {} if kepmask_kwargs is None else kepmask_kwargs
        if 'inc' not in mkw:
            mkw['inc'] = 40.
        if 'PA' not in mkw:
            mkw['PA'] = 130.
        if 'mstar' not in mkw:
            mkw['mstar'] = 1.0
        if 'vlsr' not in mkw:
            mkw['vlsr'] = 0.0e3
        if 'dist' not in mkw:
            mkw['dist'] = 150.
        _ = make_kepmask(outfile+'.image', **mkw)

        # Now assign the mask
        os.system('rm -rf '+outfile+'.kep.mask')
        os.system('mv '+outfile+'.mask.image '+outfile+'.kep.mask')
        kw['mask'] = outfile+'.kep.mask'

    else:
        kw['mask'] = kw.pop('mask', '') 

    # Prepare workspace by removing old files
    ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt', 'mask']
    [os.system('rm -rf '+outfile+'.'+j) for j in ext]

    # Image
    kw['niter'] = niter_hold
    tclean(msfile, imagename=outfile, **kw)
