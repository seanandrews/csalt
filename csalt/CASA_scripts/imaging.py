import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf /pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA'+j) for j in ext]

tclean(vis='/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/radmc_std_pure.DATA.ms', 
       imagename='/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='-5.00km/s', width='0.16km/s', nchan=125, 
       restfreq='230.538GHz', imsize=256, cell='0.025arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=0, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='10mJy', restoringbeam='common', 
       mask='', interpolation='linear')

make_mask('/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA.image',
          inc=40.0, PA=130.0, dx0=0.0, dy0=0.0,
          mstar=1.0, dist=150.0, vlsr=5000.0, zr=0.2484347255922589,
          r_max=2.0, nbeams=1.5)

out_mask_name = '/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/kep.mask'
os.system('rm -rf '+out_mask_name)
os.system('mv /pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA.mask.image '+out_mask_name)

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf /pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA'+j) for j in ext]

tclean(vis='/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/radmc_std_pure.DATA.ms', 
       imagename='/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='-5.00km/s', width='0.16km/s', nchan=125, 
       restfreq='230.538GHz', imsize=256, cell='0.025arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=50000, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='10mJy', restoringbeam='common', 
       mask='/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/kep.mask', interpolation='linear')

exportfits('/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA.image', '/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA.image.fits', overwrite=True)
exportfits('/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA.mask', '/pool/asha0/SCIENCE/csalt/storage/data/radmc_std/images/radmc_std_pure.DATA.mask.fits', overwrite=True)


