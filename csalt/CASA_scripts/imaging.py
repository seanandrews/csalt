import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf /pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA'+j) for j in ext]

tclean(vis='/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/ALMAC9_late1_noisy.DATA.ms', 
       imagename='/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='2.72km/s', width='0.16km/s', nchan=17, 
       restfreq='230.538GHz', imsize=256, cell='0.025arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=0, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='5mJy', restoringbeam='common', 
       mask='')

make_mask('/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA.image',
          inc=30.0, PA=130.0, dx0=0.0, dy0=0.0,
          mstar=0.25, dist=150.0, vlsr=4000.0, zr=0.2,
          r_max=0.4, nbeams=1.5)

out_mask_name = '/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/kep.mask'
os.system('rm -rf '+out_mask_name)
os.system('mv /pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA.mask.image '+out_mask_name)

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf /pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA'+j) for j in ext]

tclean(vis='/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/ALMAC9_late1_noisy.DATA.ms', 
       imagename='/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='2.72km/s', width='0.16km/s', nchan=17, 
       restfreq='230.538GHz', imsize=256, cell='0.025arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=50000, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='5mJy', restoringbeam='common', 
       mask='/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/kep.mask')

exportfits('/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA.image', '/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA.image.fits', overwrite=True)
exportfits('/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA.mask', '/pool/asha0/SCIENCE/csalt/storage/data/ALMAC9_late1/images/ALMAC9_late1_noisy.DATA.mask.fits', overwrite=True)

os.system('rm -rf *.last')
