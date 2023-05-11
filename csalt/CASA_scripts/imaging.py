import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf /pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA'+j) for j in ext]

tclean(vis='/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/sg_modelbi_pure.DATA.ms', 
       imagename='/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='-5.6km/s', width='0.08km/s', nchan=141, 
       restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=0, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='10mJy', restoringbeam='common', 
       mask='', interpolation='linear')

make_mask('/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA.image',
          inc=40.0, PA=130.0, dx0=0.0, dy0=0.0,
          mstar=1.0, dist=150.0, vlsr=0.0, zr=0.3,
          r_max=2.6, nbeams=2)

out_mask_name = '/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/kep.mask'
os.system('rm -rf '+out_mask_name)
os.system('mv /pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA.mask.image '+out_mask_name)

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf /pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA'+j) for j in ext]

tclean(vis='/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/sg_modelbi_pure.DATA.ms', 
       imagename='/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='-5.6km/s', width='0.08km/s', nchan=141, 
       restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=100000, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='10mJy', restoringbeam='common', 
       mask='/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/kep.mask', interpolation='linear')

exportfits('/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA.image', '/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA.image.fits', overwrite=True)
exportfits('/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA.mask', '/pool/asha0/SCIENCE/csalt/storage/data/sg_modelbi/images/sg_modelbi_pure.DATA.mask.fits', overwrite=True)


