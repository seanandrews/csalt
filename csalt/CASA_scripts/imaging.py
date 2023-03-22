import os, sys
import numpy as np
execfile('/home/caitlynh/bd34_scratch/chardiman/keplerian_mask/keplerian_mask.py')

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf test'+j) for j in ext]

tclean(vis='/home/caitlynh/bd34_scratch/chardiman/DM_Tau_ACASBLB_no_ave_selfcal_time_ave_12CO_spwsel.ms', 
       imagename='test',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='-1.0km/s', width='0.03km/s', nchan=400, 
       restfreq='345.7959899GHz', imsize=512, cell='0.0125arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=0, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='14mJy', restoringbeam='common', 
       mask='', interpolation='linear')

make_mask('test.image',
          inc=40.0, PA=130.0, dx0=0.0, dy0=0.0,
          mstar=0.7, dist=150.0, vlsr=0.0, zr=2.5,
          r_max=1.6, nbeams=1.5)

out_mask_name = 'storage/data/exoALMA/images/kep.mask'
os.system('rm -rf '+out_mask_name)
os.system('mv test.mask.image '+out_mask_name)

ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
[os.system('rm -rf test'+j) for j in ext]

tclean(vis='/home/caitlynh/bd34_scratch/chardiman/DM_Tau_ACASBLB_no_ave_selfcal_time_ave_12CO_spwsel.ms', 
       imagename='test',
       specmode='cube', datacolumn='data', outframe='LSRK', veltype='radio', 
       start='-1.0km/s', width='0.03km/s', nchan=400, 
       restfreq='345.7959899GHz', imsize=512, cell='0.0125arcsec', 
       deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1,
       niter=50000, interactive=False, weighting='briggs', robust=0.5, 
       uvtaper='', threshold='14mJy', restoringbeam='common', 
       mask='storage/data/exoALMA/images/kep.mask', interpolation='linear')

exportfits('test.image', 'test.image.fits', overwrite=True)
exportfits('test.mask', 'test.mask.fits', overwrite=True)
