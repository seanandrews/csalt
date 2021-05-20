import os, sys
import numpy as np
execfile('mconfig.py')
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')


# file extensions (for tidiness)
ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']


# loop through MS files to image
for i in do_img:

    # image filename format
    fname = 'im_'+basename+'-'+template+i

    # make a dirty image (to guide a mask)
    for j in ext:
        os.system('rm -rf data/images/'+fname+'_dirty'+j)
    tclean(vis='data/'+basename+'-'+template+i+'.ms',
           imagename='data/images/'+fname+'_dirty',
           specmode='cube', start=str(chanstart_out/1e3)+'km/s', 
           width=str(chanwidth_out/1e3)+'km/s', nchan=nchan_out, 
           outframe='LSRK', restfreq=str(restfreq/1e9)+'GHz',
           imsize=imsize, cell=cell, deconvolver='multiscale', scales=cscales,
           niter=0, weighting='briggs', robust=robust, interactive=False,
           nterms=1, restoringbeam='common')

    # make a Keplerian mask from the dirty image
    os.system('rm -rf data/images/'+fname+'_dirty.mask.image')
    make_mask('data/images/'+fname+'_dirty.image', inc=pars[0], PA=pars[1], 
              zr=pars[4]/10, mstar=pars[2], dist=dist, vlsr=pars[10], 
              r_max=1.2*pars[3], nbeams=1.5)

    # make a CLEAN image
    for j in ext:
        os.system('rm -rf data/images/'+fname+j)
    tclean(vis='data/'+basename+'-'+template+i+'.ms',
           imagename='data/images/'+fname, specmode='cube', 
           start=str(chanstart_out/1e3)+'km/s', 
           width=str(chanwidth_out/1e3)+'km/s', nchan=nchan_out,
           outframe='LSRK', restfreq=str(restfreq/1e9)+'GHz', imsize=imsize,
           cell=cell, deconvolver='multiscale', scales=cscales, niter=1000000,
           threshold=thresh, weighting='briggs', robust=robust, nterms=1,
           mask='data/images/'+fname+'_dirty.mask.image',
           interactive=False, restoringbeam='common')


# cleanup
os.system('rm -rf *.last *.log')

    
    