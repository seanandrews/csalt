import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')
execfile('fconfig.py')


def generate_kepmask(MSname, imgname):

    # remove lingering files from previous runs
    ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
    for j in ext:
        os.system('rm -rf '+imgname+'_dirty'+j)

    # make a dirty image cube to guide the mask
    tclean(vis=MSname+'.ms', imagename=imgname+'_dirty', specmode='cube', 
           start=chanstart, width=chanwidth, nchan=nchan_out,
           outframe='LSRK', veltype='radio', restfreq=str(nu_rest/1e9)+'GHz',
           imsize=imsize, cell=cell, deconvolver='multiscale', scales=scales, 
           gain=gain, niter=0, nterms=1, interactive=False, weighting='briggs', 
           robust=robust, uvtaper=uvtaper, restoringbeam='common')

    # make a Keplerian mask
    make_mask(imgname+'_dirty.image', inc=incl, PA=PA+180, dx0=xoff, dy0=yoff,
              mstar=mstar, dist=dist, vlsr=vsys, zr=z0 / r0, 
              r_max=1.2 * r_l / dist, nbeams=1.5)

    os.system('rm -rf *.last')

    return 0



def clean_cube(MSname, imgname, maskname=None):

    # if necessary, make a mask
    if maskname is None:
        foo = generate_kepmask(MSname, imgname)
        maskname = imgname+'_dirty.mask.image'

    # remove lingering files from previous runs
    ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
    for j in ext:
        os.system('rm -rf '+imgname+j)

    # make a clean image cube
    tclean(vis=MSname+'.ms', imagename=imgname, specmode='cube',
           datacolumn='data',
           start=chanstart, width=chanwidth, nchan=nchan_out,
           outframe='LSRK', veltype='radio', restfreq=str(nu_rest/1e9)+'GHz',  
           imsize=imsize, cell=cell, deconvolver='multiscale', scales=scales, 
           gain=gain, niter=1000000, nterms=1, interactive=False, 
           weighting='briggs', robust=robust, uvtaper=uvtaper,
           threshold=threshold, mask=maskname, restoringbeam='common')

    os.system('rm -rf *.last')

    return 0
