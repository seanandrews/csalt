import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')


def generate_kepmask(cfg_file, MSname, imgname):

    execfile('configs/mdl_'+cfg_file+'.py')

    # make a dirty image cube to guide the mask
    ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
    [os.system('rm -rf '+imgname+'.dirty'+j) for j in ext]
    tclean(vis=MSname+'.ms', imagename=imgname+'.dirty', specmode='cube',
           start=chanstart, width=chanwidth, nchan=nchan_out,
           outframe='LSRK', veltype='radio', restfreq=str(nu_rest/1e9)+'GHz',
           imsize=imsize, cell=cell, deconvolver='multiscale', scales=scales,
           gain=gain, niter=0, nterms=1, interactive=False, weighting='briggs',
           robust=robust, uvtaper=uvtaper, restoringbeam='common')

    # make a Keplerian mask
    make_mask(imgname+'.dirty.image', inc=incl, PA=PA+180, dx0=dx, dy0=dy,
              mstar=mstar, dist=dist, vlsr=Vsys, zr=z0 / 10,
              r_max=1.2 * r_l / dist, nbeams=1.5)

    # store this as a 'default' mask to avoid repeated calls
    out_mask_name = img_dir+basename+'.default.mask'
    os.system('rm -rf '+out_mask_name)
    os.system('mv '+imgname+'.dirty.mask.image '+ out_mask_name)

    # cleanup
    ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
    [os.system('rm -rf '+imgname+'.dirty'+j) for j in ext]
    os.system('rm -rf *.last')

    return out_mask_name



def clean_cube(cfg_file, MSname, imgname, mask_name=None):

    execfile('configs/mdl_'+cfg_file+'.py')

    # if necessary, make a mask
    if mask_name is None:
        foo = generate_kepmask(MSname, imgname)
        mask_name = img_dir+basename+'.default.mask'

    # remove lingering files from previous runs
    ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
    [os.system('rm -rf '+imgname+j) for j in ext]

    # make a clean image cube
    tclean(vis=MSname, imagename=imgname, specmode='cube', datacolumn='data',
           start=chanstart, width=chanwidth, nchan=nchan_out,
           outframe='LSRK', veltype='radio', restfreq=str(nu_rest/1e9)+'GHz',
           imsize=imsize, cell=cell, deconvolver='multiscale', scales=scales,
           gain=gain, niter=niter, nterms=1, interactive=False,
           weighting='briggs', robust=robust, uvtaper=uvtaper,
           threshold=threshold, mask=mask_name, restoringbeam='common')

    # export image and mask to FITS
    exportfits(imgname+'.image', imgname+'.image.fits', overwrite=True)
    exportfits(imgname+'.mask', imgname+'.mask.fits', overwrite=True)

    # cleanup
    os.system('rm -rf *.last')

    return




cfg_file, cubetype, mask_name = sys.argv[-3:]

# Load configuration file
execfile('configs/mdl_'+cfg_file+'.py')

# Make a (Keplerian) mask if requested (or it doesn't already exist)
if np.logical_or(mask_name == 'None', not os.path.exists(mask_name)):
    mask_ = generate_kepmask(cfg_file, dataname+_ext+'.'+cubetype, 
                             img_dir+basename+_ext+'.'+cubetype)

# Image the cube
clean_cube(cfg_file, dataname+_ext+'.'+cubetype+'.ms', 
           img_dir+basename+_ext+'.'+cubetype, mask_name=mask_)
