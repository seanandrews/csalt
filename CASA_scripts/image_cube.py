import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')


def generate_kepmask(name, mask_pars, subname=''):

    # load the config file corresponding to this basename
    execfile('fconfig_'+name+'.py')

    # remove lingering files from previous runs
    ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']
    for j in ext:
        os.system('rm -rf '+dataname+subname+j)

    # make a dirty image cube to guide the mask
    tclean(vis=dataname+subname+'.ms', imagename=dataname+subname,
           specmode='cube', start=chanstart, width=chanwidth, nchan=nchan_out,
           outframe='LSRK', veltype='radio', restfreq=str(nu_rest/1e9)+'GHz',
           imsize=imsize, cell=cell, deconvolver='multiscale', scales=scales, 
           gain=gain, niter=0, nterms=1, interactive=False, weighting='briggs', 
           robust=robust, uvtaper=uvtaper, restoringbeam='common')

    # make a Keplerian mask
    make_mask(dataname+subname+'.image', inc=mask_pars[0], PA=mask_pars[1]+180,
              mstar=mask_pars[2], dist=mask_pars[3], vlsr=mask_pars[4], 
              r_max=1.2*mask_pars[5]/mask_pars[3], zr=mask_pars[6]/10, 
              nbeams=1.5)
           




    # make a Keplerian mask from the dirty image
    os.system('rm -rf data/images/'+fname+'_dirty.mask.image')
    make_mask('data/images/'+fname+'_dirty.image', inc=pars[0], PA=pars[1]+180, 
              zr=pars[4]/10, mstar=pars[2], dist=dist, vlsr=pars[10], 
              r_max=1.2*pars[3]/dist, nbeams=1.5)

    # make a CLEAN image
    for j in ext:
        os.system('rm -rf data/images/'+fname+j)
    tclean(vis='data/'+basename+'-'+template+i+'.ms',
           imagename='data/images/'+fname, specmode='cube', 
           start=str(ivel[0]/1e3)+'km/s', 
           width=str(dvel0/1e3)+'km/s', nchan=len(ivel),
           outframe='LSRK', restfreq=str(restfreq/1e9)+'GHz', imsize=imsize,
           cell=cell, deconvolver='multiscale', scales=cscales, niter=1000000,
           threshold=thresh, weighting='briggs', robust=robust, nterms=1,
           mask='data/images/'+fname+'_dirty.mask.image',
           interactive=False, restoringbeam='common')

    # cleanup
    os.system('rm -rf data/images/'+fname+'_dirty*')

# cleanup
os.system('rm -rf *.last *.log')
