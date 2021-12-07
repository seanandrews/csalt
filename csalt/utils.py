import os, sys, importlib
import numpy as np
import scipy.constants as sc
from astropy.io import fits
import matplotlib.pyplot as plt
sys.path.append('configs/')


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



def mk_tclean(in_MS, outname, 
              specmode='cube', chanstart='-5.0km/s', chanwidth='0.1km/s', 
              nchan=100, restfreq='230.538GHz', imsize=256, cell='0.1arcsec', 
              deconvolver='multiscale', scales=[0, 10, 30, 50], gain=0.1, 
              niter=100000, nterms=1, threshold='5mJy', interactive=False, 
              weighting='briggs', robust=0.5, uvtaper='', 
              restoringbeam='common', maskfile=''):

    fstr = \
        "ext = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', " + \
               "'.sumwt']\n" + \
        "[os.system('rm -rf "+outname+"'+j) for j in ext]\n\n" + \
        "tclean(vis='"+in_MS+".ms', \n       imagename='"+outname + "',\n" + \
        "       specmode='"+specmode+"', datacolumn='data', " + \
        "outframe='LSRK', veltype='radio', \n" + \
        "       start='"+chanstart+"', width='"+chanwidth+"', " + \
        "nchan="+str(nchan)+", \n" + \
        "       restfreq='"+restfreq+"', imsize="+str(imsize) + \
        ", cell='"+cell+"', \n       deconvolver='"+deconvolver+"', " + \
        "scales="+str(scales)+", gain="+str(gain)+",\n       " + \
        "niter="+str(niter)+", interactive="+str(interactive)+", " + \
        "weighting='"+weighting+"', robust="+str(robust)+", \n" + \
        "       uvtaper='"+uvtaper+"', threshold='"+threshold+"', " + \
        "restoringbeam='common', \n       mask='"+maskfile+"')"

    return fstr 




def img_cube(in_MS, out_img, cfg_file, mask='', masktype='kep'):

    # Load the information in the configuration file
    inp = importlib.import_module(cfg_file)

    # masktype = 'user' implies mask is a string input region
    # masktype = 'file' implies an already existing .mask file exists
    if np.logical_or((masktype == 'user'), (masktype == 'file')):

        # make sure mask file exists
        if masktype == 'file':
            if not os.path.exists(mask):
                print('Mask file "'+mask+'" does not exist.  Exiting...')
                return

        # Create the imaging script
        cstr = mk_tclean(in_MS, out_img, specmode='cube', 
                         chanstart=inp.chanstart, chanwidth=inp.chanwidth, 
                         nchan=inp.nchan_out, 
                         restfreq=str(inp.nu_rest/1e9)+'GHz', 
                         imsize=inp.imsize, cell=inp.cell, 
                         deconvolver='multiscale', scales=inp.scales, 
                         gain=inp.gain, niter=inp.niter, 
                         threshold=inp.threshold, robust=inp.robust, 
                         uvtaper=inp.uvtaper, restoringbeam='common', 
                         maskfile=mask)

        with open('csalt/CASA_scripts/imaging.py', "w") as outfile:
            print(cstr, file=outfile)

        # Run the imaging script
        os.system('rm -rf '+inp.casalogs_dir+'imaging.log')
        os.system('casa --nologger --logfile '+inp.casalogs_dir + \
                  'imaging.log '+'-c csalt/CASA_scripts/imaging.py')

    # masktype = 'kep' will create a Keplerian rotation mask
    elif masktype == 'kep':

        # Prefix script for creating a Keplerian mask
        pstr = \
            "import os, sys\n" + \
            "import numpy as np\n" + \
            "execfile('"+inp.kepmask_dir+"keplerian_mask.py')\n\n"

        # Script to make a dirty cube as guidance 
        cstr = mk_tclean(in_MS, out_img, specmode='cube', 
                         chanstart=inp.chanstart, chanwidth=inp.chanwidth, 
                         nchan=inp.nchan_out, 
                         restfreq=str(inp.nu_rest/1e9)+'GHz', 
                         imsize=inp.imsize, cell=inp.cell, 
                         deconvolver='multiscale', scales=inp.scales,
                         gain=inp.gain, niter=0, threshold=inp.threshold, 
                         robust=inp.robust, uvtaper=inp.uvtaper, 
                         restoringbeam='common', maskfile='')

        # Script to make a Keplerian mask
        incl = inp.incl if hasattr(inp, 'incl') else 30.
        PA = inp.PA if hasattr(inp, 'PA') else 130.
        dx = inp.dx if hasattr(inp, 'dx') else 0.
        dy = inp.dy if hasattr(inp, 'dy') else 0.
        mstar = inp.mstar if hasattr(inp, 'mstar') else 1.0
        dist = inp.dist if hasattr(inp, 'dist') else 150.
        Vsys = inp.Vsys if hasattr(inp, 'Vsys') else 0.
        zr = inp.zr if hasattr(inp, 'zr') else 0.
        r_max = inp.r_max if hasattr(inp, 'r_max') else 1.
        nbeams = inp.nbeams if hasattr(inp, 'nbeams') else 1.
        mstr = \
            "make_mask('"+out_img+".image',\n          inc="+str(incl)+", " + \
            "PA="+str(PA)+", dx0="+str(dx)+", dy0="+str(dy)+",\n" + \
            "          mstar="+str(mstar)+", dist="+str(dist)+", vlsr=" + \
            str(Vsys)+", zr="+str(zr)+",\n"+"          r_max="+str(r_max) + \
            ", nbeams="+str(nbeams)+")\n\n" + \
            "out_mask_name = '"+inp.reduced_dir+inp.basename + \
            "/images/kep.mask'\n" + \
            "os.system('rm -rf '+out_mask_name)\n" + \
            "os.system('mv "+out_img+".mask.image '+out_mask_name)"

        # Script to make a final cube with the mask
        istr = mk_tclean(in_MS, out_img, specmode='cube',
                         chanstart=inp.chanstart, chanwidth=inp.chanwidth,
                         nchan=inp.nchan_out,
                         restfreq=str(inp.nu_rest/1e9)+'GHz',
                         imsize=inp.imsize, cell=inp.cell,
                         deconvolver='multiscale', scales=inp.scales,
                         gain=inp.gain, niter=inp.niter, 
                         threshold=inp.threshold, robust=inp.robust, 
                         uvtaper=inp.uvtaper, restoringbeam='common', 
                         maskfile=inp.reduced_dir+inp.basename+\
                                  '/images/kep.mask')

        # Script to output the cube and mask in FITS format
        ostr = \
            "exportfits('"+out_img+".image', '"+out_img+".image.fits', " + \
            "overwrite=True)\n" + \
            "exportfits('"+out_img+".mask', '"+out_img+".mask.fits', " + \
            "overwrite=True)\n\n" + \
            "os.system('rm -rf *.last')"

        #print('\n' + pstr + cstr + '\n\n' + mstr + '\n\n' + istr + '\n\n' + ostr + '\n')

        with open('csalt/CASA_scripts/imaging.py', "w") as outfile:
            print(pstr+cstr+'\n\n'+mstr+'\n\n' + istr + '\n\n' + ostr, 
                  file=outfile)

        # Run the imaging script
        os.system('rm -rf '+inp.casalogs_dir+'imaging.log')
        os.system('casa --nologger --logfile '+inp.casalogs_dir + \
                  'imaging.log '+'-c csalt/CASA_scripts/imaging.py')
