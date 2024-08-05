import os
import sys
import warnings
import importlib
import datetime
import numpy as np
import scipy.constants as sc
from astropy.io import fits, ascii
from csalt.keplerian_mask import *
from casatasks import tclean
from vis_sample.classes import SkyImage
import casatools
import emcee
import importlib
import matplotlib.pyplot as plt
_ = importlib.import_module('plot_setups')
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_img.mplstyle'])


warnings.filterwarnings("ignore")

class dataset:

    def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # Spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK
        self.nchan = len(nu_TOPO)

        # Spatial frequencies in meters and lambda units
        self.um = um
        self.vm = vm
        self.ulam = self.um * np.mean(self.nu_TOPO) / sc.c
        self.vlam = self.vm * np.mean(self.nu_TOPO) / sc.c

        # Visibilities, weights, and timestamp IDs
        self.vis = vis
        self.wgt = wgt
        self.tstamp = tstamp_ID

        # Utility size trackers
        self.npol, self.nvis = vis.shape[0], vis.shape[2]
        self.nstamps = len(np.unique(tstamp_ID))


"""
    Function to read contents of MS file into a data dictionary.
"""
def read_MS(msfile, overwrite_topo=False):

    # Make sure the file exists
    if not os.path.exists(msfile):
        print('I cannot find '+msfile+'.  Exiting.')
        sys.exit()

    # Ingest the SPW information into a dictionary
    ms = casatools.ms()
    ms.open(msfile)
    spw_dict = ms.getspectralwindowinfo()
    ms.close()

    # Identify the number of distinct execution blocks
    Nobs = len(spw_dict)

    # Initialize a data dictionary
    data_dict = {'Nobs': Nobs, 'input_file': msfile}

    # Loop over executions to load dataset objects into the data dictionary
    for EB in range(Nobs):
        # Compute the TOPO frequencies
        spw = spw_dict[str(EB)]
        nu = spw['Chan1Freq'] + spw['ChanWidth'] * np.arange(spw['NumChan'])

        # Open the MS file for this EB
        ms.open(msfile)
        ms.selectinit(datadescid=EB)

        # Load the data into a dictionary
        d = ms.getdata(['data', 'weight', 'u', 'v', 'time'])

        # Identify the unique timestamps
        tstamps = np.unique(d['time'])

        # Allocate the timestamp index and LSRK frequency grids
        tstamp_ID = np.empty_like(d['time'])
        nu_ = np.empty((len(tstamps), len(nu)))

        # Loop over timestamps to populate index and LSRK frequency grids
        if overwrite_topo:
            for istamp in range(len(tstamps)):
                nu_[istamp,:] = 1. * nu
        else:
            for istamp in range(len(tstamps)):
                tstamp_ID[d['time'] == tstamps[istamp]] = istamp
                nu_[istamp,:] = ms.cvelfreqs(spwids=[EB],
                                             mode='channel',
                                             outframe='LSRK',
                                             obstime=str(tstamps[istamp])+'s')

        # Close the MS file
        ms.close()

        # Append a dataset object to the data dictionary
        data_dict[str(EB)] = dataset(d['u'],
                                     d['v'],
                                     d['data'],
                                     d['weight'],
                                     nu,
                                     nu_,
                                     tstamp_ID)

    return data_dict


"""
Function to write contents of a data dictionary to MS file 
"""
def write_MS(data_dict, outfile='out.ms', resid=False, direct_file=False):

    # Copy the input MS file to the output file
    if not direct_file:
        if not os.path.exists(data_dict['input_file']):
            print('I cannot find the input MS file '+data_dict['input_file']+\
                  ' to make a copy.  Exiting.')
            sys.exit()
        else:
            os.system('rm -rf '+outfile)
            os.system('cp -r '+data_dict['input_file']+' '+outfile)

    # Loop over the observations to pack into the MS file
    ms = casatools.ms()
    for EB in range(data_dict['Nobs']):
        # open the MS file for this EB
        ms.open(outfile, nomodify=False)
        ms.selectinit(datadescid=EB)

        # pull the data array
        d = ms.getdata(['data'])

        # replace with the model array or the residuals
        if resid:
            d['data'] -= data_dict[str(EB)].vis
        else:
            d['data'] = data_dict[str(EB)].vis
        ms.putdata(d)

        # close the MS file
        ms.close()

    return None


""" TCLEAN WRAPPER """
def imagecube(msfile, outfile,
              mk_kepmask=True, kepmask_kwargs=None, tclean_kwargs=None,
              dirty_only=False):

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
    if 'uvtaper' not in kw:
        kw['uvtaper'] = ''
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

    if not dirty_only:
        # Prepare workspace by removing old files
        ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt', 'mask']
        [os.system('rm -rf '+outfile+'.'+j) for j in ext]

        # Image
        kw['niter'] = niter_hold
        tclean(msfile, imagename=outfile, **kw)

    return None



def LST_to_UTC(date, LST, longitude):

    # Load the measures and quanta tools
    me = casatools.measures()
    qa = casatools.quanta()

    # Parse input LST into hours
    h, m, s = LST.split(LST[2])
    LST_hours = int(h) + int(m) / 60. + float(s) / 3600.

    # Calculate the MJD
    mjd = me.epoch('utc', date)['m0']['value']
    jd = mjd + 2400000.5
    T = (jd - 2451545.0) / 36525.0
    sidereal = 280.46061837 + 360.98564736629 * (jd - 2451545.) \
               + 0.000387933*T**2 - (1. / 38710000.)*T**3
    sidereal += longitude
    sidereal /= 360.
    sidereal -= np.floor(sidereal)
    sidereal *= 24.0
    if (sidereal < 0): sidereal += 24
    if (sidereal >= 24): sidereal -= 24
    delay = (LST_hours - sidereal) / 24.0
    if (delay < 0.0): delay += 1.0
    mjd += delay / 1.002737909350795

    # Convert to UTC date/time string
    today = me.epoch('utc', 'today')
    today['m0']['value'] = mjd
    hhmmss = qa.time(today['m0'], form='', prec=6, showform=False)[0]
    dt = qa.splitdate(today['m0'])
    ut = '%s%s%02d%s%02d%s%s' % \
         (dt['year'], '/', dt['month'], '/', dt['monthday'], '/', hhmmss)

    return ut


def doppler_set(nu_rest, vel_tune, datestring, RA, DEC,
                equinox='J2000', observatory='ALMA'):

    # Load the measures and quanta tools
    me = casatools.measures()

    # Set direction and epoch
    position = me.direction(equinox, RA, DEC.replace(':', '.'))
    obstime = me.epoch('utc', datestring)

    # Define the radial velocity
    rvel = me.radialvelocity('LSRK', str(vel_tune * 1e-3)+'km/s')

    # Define the observing frame
    me.doframe(position)
    me.doframe(me.observatory(observatory))
    me.doframe(obstime)
    me.showframe()

    # Convert to the TOPO frame
    rvel_top = me.measure(rvel, 'TOPO')
    dopp = me.todoppler('RADIO', rvel_top)
    nu_top = me.tofrequency('TOPO', dopp,
                            me.frequency('rest', str(nu_rest / 1e9)+'GHz'))

    return nu_top['m0']['value']


def cube_to_fits(sky_image, fitsout, RA=0., DEC=0., restfreq=230.538e9,
                 bunit='JY/PIXEL', beam=None):

    # revert to proper formatting
    cube = np.rollaxis(np.fliplr(sky_image.data), -1)

    # extract coordinate information
    im_nfreq, im_ny, im_nx = cube.shape
    pixsize_x = np.abs(np.diff(sky_image.ra)[0])
    pixsize_y = np.abs(np.diff(sky_image.dec)[0])
    CRVAL3 = sky_image.freqs[0]
    if len(sky_image.freqs) > 1:
        CDELT3 = np.diff(sky_image.freqs)[0]
    else:
        CDELT3 = 1

    # generate the primary HDU
    hdu = fits.PrimaryHDU(np.float32(cube))

    # generate the header
    header = hdu.header
    header['EPOCH'] = 2000.
    header['EQUINOX'] = 2000.

    # Latitude and Longitude of the pole of the coordinate system.
    header['LATPOLE'] = -1.436915713634E+01
    header['LONPOLE'] = 180.

    # Define the RA coordinate
    header['CTYPE1'] = 'RA---SIN'
    header['CUNIT1'] = 'DEG'
    header['CDELT1'] = -pixsize_x / 3600
    header['CRPIX1'] = 0.5 * im_nx + 0.5
    header['CRVAL1'] = RA

    # Define the DEC coordinate
    header['CTYPE2'] = 'DEC--SIN'
    header['CUNIT2'] = 'DEG'
    header['CDELT2'] = pixsize_y / 3600
    header['CRPIX2'] = 0.5 * im_ny + 0.5
    header['CRVAL2'] = DEC

    # Define the frequency coordiante
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['CRPIX3'] = 1.
    header['CDELT3'] = CDELT3
    header['CRVAL3'] = CRVAL3

    header['SPECSYS'] = 'LSRK'
    header['VELREF'] = 257
    header['RESTFREQ'] = restfreq
    header['BSCALE'] = 1.
    header['BZERO'] = 0.
    header['BUNIT'] = bunit
    header['BTYPE'] = 'Intensity'

    if beam is not None:
        header['BMAJ'] = beam[0] / 3600
        header['BMIN'] = beam[1] / 3600
        header['BPA'] = beam[2]

    hdu.writeto(fitsout, overwrite=True)

    return None


def radmc_to_fits(path_to_image, fitsout, 
                  dist=150., RA=0., DEC=0., restfreq=230.538e9):

    # Image filename
    if path_to_image[-4:] == '.out':
        ifile = path_to_image
    else:
        ifile = path_to_image + 'image.out'

    # load the output into a proper cube array
    imagefile = open(ifile)
    iformat = imagefile.readline()
    im_nx, im_ny = imagefile.readline().split() #npixels along x and y axes
    im_nx, im_ny = int(im_nx), int(im_ny)
    nlam = int(imagefile.readline())

    pixsize_x, pixsize_y = imagefile.readline().split() #pixel sizes in cm 
    pixsize_x = float(pixsize_x)
    pixsize_y = float(pixsize_y)

    imvals = ascii.read(ifile, format='fast_csv',
                        guess=False, data_start=4,
                        fast_reader={'use_fast_converter':True})['1']
    lams = imvals[:nlam]

    # erg cm^-2 s^-1 Hz^-1 str^-1 --> Jy / pixel
    cube = np.reshape(imvals[nlam:],[nlam, im_ny, im_nx])
    cube *= 1e23 * pixsize_x * pixsize_y / (dist * 1e2 * sc.parsec)**2

    # Pack the cube into a vis_sample SkyImage object and FITS file
    mod_data = np.rollaxis(cube, 0, 3)
    mod_ra  = pixsize_x * (np.arange(im_nx) - 0.5 * im_nx)
    mod_dec = pixsize_y * (np.arange(im_ny) - 0.5 * im_ny)
    freq = sc.c / (lams * 1e-6)

    skyim = SkyImage(mod_data, mod_ra, mod_dec, freq, None)
    foo = cube_to_fits(skyim, fitsout, RA, DEC, restfreq=restfreq)

    return



""" Tools for analyzing posteriors for ensemble sampler MCMC """
""" some of this is copied from emcee """
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorr function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

# Following @fardal suggestion on emcee github
def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]



def load_posteriors(outfile, autocorr=True, prune_outliers=False, dev_lev=2,
                    maxtau=50, cutfact=10, burnfact=10, thinfact=1,
                    return_full=False, return_probs=True):

    # load the backend file
    reader = emcee.backends.HDFBackend(outfile)

    # get the full samples
    all_samples = reader.get_chain(discard=0, flat=False)

    # the log-posterior and log-prior values
    log_post = reader.get_log_prob(discard=0, flat=False)
    log_prior = reader.get_blobs(discard=0, flat=False)

    if prune_outliers:
        # all_samples shape
        nstep, nwalk, ndim = all_samples.shape

        # find outlier walkers
        ncut = round(nstep / cutfactor)
        dev_ = (np.median(log_prob[ncut:,:], axis=0) - \
                np.median(log_prob[ncut:,:])) / np.std(log_prob[ncut:,:])
        out_ix = np.where(np.abs(dev_) >= dev_lev)

        # remove them
        all_samples = np.delete(all_samples, out_ix, axis=1)
        log_post = np.delete(log_post, out_ix, axis=1)
        log_prior = np.delete(log_prior, out_ix, axis=1)

    if return_full:
        # return the posterior samples without subsequent reduction
        if return_probs:
            return all_samples, log_post, log_prior
        else:
            return all_samples
    else:
        # compute autocorrelation times (or assign them)
        if autocorr:
            tau_ = np.array([autocorr_new(all_samples[:-1,:,ix].T)
                             for ix in range(all_samples.shape[-1])])
            tau = np.min([tau_.max(), maxtau])
        else:
            tau = 1. * maxtau

        # burn and thin
        nburn, nthin = round(burnfact * tau), round(thinfact * tau)
        out_ = all_samples[nburn::nthin,:,:]
        s = list(out_.shape[1:])
        s[0] = np.prod(out_.shape[:2])
        flat_chain = out_.reshape(s)

        # output
        if return_probs:
            _log_post = log_post[nburn::nthin,:]
            _log_prior = log_prior[nburn::nthin,:]
            flat_log_post = _log_post.reshape(np.prod(_log_post.shape))
            flat_log_prior = _log_prior.reshape(np.prod(_log_prior.shape))
            return flat_chain, flat_log_post, flat_log_prior
        else:
            return flat_chain


def autocorr_evol_plot(outfile, prune_outliers=False, dev_lev=2, Nstep=5):

    # load chains
    all_samples = load_posteriors(outfile, prune_outliers=prune_outliers,
                                  dev_lev=dev_lev, return_full=True, 
                                  return_probs=False)

    # compute autocorrelation time every Nstep steps
    Nmax = all_samples.shape[0]
    if (Nmax > Nstep):
        tau_ix = np.empty(int(Nmax / Nstep))
        ix = np.empty(int(Nmax / Nstep))
        for i in range(len(tau_ix)):
            nn = (i + 1) * Nstep
            ix[i] = nn
            tau = emcee.autocorr.integrated_time(all_samples[:nn,:,:], tol=0)
            tau_ix[i] = np.nanmean(tau)
    tau_ix = np.nan_to_num(tau_ix)

    fig = plt.figure(constrained_layout=True)
    plt.plot(ix, tau_ix, '-o')
    plt.xlabel('steps')
    plt.ylabel('autocorr time (steps)')
    plt.xlim([0, Nmax])
    plt.ylim([0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min())])
    fig.savefig('autocorr_evol.png')
    fig.clf()


def trace_plot(outfile, ncols=3, figsize=(10, 11), prune_outliers=False, 
               dev_lev=2, labels=None, show_probs=True):

    # load chains
    inps = load_posteriors(outfile, prune_outliers=prune_outliers,
                           dev_lev=dev_lev, return_full=True,
                           return_probs=show_probs)
    if show_probs:
        all_, post, pri = inps
        all_samples = np.dstack((all_, post, pri))
        labels += ['log post', 'log prior']
    else:
        all_samples = 1. * inps

    # make the figure object
    nstep, nwalk, ndim = all_samples.shape
    nrows = int(np.ceil(ndim / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,
                            constrained_layout=True)

    # loop over the dimensions and plot the traces
    for idim in range(ndim):
        # choose panel
        ax = axs[np.floor_divide(idim, ncols), idim % ncols]

        for iw in range(nwalk):
            # plot the trace
            ax.plot(np.arange(nstep), all_samples[:,iw,idim], 
                    color='C0', alpha=0.05)

        # boundaries
        ax.set_xlim([0, nstep])
        if labels is not None:
            ax.set_ylabel(labels[idim])

    fig.savefig('traces.png')
    fig.clf()



